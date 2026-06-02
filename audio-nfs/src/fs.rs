use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Result;
use nfsserve::nfs::*;
use nfsserve::vfs::{DirEntry, NFSFileSystem, ReadDirResult, VFSCapabilities};
use tokio::sync::RwLock;
use tracing::{debug, warn};

use crate::config::Config;
use crate::handle::{parse_trim, HandleMap, NodeKey};
use crate::transcode::{self, UNKNOWN_SIZE_SENTINEL};

const ROOT_ID: fileid3 = 1;

pub struct AudioNFS {
    pub cfg: Arc<Config>,
    pub handles: Arc<HandleMap>,
    /// Simple in-memory cache: handle id → fully transcoded bytes.
    /// For v1 we buffer the whole transcode; proper streaming is future work.
    transcode_cache: Arc<RwLock<lru::LruCache<u64, Arc<Vec<u8>>>>>,
}

impl AudioNFS {
    pub fn new(cfg: Config) -> Self {
        use std::num::NonZeroUsize;
        let cap = NonZeroUsize::new(cfg.cache.max_open_handles).unwrap();
        Self {
            cfg: Arc::new(cfg),
            handles: Arc::new(HandleMap::new()),
            transcode_cache: Arc::new(RwLock::new(lru::LruCache::new(cap))),
        }
    }

    fn source_dir(&self) -> &Path {
        &self.cfg.server.source_dir
    }

    fn is_audio(&self, path: &Path) -> bool {
        let Some(ext) = path.extension().and_then(|e| e.to_str()) else {
            return false;
        };
        matches!(
            ext.to_ascii_lowercase().as_str(),
            "mp3" | "wav" | "flac" | "ogg" | "opus" | "aac" | "m4a" | "wv" | "ape" | "aiff"
        )
    }

    fn real_path(&self, node: &NodeKey) -> Option<PathBuf> {
        match node {
            NodeKey::RealDir(p) | NodeKey::RealFile(p) => Some(p.clone()),
            NodeKey::VirtualFormatDir { source, .. }
            | NodeKey::VirtualSrDir { source, .. }
            | NodeKey::VirtualTrimDir { source, .. }
            | NodeKey::TranscodeFile { source, .. } => Some(source.clone()),
        }
    }

    fn metadata_to_fattr(&self, id: fileid3, meta: &std::fs::Metadata) -> fattr3 {
        let mtime = meta
            .modified()
            .unwrap_or(UNIX_EPOCH)
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default();
        fattr3 {
            ftype: ftype3::NF3REG,
            mode: 0o444,
            nlink: 1,
            uid: 0,
            gid: 0,
            size: meta.len(),
            used: meta.len(),
            rdev: specdata3 { specdata1: 0, specdata2: 0 },
            fsid: 0,
            fileid: id,
            atime: nfstime3 { seconds: mtime.as_secs() as u32, nseconds: mtime.subsec_nanos() },
            mtime: nfstime3 { seconds: mtime.as_secs() as u32, nseconds: mtime.subsec_nanos() },
            ctime: nfstime3 { seconds: mtime.as_secs() as u32, nseconds: mtime.subsec_nanos() },
        }
    }

    fn dir_fattr(&self, id: fileid3) -> fattr3 {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default();
        fattr3 {
            ftype: ftype3::NF3DIR,
            mode: 0o555,
            nlink: 2,
            uid: 0,
            gid: 0,
            size: 0,
            used: 0,
            rdev: specdata3 { specdata1: 0, specdata2: 0 },
            fsid: 0,
            fileid: id,
            atime: nfstime3 { seconds: now.as_secs() as u32, nseconds: 0 },
            mtime: nfstime3 { seconds: now.as_secs() as u32, nseconds: 0 },
            ctime: nfstime3 { seconds: now.as_secs() as u32, nseconds: 0 },
        }
    }

    async fn get_or_transcode(&self, id: fileid3, key: &NodeKey) -> Result<Arc<Vec<u8>>, nfsstat3> {
        {
            let cache = self.transcode_cache.read().await;
            // LruCache peek without mut — we need write for get
        }
        let mut cache = self.transcode_cache.write().await;
        if let Some(bytes) = cache.get(&id) {
            return Ok(bytes.clone());
        }
        drop(cache);

        let (source, format, sr, start, end) = match key {
            NodeKey::TranscodeFile { source, format, sr, start, end } => {
                (source.clone(), format.clone(), *sr, *start, *end)
            }
            _ => return Err(nfsstat3::NFS3ERR_INVAL),
        };

        let bytes = tokio::task::spawn_blocking(move || {
            transcode::transcode_to_bytes(&source, &format, sr, start, end)
        })
        .await
        .map_err(|_| nfsstat3::NFS3ERR_IO)?
        .map_err(|e| {
            warn!("transcode error: {e}");
            nfsstat3::NFS3ERR_IO
        })?;

        let arc = Arc::new(bytes);
        self.transcode_cache.write().await.put(id, arc.clone());
        Ok(arc)
    }
}

impl NFSFileSystem for AudioNFS {
    fn root_dir(&self) -> fileid3 {
        ROOT_ID
    }

    fn capabilities(&self) -> VFSCapabilities {
        VFSCapabilities::ReadOnly
    }

    async fn lookup(&self, dirid: fileid3, filename: &filename3) -> Result<fileid3, nfsstat3> {
        let name = std::str::from_utf8(filename).map_err(|_| nfsstat3::NFS3ERR_INVAL)?;
        debug!("lookup dirid={dirid} name={name}");

        // Resolve parent key
        let parent_key = if dirid == ROOT_ID {
            NodeKey::RealDir(self.source_dir().to_path_buf())
        } else {
            self.handles.get_key(dirid).ok_or(nfsstat3::NFS3ERR_STALE)?
        };

        let child_key = match &parent_key {
            NodeKey::RealDir(dir) => {
                let child_path = dir.join(name);
                if !child_path.exists() {
                    return Err(nfsstat3::NFS3ERR_NOENT);
                }
                if child_path.is_dir() {
                    NodeKey::RealDir(child_path)
                } else if self.is_audio(&child_path) {
                    // Audio file acts as both regular file and virtual dir root.
                    // lookup returns the RealFile node; the client can then
                    // look up format children inside it.
                    NodeKey::RealFile(child_path)
                } else {
                    NodeKey::RealFile(child_path)
                }
            }

            // Audio file acts as a directory: next component is format name
            NodeKey::RealFile(source) => {
                if !self.cfg.formats.enabled.contains(&name.to_string()) {
                    return Err(nfsstat3::NFS3ERR_NOENT);
                }
                NodeKey::VirtualFormatDir {
                    source: source.clone(),
                    format: name.to_string(),
                }
            }

            NodeKey::VirtualFormatDir { source, format } => {
                // Next component is sample rate
                let sr: u32 = name.parse().map_err(|_| nfsstat3::NFS3ERR_NOENT)?;
                if !self.cfg.samplerates.enabled.contains(&sr) {
                    return Err(nfsstat3::NFS3ERR_NOENT);
                }
                // SR dir is also the transcode leaf when there's no trim
                NodeKey::VirtualSrDir {
                    source: source.clone(),
                    format: format.clone(),
                    sr,
                }
            }

            NodeKey::VirtualSrDir { source, format, sr } => {
                // Next component is either a trim range ("0-480000") or nothing.
                // With a trim component present, it's the transcode leaf.
                if let Some((start, end)) = parse_trim(name) {
                    NodeKey::TranscodeFile {
                        source: source.clone(),
                        format: format.clone(),
                        sr: *sr,
                        start: Some(start),
                        end: Some(end),
                    }
                } else {
                    return Err(nfsstat3::NFS3ERR_NOENT);
                }
            }

            // No further children below transcode file or trim dir
            NodeKey::VirtualTrimDir { .. } | NodeKey::TranscodeFile { .. } => {
                return Err(nfsstat3::NFS3ERR_NOTDIR);
            }
        };

        Ok(self.handles.get_or_insert(child_key))
    }

    async fn getattr(&self, id: fileid3) -> Result<fattr3, nfsstat3> {
        debug!("getattr id={id}");

        if id == ROOT_ID {
            return Ok(self.dir_fattr(ROOT_ID));
        }

        let key = self.handles.get_key(id).ok_or(nfsstat3::NFS3ERR_STALE)?;

        match &key {
            NodeKey::RealDir(_) => Ok(self.dir_fattr(id)),

            NodeKey::RealFile(path) => {
                let meta = std::fs::metadata(path).map_err(|_| nfsstat3::NFS3ERR_IO)?;
                // Audio file appears as REG for passthrough; also as DIR for virtual children.
                // NFS clients see it as REG — virtual dir lookup works via lookup() calls.
                Ok(self.metadata_to_fattr(id, &meta))
            }

            NodeKey::VirtualFormatDir { .. } | NodeKey::VirtualSrDir { .. } | NodeKey::VirtualTrimDir { .. } => {
                Ok(self.dir_fattr(id))
            }

            NodeKey::TranscodeFile { source, format, sr, start, end } => {
                let size = if format == "wav" {
                    // Probe the source to compute exact WAV size
                    let source = source.clone();
                    let format = format.clone();
                    let sr = *sr;
                    let start = *start;
                    let end = *end;
                    let channels = 2u32; // conservative; transcode will use actual
                    tokio::task::spawn_blocking(move || {
                        transcode::wav_exact_size(&source, sr, channels, start, end)
                    })
                    .await
                    .map_err(|_| nfsstat3::NFS3ERR_IO)?
                    .unwrap_or(UNKNOWN_SIZE_SENTINEL)
                } else {
                    // FLAC/Opus: unknown size before encoding
                    UNKNOWN_SIZE_SENTINEL
                };

                let now = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default();
                Ok(fattr3 {
                    ftype: ftype3::NF3REG,
                    mode: 0o444,
                    nlink: 1,
                    uid: 0,
                    gid: 0,
                    size,
                    used: size,
                    rdev: specdata3 { specdata1: 0, specdata2: 0 },
                    fsid: 0,
                    fileid: id,
                    atime: nfstime3 { seconds: now.as_secs() as u32, nseconds: 0 },
                    mtime: nfstime3 { seconds: now.as_secs() as u32, nseconds: 0 },
                    ctime: nfstime3 { seconds: now.as_secs() as u32, nseconds: 0 },
                })
            }
        }
    }

    async fn read(
        &self,
        id: fileid3,
        offset: u64,
        count: u32,
    ) -> Result<(Vec<u8>, bool), nfsstat3> {
        debug!("read id={id} offset={offset} count={count}");

        let key = self.handles.get_key(id).ok_or(nfsstat3::NFS3ERR_STALE)?;

        match &key {
            NodeKey::RealFile(path) => {
                if !self.cfg.passthrough.enabled {
                    return Err(nfsstat3::NFS3ERR_PERM);
                }
                use std::io::{Read, Seek, SeekFrom};
                let mut f = std::fs::File::open(path).map_err(|_| nfsstat3::NFS3ERR_IO)?;
                let file_len = f.metadata().map_err(|_| nfsstat3::NFS3ERR_IO)?.len();
                if offset >= file_len {
                    return Ok((vec![], true));
                }
                f.seek(SeekFrom::Start(offset)).map_err(|_| nfsstat3::NFS3ERR_IO)?;
                let to_read = count.min((file_len - offset) as u32) as usize;
                let mut buf = vec![0u8; to_read];
                f.read_exact(&mut buf).map_err(|_| nfsstat3::NFS3ERR_IO)?;
                let eof = offset + to_read as u64 >= file_len;
                Ok((buf, eof))
            }

            NodeKey::TranscodeFile { .. } => {
                let bytes = self.get_or_transcode(id, &key).await?;
                if offset >= bytes.len() as u64 {
                    return Ok((vec![], true));
                }
                let end = (offset + count as u64).min(bytes.len() as u64) as usize;
                let slice = bytes[offset as usize..end].to_vec();
                let eof = end >= bytes.len();
                Ok((slice, eof))
            }

            _ => Err(nfsstat3::NFS3ERR_ISDIR),
        }
    }

    async fn readdir(
        &self,
        dirid: fileid3,
        start_after: fileid3,
        max_entries: usize,
    ) -> Result<ReadDirResult, nfsstat3> {
        debug!("readdir dirid={dirid} start_after={start_after}");

        let key = if dirid == ROOT_ID {
            NodeKey::RealDir(self.source_dir().to_path_buf())
        } else {
            self.handles.get_key(dirid).ok_or(nfsstat3::NFS3ERR_STALE)?
        };

        match &key {
            NodeKey::RealDir(dir) => {
                let entries = read_real_dir(dir, &self.handles, start_after, max_entries)?;
                Ok(entries)
            }

            NodeKey::RealFile(source) => {
                // Virtual dir: enumerate enabled format names
                let entries = self
                    .cfg
                    .formats
                    .enabled
                    .iter()
                    .filter_map(|fmt| {
                        let child = NodeKey::VirtualFormatDir {
                            source: source.clone(),
                            format: fmt.clone(),
                        };
                        let id = self.handles.get_or_insert(child);
                        if id <= start_after {
                            return None;
                        }
                        Some(DirEntry {
                            fileid: id,
                            name: fmt.as_bytes().to_vec(),
                            attr: Some(self.dir_fattr(id)),
                        })
                    })
                    .take(max_entries)
                    .collect::<Vec<_>>();
                let end = entries.len() < max_entries;
                Ok(ReadDirResult { entries, end })
            }

            NodeKey::VirtualFormatDir { source, format } => {
                let entries = self
                    .cfg
                    .samplerates
                    .enabled
                    .iter()
                    .filter_map(|sr| {
                        let child = NodeKey::VirtualSrDir {
                            source: source.clone(),
                            format: format.clone(),
                            sr: *sr,
                        };
                        let id = self.handles.get_or_insert(child);
                        if id <= start_after {
                            return None;
                        }
                        Some(DirEntry {
                            fileid: id,
                            name: sr.to_string().into_bytes(),
                            attr: Some(self.dir_fattr(id)),
                        })
                    })
                    .take(max_entries)
                    .collect::<Vec<_>>();
                let end = entries.len() < max_entries;
                Ok(ReadDirResult { entries, end })
            }

            // SR dir and trim dir have no enumerable children (trim is a leaf)
            NodeKey::VirtualSrDir { .. } | NodeKey::VirtualTrimDir { .. } => {
                Ok(ReadDirResult { entries: vec![], end: true })
            }

            NodeKey::TranscodeFile { .. } => Err(nfsstat3::NFS3ERR_NOTDIR),
        }
    }

    // Read-only server — all mutating ops return ROFS
    async fn setattr(&self, _id: fileid3, _setattr: sattr3) -> Result<fattr3, nfsstat3> {
        Err(nfsstat3::NFS3ERR_ROFS)
    }

    async fn write(
        &self,
        _id: fileid3,
        _offset: u64,
        _data: &[u8],
    ) -> Result<(fileid3, fattr3), nfsstat3> {
        Err(nfsstat3::NFS3ERR_ROFS)
    }

    async fn create(
        &self,
        _dirid: fileid3,
        _filename: &filename3,
        _attr: sattr3,
    ) -> Result<(fileid3, fattr3), nfsstat3> {
        Err(nfsstat3::NFS3ERR_ROFS)
    }

    async fn mkdir(
        &self,
        _parent: fileid3,
        _dirname: &filename3,
        _attr: sattr3,
    ) -> Result<(fileid3, fattr3), nfsstat3> {
        Err(nfsstat3::NFS3ERR_ROFS)
    }

    async fn remove(&self, _dirid: fileid3, _filename: &filename3) -> Result<(), nfsstat3> {
        Err(nfsstat3::NFS3ERR_ROFS)
    }

    async fn rename(
        &self,
        _from_dirid: fileid3,
        _from_filename: &filename3,
        _to_dirid: fileid3,
        _to_filename: &filename3,
    ) -> Result<(), nfsstat3> {
        Err(nfsstat3::NFS3ERR_ROFS)
    }

    async fn symlink(
        &self,
        _dirid: fileid3,
        _linkname: &filename3,
        _symlink: &nfspath3,
        _attr: sattr3,
    ) -> Result<(fileid3, fattr3), nfsstat3> {
        Err(nfsstat3::NFS3ERR_ROFS)
    }
}

fn read_real_dir(
    dir: &Path,
    handles: &HandleMap,
    start_after: fileid3,
    max_entries: usize,
) -> Result<ReadDirResult, nfsstat3> {
    let rd = std::fs::read_dir(dir).map_err(|_| nfsstat3::NFS3ERR_IO)?;

    let mut entries = Vec::new();
    for entry in rd.flatten() {
        let path = entry.path();
        let key = if path.is_dir() {
            NodeKey::RealDir(path.clone())
        } else {
            NodeKey::RealFile(path.clone())
        };
        let id = handles.get_or_insert(key);
        if id <= start_after {
            continue;
        }
        let name = entry.file_name().into_encoded_bytes();
        let ftype = if path.is_dir() { ftype3::NF3DIR } else { ftype3::NF3REG };
        let meta = entry.metadata().ok();
        let attr = meta.map(|m| fattr3 {
            ftype,
            mode: if path.is_dir() { 0o555 } else { 0o444 },
            nlink: 1,
            uid: 0,
            gid: 0,
            size: m.len(),
            used: m.len(),
            rdev: specdata3 { specdata1: 0, specdata2: 0 },
            fsid: 0,
            fileid: id,
            atime: nfstime3 { seconds: 0, nseconds: 0 },
            mtime: nfstime3 { seconds: 0, nseconds: 0 },
            ctime: nfstime3 { seconds: 0, nseconds: 0 },
        });
        entries.push(DirEntry { fileid: id, name, attr });
        if entries.len() >= max_entries {
            break;
        }
    }

    let end = entries.len() < max_entries;
    Ok(ReadDirResult { entries, end })
}
