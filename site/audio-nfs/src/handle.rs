use dashmap::DashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

/// Canonical representation of every node in the virtual tree.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum NodeKey {
    RealDir(PathBuf),
    RealFile(PathBuf),
    VirtualFormatDir {
        source: PathBuf,
        format: String,
    },
    VirtualSrDir {
        source: PathBuf,
        format: String,
        sr: u32,
    },
    /// Exists only as an intermediate lookup level; leaf reads go to TranscodeFile.
    VirtualTrimDir {
        source: PathBuf,
        format: String,
        sr: u32,
        start: u64,
        end: u64,
    },
    TranscodeFile {
        source: PathBuf,
        format: String,
        sr: u32,
        start: Option<u64>,
        end: Option<u64>,
    },
}

pub struct HandleMap {
    counter: AtomicU64,
    id_to_key: DashMap<u64, NodeKey>,
    key_to_id: DashMap<NodeKey, u64>,
}

impl HandleMap {
    pub fn new() -> Self {
        Self {
            counter: AtomicU64::new(2), // 1 is reserved for root
            id_to_key: DashMap::new(),
            key_to_id: DashMap::new(),
        }
    }

    pub fn get_or_insert(&self, key: NodeKey) -> u64 {
        if let Some(id) = self.key_to_id.get(&key) {
            return *id;
        }
        let id = self.counter.fetch_add(1, Ordering::Relaxed);
        self.id_to_key.insert(id, key.clone());
        self.key_to_id.insert(key, id);
        id
    }

    pub fn get_key(&self, id: u64) -> Option<NodeKey> {
        self.id_to_key.get(&id).map(|r| r.clone())
    }
}

impl Default for HandleMap {
    fn default() -> Self {
        Self::new()
    }
}

/// Parse a `start-end` trim component, e.g. `"0-480000"`.
pub fn parse_trim(s: &str) -> Option<(u64, u64)> {
    let (a, b) = s.split_once('-')?;
    let start = a.parse().ok()?;
    let end = b.parse().ok()?;
    Some((start, end))
}
