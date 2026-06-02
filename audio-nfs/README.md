# audio-nfs

Userspace NFSv3 server that serves a source audio directory with on-the-fly transcoding via `ffmpeg-next` (libav bindings).

## Path convention

```
# Real file (passthrough, if enabled)
/mount/subdir/interview.mp3

# Transcoded views — the file acts as a virtual directory
/mount/subdir/interview.mp3/wav/16000
/mount/subdir/interview.mp3/flac/44100
/mount/subdir/interview.mp3/opus/24000

# Trimmed views (sample offsets at output SR)
/mount/subdir/interview.mp3/wav/16000/0-480000
/mount/subdir/interview.mp3/flac/44100/88200-176400
```

Trim ranges are in **output-sample-rate samples** (same convention as torchaudio).

## Build

Requires libavformat/libavcodec/libavutil/libswresample dev headers.
Use the included dev container for a pre-configured environment:

```
# VS Code: Reopen in Container
# Or manually:
docker build -t audio-nfs-dev .devcontainer/
docker run --rm -it -v $(pwd):/workspace audio-nfs-dev bash
cd /workspace && cargo build --release
```

## Configuration

Copy and edit `config.toml`:

```toml
[server]
port = 11111
source_dir = "/data/audio"

[formats]
enabled = ["wav", "flac", "opus"]

[samplerates]
enabled = [8000, 16000, 22050, 44100, 48000]

[passthrough]
enabled = true

[ffmpeg]
use_library = true
binary = "/usr/bin/ffmpeg"
decode_and_discard_threshold = 8192

[cache]
max_open_handles = 64
chunk_size_bytes = 65536
```

## Running

```bash
./target/release/audio-nfs config.toml
```

## Mounting (Linux)

```bash
sudo mkdir -p /mnt/audio
sudo mount -t nfs -o port=11111,mountport=11111,nfsvers=3,tcp,nolock \
    localhost:/ /mnt/audio
```

## Known limitations

- **FLAC / Opus `getattr` size**: true byte size is unknown before encoding. The server reports a sentinel value (`i64::MAX / 2`). Clients will discover EOF normally. Use WAV when size-accuracy matters.
- **Transcode buffering**: v1 buffers the entire transcode in memory before serving reads. Large files or many concurrent clients will use significant RAM.
- **Concurrent seeks**: two simultaneous `read()` calls with different offsets on the same handle re-use a single cached buffer. Non-sequential access is handled by re-reading from the cache (already buffered).

## Build order / status

- [x] Config parsing and handle/ID map
- [x] Static directory tree — NFS mount + `ls`
- [x] Passthrough reads for real files
- [x] Virtual dir enumeration (`format/` and `sr/` children)
- [x] WAV transcode via ffmpeg-next (full file and trimmed)
- [x] FLAC and Opus via subprocess fallback (subprocess path)
- [ ] Sample-accurate seeking via libav (currently decodes full stream then discards)
- [ ] Proper streaming with AVIOContext ring buffer
- [ ] Native FLAC/Opus encoding via libav (no subprocess)
- [ ] Subprocess ffmpeg config flag wired into transcode dispatch
