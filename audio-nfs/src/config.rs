use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::PathBuf;

#[derive(Debug, Deserialize, Clone)]
pub struct Config {
    pub server: ServerConfig,
    pub formats: FormatsConfig,
    pub samplerates: SampleRatesConfig,
    pub passthrough: PassthroughConfig,
    pub ffmpeg: FfmpegConfig,
    pub cache: CacheConfig,
}

#[derive(Debug, Deserialize, Clone)]
pub struct ServerConfig {
    pub port: u16,
    pub source_dir: PathBuf,
}

#[derive(Debug, Deserialize, Clone)]
pub struct FormatsConfig {
    pub enabled: Vec<String>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct SampleRatesConfig {
    pub enabled: Vec<u32>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct PassthroughConfig {
    pub enabled: bool,
}

#[derive(Debug, Deserialize, Clone)]
pub struct FfmpegConfig {
    pub use_library: bool,
    pub binary: PathBuf,
    pub decode_and_discard_threshold: u64,
}

#[derive(Debug, Deserialize, Clone)]
pub struct CacheConfig {
    pub max_open_handles: usize,
    pub chunk_size_bytes: usize,
}

impl Config {
    pub fn from_file(path: &std::path::Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading config {}", path.display()))?;
        toml::from_str(&text).context("parsing config TOML")
    }
}
