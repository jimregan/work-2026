mod config;
mod fs;
mod handle;
mod transcode;

use anyhow::{Context, Result};
use nfsserve::tcp::{NFSTcp, NFSTcpListener};
use std::path::PathBuf;
use tracing::info;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("audio_nfs=debug".parse()?),
        )
        .init();

    let config_path = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("config.toml"));

    let cfg = config::Config::from_file(&config_path)
        .with_context(|| format!("loading {}", config_path.display()))?;

    info!(
        "serving {} on port {}",
        cfg.server.source_dir.display(),
        cfg.server.port
    );
    info!("enabled formats: {:?}", cfg.formats.enabled);
    info!("enabled sample rates: {:?}", cfg.samplerates.enabled);

    let port = cfg.server.port;
    let nfs = fs::AudioNFS::new(cfg);

    let listener = NFSTcpListener::bind(&format!("0.0.0.0:{port}"), nfs)
        .await
        .with_context(|| format!("bind port {port}"))?;

    listener.handle_forever().await?;
    Ok(())
}
