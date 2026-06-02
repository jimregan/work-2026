use anyhow::{bail, Context, Result};
use std::path::Path;

/// All bytes of the transcoded stream for the given parameters.
///
/// Byte offset → sample seeking is handled here following torchaudio's pattern:
///   1. avformat_open_input + find best audio stream
///   2. av_seek_frame to keyframe before target sample
///   3. decode + discard samples until exact target offset
///   4. encode remaining samples into the output format
///
/// `start_sample` and `end_sample` are in output-SR samples (None = full file).
pub fn transcode_to_bytes(
    source: &Path,
    format: &str,
    output_sr: u32,
    start_sample: Option<u64>,
    end_sample: Option<u64>,
) -> Result<Vec<u8>> {
    use ffmpeg_next as ffmpeg;

    ffmpeg::init().context("ffmpeg init")?;

    let mut ictx = ffmpeg::format::input(source)
        .with_context(|| format!("open {}", source.display()))?;

    let stream_index = ictx
        .streams()
        .best(ffmpeg::media::Type::Audio)
        .context("no audio stream")?
        .index();

    let stream = ictx.stream(stream_index).unwrap();
    let codec_params = stream.parameters();
    let time_base = stream.time_base();

    let decoder_ctx = ffmpeg::codec::context::Context::from_parameters(codec_params)
        .context("decoder context")?;
    let mut decoder = decoder_ctx.decoder().audio().context("audio decoder")?;

    let input_sr = decoder.rate();
    let input_channels = decoder.channel_layout().channels();

    // Build resampler: input → output SR, output channel layout
    let out_layout = if input_channels == 1 {
        ffmpeg::channel_layout::ChannelLayout::MONO
    } else {
        ffmpeg::channel_layout::ChannelLayout::STEREO
    };
    let out_channels = out_layout.channels() as u32;
    let out_sample_fmt = match format {
        "wav" | "flac" => ffmpeg::format::Sample::I16(ffmpeg::format::sample::Type::Packed),
        "opus" => ffmpeg::format::Sample::F32(ffmpeg::format::sample::Type::Packed),
        _ => bail!("unsupported format: {format}"),
    };

    let mut resampler = ffmpeg::software::resampling::context::Context::get(
        decoder.format(),
        decoder.channel_layout(),
        input_sr,
        out_sample_fmt,
        out_layout,
        output_sr,
    )
    .context("resampler")?;

    let bytes_per_sample: u32 = match out_sample_fmt {
        ffmpeg::format::Sample::I16(_) => 2,
        ffmpeg::format::Sample::F32(_) => 4,
        _ => 2,
    };
    let frame_size = bytes_per_sample * out_channels;

    // Seek to keyframe before start_sample if needed
    let start_s = start_sample.unwrap_or(0);
    if start_s > 0 {
        // Convert output sample offset to input timestamp (in input stream time_base)
        let target_sec = start_s as f64 / output_sr as f64;
        let ts = (target_sec * time_base.1 as f64 / time_base.0 as f64) as i64;
        ictx.seek(ts, ..ts).context("seek")?;
    }

    let mut pcm_buf: Vec<u8> = Vec::new();
    let mut samples_written: u64 = 0;
    let mut samples_discarded: u64 = 0;

    let mut decoded = ffmpeg::frame::Audio::empty();
    let mut resampled = ffmpeg::frame::Audio::empty();

    'outer: for (stream, packet) in ictx.packets() {
        if stream.index() != stream_index {
            continue;
        }
        decoder.send_packet(&packet).ok();

        while decoder.receive_frame(&mut decoded).is_ok() {
            resampler
                .run(&decoded, &mut resampled)
                .context("resample")?;

            let data = resampled.data(0);
            let n_samples = data.len() / frame_size as usize;

            for i in 0..n_samples {
                if samples_discarded < start_s {
                    samples_discarded += 1;
                    continue;
                }
                if let Some(end) = end_sample {
                    if samples_written >= end - start_s {
                        break 'outer;
                    }
                }
                let offset = i * frame_size as usize;
                pcm_buf.extend_from_slice(&data[offset..offset + frame_size as usize]);
                samples_written += 1;
            }
        }
    }

    // Flush resampler
    while resampler.flush(&mut resampled).is_ok() {
        let data = resampled.data(0);
        let n_samples = data.len() / frame_size as usize;
        for i in 0..n_samples {
            if samples_discarded < start_s {
                samples_discarded += 1;
                continue;
            }
            if let Some(end) = end_sample {
                if samples_written >= end - start_s {
                    break;
                }
            }
            let offset = i * frame_size as usize;
            pcm_buf.extend_from_slice(&data[offset..offset + frame_size as usize]);
            samples_written += 1;
        }
    }

    wrap_in_container(&pcm_buf, format, output_sr, out_channels, bytes_per_sample)
}

fn wrap_in_container(
    pcm: &[u8],
    format: &str,
    sr: u32,
    channels: u32,
    bytes_per_sample: u32,
) -> Result<Vec<u8>> {
    match format {
        "wav" => Ok(encode_wav(pcm, sr, channels, bytes_per_sample)),
        "flac" => encode_via_subprocess(pcm, sr, channels, bytes_per_sample, "flac"),
        "opus" => encode_via_subprocess(pcm, sr, channels, bytes_per_sample, "opus"),
        _ => bail!("unsupported format: {format}"),
    }
}

fn encode_wav(pcm: &[u8], sr: u32, channels: u32, bytes_per_sample: u32) -> Vec<u8> {
    let data_len = pcm.len() as u32;
    let byte_rate = sr * channels * bytes_per_sample;
    let block_align = (channels * bytes_per_sample) as u16;
    let bits_per_sample = (bytes_per_sample * 8) as u16;

    let mut wav = Vec::with_capacity(44 + pcm.len());
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&(36 + data_len).to_le_bytes());
    wav.extend_from_slice(b"WAVE");
    wav.extend_from_slice(b"fmt ");
    wav.extend_from_slice(&16u32.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes()); // PCM
    wav.extend_from_slice(&(channels as u16).to_le_bytes());
    wav.extend_from_slice(&sr.to_le_bytes());
    wav.extend_from_slice(&byte_rate.to_le_bytes());
    wav.extend_from_slice(&block_align.to_le_bytes());
    wav.extend_from_slice(&bits_per_sample.to_le_bytes());
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&data_len.to_le_bytes());
    wav.extend_from_slice(pcm);
    wav
}

fn encode_via_subprocess(
    pcm: &[u8],
    sr: u32,
    channels: u32,
    bytes_per_sample: u32,
    out_fmt: &str,
) -> Result<Vec<u8>> {
    use std::io::Write;
    use std::process::{Command, Stdio};

    let sample_fmt = if bytes_per_sample == 2 { "s16le" } else { "f32le" };
    let codec = match out_fmt {
        "flac" => "flac",
        "opus" => "libopus",
        _ => bail!("unsupported: {out_fmt}"),
    };

    let mut child = Command::new("ffmpeg")
        .args([
            "-f", sample_fmt,
            "-ar", &sr.to_string(),
            "-ac", &channels.to_string(),
            "-i", "pipe:0",
            "-c:a", codec,
            "-f", out_fmt,
            "pipe:1",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .context("spawn ffmpeg")?;

    child.stdin.take().unwrap().write_all(pcm)?;
    let output = child.wait_with_output().context("ffmpeg wait")?;
    Ok(output.stdout)
}

/// Exact output byte size for WAV only. For FLAC/Opus returns None.
pub fn wav_exact_size(
    source: &Path,
    output_sr: u32,
    channels: u32,
    start_sample: Option<u64>,
    end_sample: Option<u64>,
) -> Result<u64> {
    use ffmpeg_next as ffmpeg;
    ffmpeg::init().ok();

    let ictx = ffmpeg::format::input(source)
        .with_context(|| format!("open {}", source.display()))?;

    let stream = ictx
        .streams()
        .best(ffmpeg::media::Type::Audio)
        .context("no audio stream")?;

    // Duration in source stream's time_base units
    let duration_ts = stream.duration();
    let tb = stream.time_base();
    let duration_sec = duration_ts as f64 * tb.0 as f64 / tb.1 as f64;
    let total_samples = (duration_sec * output_sr as f64) as u64;

    let start = start_sample.unwrap_or(0);
    let end = end_sample.unwrap_or(total_samples).min(total_samples);
    let n_samples = end.saturating_sub(start);

    // WAV: 44-byte header + PCM data (s16, so 2 bytes per sample per channel)
    let data_bytes = n_samples * channels as u64 * 2;
    Ok(44 + data_bytes)
}

/// Sentinel size reported for FLAC/Opus where true size is unknown before encoding.
pub const UNKNOWN_SIZE_SENTINEL: u64 = i64::MAX as u64 / 2;
