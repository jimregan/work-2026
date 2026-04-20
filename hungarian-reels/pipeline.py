#!/usr/bin/env python3
"""
Hungarian Reel Analyzer
Analyzes Hungarian Instagram Reels for language learning using local Ollama models.

Usage:
    OLLAMA_HOST=http://your-server:11434 python3 pipeline.py <instagram_reel_url>
"""

import sys
import os
import subprocess
import tempfile
import base64
from pathlib import Path


def get_client():
    try:
        import ollama
    except ImportError:
        print("Error: ollama package not installed. Run: pip install ollama", file=sys.stderr)
        sys.exit(1)
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    return ollama.Client(host=host)


def download_reel(url: str, work_dir: Path) -> Path:
    original_dir = Path.cwd()
    os.chdir(work_dir)
    try:
        subprocess.run(
            [
                "instaloader",
                "--no-metadata-json",
                "--no-compress-json",
                "--no-pictures",
                "--filename-pattern={shortcode}",
                url,
            ],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"instaloader failed: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        os.chdir(original_dir)

    videos = list(work_dir.glob("*.mp4"))
    if not videos:
        print("No video file found after download.", file=sys.stderr)
        sys.exit(1)
    return videos[0]


def extract_audio(video_path: Path, output_path: Path):
    subprocess.run(
        [
            "ffmpeg", "-i", str(video_path),
            "-vn", "-acodec", "pcm_s16le",
            "-ar", "16000", "-ac", "1",
            str(output_path), "-y", "-loglevel", "error",
        ],
        check=True,
    )


def extract_frames(video_path: Path, frames_dir: Path) -> list:
    frames_dir.mkdir(exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-i", str(video_path),
            "-vf", "fps=1",
            str(frames_dir / "frame_%04d.jpg"),
            "-y", "-loglevel", "error",
        ],
        check=True,
    )
    return sorted(frames_dir.glob("frame_*.jpg"))


def b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def transcribe_audio(audio_path: Path, client) -> str:
    print("  Transcribing audio with hu-ear...", file=sys.stderr)
    response = client.generate(
        model="hu-ear",
        prompt="Transcribe this Hungarian audio accurately. Return only the transcription.",
        images=[b64(audio_path)],
    )
    return response["response"]


def extract_ocr(frames: list, client) -> str:
    if not frames:
        return "(no frames extracted)"
    frames = frames[:30]
    print(f"  Extracting on-screen text with hu-eye ({len(frames)} frames)...", file=sys.stderr)
    response = client.generate(
        model="hu-eye",
        prompt="Extract all visible text from these video frames. List each unique piece of text exactly as it appears.",
        images=[b64(f) for f in frames],
    )
    return response["response"]


def coordinate(transcript: str, ocr_text: str, url: str, client) -> str:
    print("  Coordinating and translating with hu-brain...", file=sys.stderr)
    prompt = f"""Audio Transcript:
{transcript}

OCR Text from frames:
{ocr_text}

The Reel URL is: {url}

Produce the Markdown output in the exact format specified in your system prompt, with [Reel]({url}) at the top."""

    response = client.chat(
        model="hu-brain",
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"]


def main():
    if len(sys.argv) < 2:
        print("Usage: pipeline.py <instagram_reel_url>", file=sys.stderr)
        sys.exit(1)

    url = sys.argv[1]
    client = get_client()

    with tempfile.TemporaryDirectory() as tmpdir:
        work_dir = Path(tmpdir)

        print("Downloading reel...", file=sys.stderr)
        video_path = download_reel(url, work_dir)

        print("Extracting audio and frames...", file=sys.stderr)
        audio_path = work_dir / "audio.wav"
        extract_audio(video_path, audio_path)
        frames = extract_frames(video_path, work_dir / "frames")

        transcript = transcribe_audio(audio_path, client)
        ocr_text = extract_ocr(frames, client)
        result = coordinate(transcript, ocr_text, url, client)

        print(result)


if __name__ == "__main__":
    main()
