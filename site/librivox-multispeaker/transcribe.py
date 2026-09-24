#!/usr/bin/env python3
import argparse, json, time, os
from pathlib import Path
import torch
import torch.multiprocessing as mp

from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration
from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor

AUDIO_EXTS = {".wav",".mp3",".flac",".mp4",".m4a",".webm",".ogg",".opus"}

def list_audio_files(audio_dir: str) -> list[str]:
    root = Path(audio_dir)
    return sorted(str(p) for p in root.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTS)

def transcribe_files(files, model_path, device, out_dir, attn_implementation,
                     max_new_tokens, temperature, top_p, num_beams):
    dtype = torch.float32 if device in ("cpu", "mps", "xpu") else torch.bfloat16

    processor = VibeVoiceASRProcessor.from_pretrained(model_path)
    model = VibeVoiceASRForConditionalGeneration.from_pretrained(
        model_path,
        dtype=dtype,
        device_map=device if device == "auto" else None,
        attn_implementation=attn_implementation,
        trust_remote_code=True,
    )
    if device != "auto":
        model = model.to(device)
    else:
        device = next(model.parameters()).device
    model.eval()

    do_sample = temperature > 0
    gen_cfg = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "num_beams": num_beams,
        "pad_token_id": processor.pad_id,
        "eos_token_id": processor.tokenizer.eos_token_id,
    }
    if do_sample:
        gen_cfg["temperature"] = temperature
        gen_cfg["top_p"] = top_p

    for f in files:
        out_path = out_dir / (Path(f).stem + ".json")
        if out_path.exists():
            print(f"⏭ skipping {out_path} (already exists)")
            continue

        inputs = processor(
            audio=f,
            sampling_rate=None,
            return_tensors="pt",
            add_generation_prompt=True,
        )
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        t0 = time.time()
        with torch.no_grad():
            output_ids = model.generate(**inputs, **gen_cfg)
        _ = time.time() - t0

        generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]
        raw_text = processor.decode(generated_ids, skip_special_tokens=True)

        try:
            segments = processor.post_process_transcription(raw_text)
        except Exception:
            segments = []

        out_list = []
        for seg in segments:
            item = {
                "Start": seg.get("start_time"),
                "End": seg.get("end_time"),
                "Content": seg.get("text", ""),
            }
            spk = seg.get("speaker_id", None)
            if spk is not None:
                item["Speaker"] = spk
            out_list.append(item)

        out_path.write_text(json.dumps(out_list, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"✅ [{device}] wrote {out_path} ({len(out_list)} segments)")

def worker(rank, gpu_ids, file_chunks, model_path, out_dir, attn_implementation,
           max_new_tokens, temperature, top_p, num_beams):
    device = f"cuda:{gpu_ids[rank]}"
    transcribe_files(file_chunks[rank], model_path, device, out_dir, attn_implementation,
                     max_new_tokens, temperature, top_p, num_beams)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--audio_dir", default="")
    ap.add_argument("--audio_files", nargs="*", default=[])
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                    choices=["cuda","cpu","mps","xpu","auto"])
    ap.add_argument("--gpu_ids", nargs="*", type=int, default=None,
                    help="CUDA device indices to use (e.g. --gpu_ids 0 1 2). "
                         "Overrides --device; one model instance per GPU.")
    ap.add_argument("--attn_implementation", default="sdpa",
                    choices=["flash_attention_2","sdpa","eager"])
    ap.add_argument("--max_new_tokens", type=int, default=32768)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--num_beams", type=int, default=1)
    args = ap.parse_args()

    files = []
    if args.audio_dir:
        files.extend(list_audio_files(args.audio_dir))
    files.extend(args.audio_files or [])
    if not files:
        raise SystemExit("No audio files found. Use --audio_dir or --audio_files.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.gpu_ids and len(args.gpu_ids) > 1:
        gpu_ids = args.gpu_ids
        n = len(gpu_ids)
        # distribute files round-robin across GPUs
        file_chunks = [files[i::n] for i in range(n)]
        mp.spawn(worker,
                 args=(gpu_ids, file_chunks, args.model_path, out_dir,
                       args.attn_implementation, args.max_new_tokens,
                       args.temperature, args.top_p, args.num_beams),
                 nprocs=n,
                 join=True)
    else:
        device = f"cuda:{args.gpu_ids[0]}" if args.gpu_ids else args.device
        transcribe_files(files, args.model_path, device, out_dir, args.attn_implementation,
                         args.max_new_tokens, args.temperature, args.top_p, args.num_beams)

if __name__ == "__main__":
    main()

