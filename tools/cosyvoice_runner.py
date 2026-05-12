#!/usr/bin/env python3
"""Small CLI used by Pixelle-Video to run local CosyVoice inference."""

import argparse
import subprocess
import sys
from pathlib import Path

import torchaudio


def _validate_mode_model(mode: str, model_name: str) -> None:
    model_lower = model_name.lower()
    if mode == "instruct" and "instruct" not in model_lower:
        raise ValueError("CosyVoice instruct mode requires an Instruct model, for example iic/CosyVoice-300M-Instruct")
    if mode == "sft" and "sft" not in model_lower:
        raise ValueError("CosyVoice preset speaker mode requires an SFT model, for example iic/CosyVoice-300M-SFT")
    if mode == "zero_shot" and ("sft" in model_lower or "instruct" in model_lower):
        raise ValueError("CosyVoice zero-shot mode requires a base CosyVoice model, not SFT/Instruct")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-dir", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="iic/CosyVoice-300M-SFT")
    parser.add_argument("--mode", default="sft", choices=["sft", "instruct", "zero_shot"])
    parser.add_argument("--speaker", default="中文女")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--instruct", default="")
    parser.add_argument("--prompt-text", default="")
    parser.add_argument("--prompt-audio", default="")
    args = parser.parse_args()

    repo_dir = Path(args.repo_dir).resolve()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(repo_dir))
    sys.path.insert(0, str(repo_dir / "third_party" / "Matcha-TTS"))

    from cosyvoice.cli.cosyvoice import AutoModel
    from modelscope import snapshot_download

    model_name = args.model
    _validate_mode_model(args.mode, model_name)
    if "/" in model_name:
        model_dir = repo_dir / "pretrained_models" / model_name.split("/")[-1]
        if not model_dir.exists():
            snapshot_download(model_name, local_dir=str(model_dir))
    else:
        model_dir = Path(model_name).expanduser().resolve()

    cosyvoice = AutoModel(model_dir=str(model_dir))
    speaker = args.speaker
    speakers = cosyvoice.list_available_spks()
    if speakers and speaker not in speakers:
        speaker = speakers[0]

    tmp_wav = output.with_suffix(".cosyvoice.tmp.wav")
    if args.mode == "instruct":
        instruct = args.instruct or "用自然、专业、有停顿的语气朗读。"
        result = next(cosyvoice.inference_instruct(args.text, speaker, instruct, speed=args.speed))
    elif args.mode == "zero_shot":
        if not args.prompt_text or not args.prompt_audio:
            raise ValueError("zero_shot mode requires --prompt-text and --prompt-audio")
        result = next(cosyvoice.inference_zero_shot(args.text, args.prompt_text, args.prompt_audio, speed=args.speed))
    else:
        result = next(cosyvoice.inference_sft(args.text, speaker, stream=False, speed=args.speed))
    torchaudio.save(str(tmp_wav), result["tts_speech"], cosyvoice.sample_rate)

    if output.suffix.lower() == ".wav":
        tmp_wav.replace(output)
    else:
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-y",
                "-i",
                str(tmp_wav),
                "-c:a",
                "libmp3lame",
                "-q:a",
                "4",
                str(output),
            ],
            check=True,
        )
        tmp_wav.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
