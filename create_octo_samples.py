"""
Generate synthetic "Octo" wake word samples using Piper TTS.

Creates the directory structure expected by openWakeWord train.py:
  my_custom_model/octo/
    positive_train/   - training positive clips (~3000)
    positive_test/    - validation positive clips (~500)
    negative_train/   - training negative clips (~2000)
    negative_test/    - validation negative clips (~500)

Usage:
    python generate_samples.py
"""

import os
import asyncio
import wave
import random
import struct
import numpy as np
from pathlib import Path
from piper.voice import PiperVoice

# ── Config ──────────────────────────────────────────────────────────────────

MODELS_DIR = Path.home() / ".openocto" / "models" / "piper"
OUTPUT_DIR = Path("./my_custom_model/octo")

# Positive phrases — variations of the wake word
POSITIVE_PHRASES = [
    "Octo",
    "Hey Octo",
    "Ok Octo",
    "Octo please",
    "Octo!",
    "Hey Octo!",
]

# Negative phrases — phonetically similar, should NOT trigger
NEGATIVE_PHRASES = [
    "Otto",
    "auto",
    "audio",
    "October",
    "octet",
    "actor",
    "occupy",
    "option",
    "optical",
    "okay",
    "open",
    "awful",
    "offer",
    "often",
    "autumn",
    "octave",
    "oxygen",
]

# Available piper voice models
VOICE_MODELS = [
    MODELS_DIR / "en_US-amy-medium.onnx",
    MODELS_DIR / "en_US-lessac-high.onnx",
]

N_POSITIVE_TRAIN = 3000
N_POSITIVE_TEST  = 500
N_NEGATIVE_TRAIN = 2000
N_NEGATIVE_TEST  = 500

SAMPLE_RATE = 16000  # required by openWakeWord

# ── Helpers ──────────────────────────────────────────────────────────────────

def resample_to_16k(audio: np.ndarray, source_rate: int) -> np.ndarray:
    """Simple linear resample to 16kHz."""
    if source_rate == SAMPLE_RATE:
        return audio
    ratio = SAMPLE_RATE / source_rate
    n_out = int(len(audio) * ratio)
    indices = np.linspace(0, len(audio) - 1, n_out)
    return np.interp(indices, np.arange(len(audio)), audio).astype(np.int16)


def save_wav(path: Path, audio: np.ndarray, rate: int = SAMPLE_RATE):
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(rate)
        wf.writeframes(audio.tobytes())


def synthesize(voice: PiperVoice, text: str, length_scale: float = 1.0) -> np.ndarray:
    """Synthesize text → 16-bit int16 numpy array at 16kHz."""
    from piper.config import SynthesisConfig
    syn_config = SynthesisConfig(length_scale=length_scale)
    chunks = list(voice.synthesize(text, syn_config=syn_config))

    if not chunks:
        return np.array([], dtype=np.int16)

    audio = np.concatenate([c.audio_int16_array for c in chunks])
    source_rate = chunks[0].sample_rate  # typically 22050
    return resample_to_16k(audio, source_rate)


def load_voices() -> list[PiperVoice]:
    voices = []
    for model_path in VOICE_MODELS:
        if model_path.exists():
            try:
                voices.append(PiperVoice.load(str(model_path)))
                print(f"  ✓ Loaded voice: {model_path.name}")
            except Exception as e:
                print(f"  ✗ Failed to load {model_path.name}: {e}")
        else:
            print(f"  ⚠ Voice model not found: {model_path}")
    return voices


def generate_clips(
    voices: list[PiperVoice],
    phrases: list[str],
    output_dir: Path,
    n_samples: int,
    label: str,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = len(list(output_dir.glob("*.wav")))
    if existing >= n_samples:
        print(f"  ✓ {label}: {existing} clips already exist, skipping")
        return

    needed = n_samples - existing
    print(f"  Generating {needed} clips for {label}...")

    # Speed variations: slightly faster / slower to increase diversity
    length_scales = [0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2]

    generated = 0
    idx = existing
    while generated < needed:
        phrase = random.choice(phrases)
        voice  = random.choice(voices)
        scale  = random.choice(length_scales)

        try:
            audio = synthesize(voice, phrase, length_scale=scale)
            if len(audio) < 100:
                continue
            save_wav(output_dir / f"{idx:05d}.wav", audio)
            idx += 1
            generated += 1

            if generated % 100 == 0:
                print(f"    {generated}/{needed}")
        except Exception as e:
            print(f"    ⚠ Error: {e}")

    print(f"  ✓ {label}: {n_samples} clips ready")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("🐙 OpenOcto Wake Word — Sample Generator")
    print(f"   Output: {OUTPUT_DIR.resolve()}\n")

    print("Loading voice models...")
    voices = load_voices()
    if not voices:
        print("\n✗ No voice models found. Run: openocto setup")
        print(f"  Expected models in: {MODELS_DIR}")
        return

    print(f"\n✓ {len(voices)} voice(s) loaded\n")

    # Positive clips
    print("→ Generating POSITIVE clips (train)...")
    generate_clips(voices, POSITIVE_PHRASES, OUTPUT_DIR / "positive_train",
                   N_POSITIVE_TRAIN, "positive_train")

    print("→ Generating POSITIVE clips (test/val)...")
    generate_clips(voices, POSITIVE_PHRASES, OUTPUT_DIR / "positive_test",
                   N_POSITIVE_TEST, "positive_test")

    # Negative clips
    print("→ Generating NEGATIVE clips (train)...")
    generate_clips(voices, NEGATIVE_PHRASES, OUTPUT_DIR / "negative_train",
                   N_NEGATIVE_TRAIN, "negative_train")

    print("→ Generating NEGATIVE clips (test/val)...")
    generate_clips(voices, NEGATIVE_PHRASES, OUTPUT_DIR / "negative_test",
                   N_NEGATIVE_TEST, "negative_test")

    print("\n✅ Done! Next steps:")
    print("   1. Download datasets:  python download_data.py")
    print("   2. Augment clips:      python openwakeword/train.py --training_config octo.yaml --augment_clips")
    print("   3. Train model:        python openwakeword/train.py --training_config octo.yaml --train_model")


if __name__ == "__main__":
    main()
