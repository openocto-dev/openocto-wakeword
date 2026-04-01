"""
Download datasets required for openWakeWord training.
Uses direct wget downloads to avoid datasets library version conflicts.

Usage:
    python download_data.py
"""

import os
import subprocess
import sys
from pathlib import Path

BASE_HF = "https://huggingface.co/datasets"


def run(cmd: str) -> int:
    return subprocess.call(cmd, shell=True)


def wget(url: str, out: str) -> bool:
    print(f"    ↓ {Path(out).name}")
    ret = run(f"wget -q --show-progress -O '{out}' '{url}'")
    return ret == 0


def check(path: Path, min_mb: float = 0) -> bool:
    if not path.exists():
        return False
    if min_mb and path.stat().st_size < min_mb * 1024 * 1024:
        return False
    return True


# ── 1. MIT Room Impulse Responses ────────────────────────────────────────────

def download_rirs():
    out = Path("./mit_rirs")
    if out.exists() and len(list(out.glob("*.wav"))) > 50:
        print(f"  ✓ MIT RIRs already ready ({len(list(out.glob('*.wav')))} files)")
        return

    out.mkdir(exist_ok=True)
    print("  Downloading MIT RIRs via soundfile...")

    try:
        import soundfile as sf
        import numpy as np
        import scipy.io.wavfile
    except ImportError:
        run(f"{sys.executable} -m pip install -q soundfile scipy")
        import soundfile as sf
        import numpy as np
        import scipy.io.wavfile

    # Download the parquet snapshot directly
    parquet_url = f"{BASE_HF}/davidscripka/MIT_environmental_impulse_responses/resolve/main/data/train-00000-of-00001.parquet"
    parquet_path = Path("./mit_rirs.parquet")
    if not check(parquet_path, 0.1):
        if not wget(parquet_url, str(parquet_path)):
            print("  ⚠ Failed to download MIT RIRs parquet, skipping")
            return

    try:
        import pandas as pd
        df = pd.read_parquet(parquet_path)
        for i, row in df.iterrows():
            audio_dict = row["audio"]
            audio = np.array(audio_dict["array"], dtype=np.float32)
            sr = audio_dict["sampling_rate"]
            name = Path(audio_dict["path"]).name if audio_dict.get("path") else f"rir_{i:04d}.wav"
            scipy.io.wavfile.write(out / name, sr, (audio * 32767).astype(np.int16))
        print(f"  ✓ MIT RIRs: {len(list(out.glob('*.wav')))} files")
        parquet_path.unlink(missing_ok=True)
    except Exception as e:
        print(f"  ⚠ MIT RIRs parquet failed: {e} — creating minimal synthetic RIRs")
        _make_synthetic_rirs(out)


def _make_synthetic_rirs(out: Path):
    """Create minimal synthetic room impulse responses as fallback."""
    import numpy as np
    import scipy.io.wavfile

    print("  Creating synthetic RIRs as fallback...")
    for i in range(20):
        sr = 16000
        dur = np.random.uniform(0.1, 0.5)
        n = int(sr * dur)
        rir = np.random.randn(n).astype(np.float32)
        rir *= np.exp(-np.linspace(0, 10, n))  # exponential decay
        rir = (rir / np.abs(rir).max() * 32767).astype(np.int16)
        scipy.io.wavfile.write(out / f"synthetic_rir_{i:03d}.wav", sr, rir)
    print(f"  ✓ Synthetic RIRs: {len(list(out.glob('*.wav')))} files")


# ── 2. AudioSet background noise ─────────────────────────────────────────────

def download_audioset():
    out = Path("./audioset_16k")
    if out.exists() and len(list(out.glob("*.wav"))) > 100:
        print(f"  ✓ AudioSet already ready ({len(list(out.glob('*.wav')))} files)")
        return

    out.mkdir(exist_ok=True)
    raw = Path("./audioset")
    raw.mkdir(exist_ok=True)

    fname = "bal_train09.tar"
    tar_path = raw / fname

    if not check(tar_path, 100):
        url = f"{BASE_HF}/agkphysics/AudioSet/resolve/main/data/{fname}"
        print(f"  Downloading AudioSet shard (~1GB)...")
        if not wget(url, str(tar_path)):
            print("  ⚠ Failed to download AudioSet, skipping")
            return

    print("  Extracting...")
    run(f"cd audioset && tar -xf {fname} 2>/dev/null")

    flac_files = list(Path("audioset/audio").glob("**/*.flac"))
    if not flac_files:
        print("  ⚠ No flac files found after extraction")
        return

    print(f"  Converting {len(flac_files)} files to 16kHz WAV...")
    try:
        import soundfile as sf
        import numpy as np
        import scipy.io.wavfile
        from scipy.signal import resample_poly
        from math import gcd

        for f in flac_files:
            try:
                audio, sr = sf.read(str(f))
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                if sr != 16000:
                    g = gcd(sr, 16000)
                    audio = resample_poly(audio, 16000 // g, sr // g)
                out_name = f.stem + ".wav"
                scipy.io.wavfile.write(out / out_name, 16000, (audio * 32767).astype(np.int16))
            except Exception:
                pass
        print(f"  ✓ AudioSet: {len(list(out.glob('*.wav')))} files")
    except ImportError:
        run(f"{sys.executable} -m pip install -q soundfile scipy")
        download_audioset()


# ── 3. FMA music ─────────────────────────────────────────────────────────────

def download_fma():
    out = Path("./fma")
    if out.exists() and len(list(out.glob("*.wav"))) > 50:
        print(f"  ✓ FMA already ready ({len(list(out.glob('*.wav')))} files)")
        return

    out.mkdir(exist_ok=True)
    print("  Downloading FMA small (~7GB compressed) — this takes a while...")
    print("  ⚠ Skipping FMA for now — AudioSet is sufficient for training.")
    print("  To add FMA later: https://huggingface.co/datasets/rudraml/fma")


# ── 4. Pre-computed features ─────────────────────────────────────────────────

def download_features():
    base = f"{BASE_HF}/davidscripka/openwakeword_features/resolve/main"

    train_path = Path("./openwakeword_features_ACAV100M_2000_hrs_16bit.npy")
    if check(train_path, 100):
        print(f"  ✓ ACAV100M features already exist ({train_path.stat().st_size // 1024 // 1024} MB)")
    else:
        print("  Downloading ACAV100M features (~4GB)...")
        wget(f"{base}/openwakeword_features_ACAV100M_2000_hrs_16bit.npy", str(train_path))
        if check(train_path, 100):
            print(f"  ✓ ACAV100M features: {train_path.stat().st_size // 1024 // 1024} MB")
        else:
            print("  ✗ Download failed or incomplete")

    val_path = Path("./validation_set_features.npy")
    if check(val_path, 10):
        print(f"  ✓ Validation features already exist ({val_path.stat().st_size // 1024 // 1024} MB)")
    else:
        print("  Downloading validation features (~200MB)...")
        wget(f"{base}/validation_set_features.npy", str(val_path))
        if check(val_path, 10):
            print(f"  ✓ Validation features: {val_path.stat().st_size // 1024 // 1024} MB")
        else:
            print("  ✗ Download failed")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.chdir(Path(__file__).parent)
    print("🐙 OpenOcto Wake Word — Dataset Downloader\n")

    print("→ [1/4] MIT Room Impulse Responses")
    download_rirs()

    print("\n→ [2/4] AudioSet background noise (~1GB)")
    download_audioset()

    print("\n→ [3/4] FMA music")
    download_fma()

    print("\n→ [4/4] Pre-computed openWakeWord features (~4.2GB total)")
    download_features()

    print("\n✅ Done!")
    print("\nNext:")
    print("  .venv/bin/python openwakeword/train.py --training_config octo.yaml --augment_clips")
    print("  .venv/bin/python openwakeword/train.py --training_config octo.yaml --train_model")


if __name__ == "__main__":
    main()
