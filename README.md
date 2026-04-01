# openocto-wakeword

Train custom wake word models for [OpenOcto](https://openocto.dev) on **Apple Silicon (Mac M4/M3/M2/M1)** — no CUDA required.

Based on [openWakeWord](https://github.com/dscripka/openWakeWord) with patches and tooling to make training work on macOS.

## What's included

| File | Description |
|---|---|
| `create_octo_samples.py` | Generate synthetic voice clips via Piper TTS (replaces Linux-only piper-sample-generator) |
| `download_data.py` | Download training datasets without the `datasets` library (avoids dependency conflicts) |
| `octo.yaml` | Training config for the "Octo" wake word (Hey Octo / Hi Octo / Ok Octo) |
| `openwakeword/train.py` | Patched train.py: `num_workers=0` to fix macOS multiprocessing pickling error |

## Trained models

Pre-trained models are published to HuggingFace:

- **octo_v0.1** — responds to "Hey Octo", "Hi Octo", "Ok Octo"
  `https://huggingface.co/openocto-dev/openocto-models/resolve/main/octo.onnx`

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.11 (required — piper-tts is not compatible with 3.12+)
- ~25 GB free disk space for datasets

## Setup

```bash
# Clone this repo
git clone https://github.com/openocto-dev/openocto-wakeword.git
cd openocto-wakeword

# Create venv with Python 3.11
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install openwakeword==0.6.0 piper-tts==1.2.0
pip install torchinfo torchmetrics speechbrain==0.5.14
pip install audiomentations==0.33.0 torch-audiomentations==0.12.0
pip install acoustics pronouncing mutagen onnx
pip install torch==2.1.2 torchaudio==2.1.2
pip install scipy==1.11.4  # required — newer scipy breaks acoustics
```

## Train your own wake word

### Step 1 — Generate voice samples

Edit `octo.yaml` and set your target phrase, then run:

```bash
python create_octo_samples.py
```

This generates ~6000 synthetic WAV clips using Piper TTS voices. Clips are saved to `my_custom_model/<model_name>/`.

You need a Piper voice model in `~/.openocto/models/piper/` (e.g. `en_US-lessac-high.onnx`).
Download from [rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices).

### Step 2 — Download training datasets

```bash
python download_data.py
```

Downloads (~17 GB total):
- ACAV100M pre-computed features (16 GB) — background noise negatives
- Validation set features (176 MB)
- MIT RIRs — room impulse responses for augmentation

### Step 3 — Augment clips

```bash
python openwakeword/train.py --training_config octo.yaml --augment_clips
```

### Step 4 — Train model

```bash
python openwakeword/train.py --training_config octo.yaml --train_model
```

Training takes ~8 minutes on M4. Output: `my_custom_model/<model_name>.onnx`

## macOS-specific fixes applied

| Problem | Fix |
|---|---|
| `piper-sample-generator` is Linux-only | Replaced with `create_octo_samples.py` using Piper TTS directly |
| `torchaudio` 2.x requires `torchcodec` | Pinned to `torchaudio==2.1.2` |
| `torch.multiprocessing` can't pickle lambda on macOS | Set `num_workers=0` in DataLoader |
| `scipy 1.17` breaks `acoustics` | Pinned to `scipy==1.11.4` |
| `datasets` library conflicts with pyarrow | Rewrote `download_data.py` without `datasets` |

## Use with OpenOcto

After training, upload your `.onnx` to HuggingFace and add it to `model_downloader.py`:

```python
WAKE_WORD_MODELS = {
    "my_word_v0.1": {
        "url": "https://huggingface.co/your-org/your-models/resolve/main/my_word.onnx",
        "filename": "my_word_v0.1.onnx",
        "builtin": False,
    },
}
```

Then in your OpenOcto config:

```yaml
wakeword:
  enabled: true
  model: my_word_v0.1
  threshold: 0.5
```

## License

Apache 2.0 — same as [openWakeWord](https://github.com/dscripka/openWakeWord).
