"""
Stub for piper-sample-generator compatibility.
Clips are pre-generated via create_octo_samples.py — this function is never called.
"""


def generate_samples(*args, **kwargs):
    raise RuntimeError(
        "generate_samples() called unexpectedly. "
        "Run create_octo_samples.py to pre-generate clips, then use --augment_clips / --train_model."
    )
