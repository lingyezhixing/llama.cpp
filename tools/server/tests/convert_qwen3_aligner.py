#!/usr/bin/env python3
"""Converts a Qwen3-ForcedAligner checkpoint and checks the ASR metadata keys.

This is a gated helper for a local checkpoint, not part of the default test run:

    python tools/server/tests/convert_qwen3_aligner.py \
        --hf-dir E:/models/LLM/Qwen3-ForcedAligner-0.6B \
        --outfile C:/tmp/aligner-q8.gguf

The aligner shares the qwen3vl architecture with the ASR text model and only adds
a 5000-class timestamp head, so the checks below are what the C++ loader relies on.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def convert(hf_dir: Path, outfile: Path, mmproj: bool) -> None:
    cmd = [
        sys.executable, str(REPO_ROOT / "convert_hf_to_gguf.py"),
        str(hf_dir),
        "--outtype", "q8_0",
        "--outfile", str(outfile),
    ]
    if mmproj:
        cmd.append("--mmproj")
    print("running:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def load(path: Path):
    sys.path.insert(0, str(REPO_ROOT / "gguf-py"))
    import gguf
    return gguf, gguf.GGUFReader(path)


def field(reader, key):
    entry = reader.get_field(key)
    assert entry is not None, f"missing metadata key {key!r}"
    return entry.contents()


def find_tensor(reader, name):
    for tensor in reader.tensors:
        if tensor.name == name:
            return tensor
    return None


def check_aligner(path: Path) -> None:
    gguf, reader = load(path)

    arch = field(reader, "general.architecture")
    assert arch == "qwen3vl", f"architecture is {arch!r}, expected qwen3vl"

    assert field(reader, "qwen3vl.classifier.out_count") == 5000
    assert field(reader, "qwen3vl.asr.timestamp_token_id") == 151705
    assert abs(float(field(reader, "qwen3vl.asr.timestamp_segment_time_ms")) - 80.0) < 1e-6

    cls_out = find_tensor(reader, "cls.output.weight")
    assert cls_out is not None, "cls.output.weight is missing"
    assert tuple(int(dim) for dim in cls_out.shape) == (1024, 5000), \
        f"cls.output.weight has shape {tuple(int(dim) for dim in cls_out.shape)}"
    assert find_tensor(reader, "output.weight") is None, "the aligner must not carry output.weight"

    print(f"aligner ok: {path}")


def check_mmproj(path: Path) -> None:
    gguf, reader = load(path)

    assert field(reader, "clip.audio.projector_type") == "qwen3a"
    assert field(reader, "clip.audio.num_mel_bins") == 128

    print(f"aligner mmproj ok: {path}")


def check_asr_text(path: Path) -> None:
    gguf, reader = load(path)

    assert field(reader, "general.architecture") == "qwen3vl"
    assert reader.get_field("qwen3vl.classifier.out_count") is None, \
        "the ASR text model must not carry classifier.out_count"
    assert reader.get_field("qwen3vl.asr.timestamp_token_id") is None, \
        "the ASR text model must not carry asr.timestamp_token_id"
    assert find_tensor(reader, "output.weight") is not None, \
        "the ASR text model keeps its vocabulary head"

    print(f"asr text model ok: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-dir", type=Path, required=True, help="Qwen3-ForcedAligner checkpoint")
    parser.add_argument("--outfile", type=Path, required=True, help="aligner GGUF to write")
    parser.add_argument("--mmproj-outfile", type=Path, help="aligner mmproj GGUF to write")
    parser.add_argument("--asr-dir", type=Path, help="optional Qwen3-ASR text checkpoint for the regression check")
    parser.add_argument("--asr-outfile", type=Path, help="where to write the ASR text GGUF")
    parser.add_argument("--skip-aligner", action="store_true", help="only run the mmproj and optional ASR checks")
    args = parser.parse_args()

    if not args.skip_aligner:
        convert(args.hf_dir, args.outfile, mmproj=False)
        check_aligner(args.outfile)

    if args.mmproj_outfile is not None:
        convert(args.hf_dir, args.mmproj_outfile, mmproj=True)
        check_mmproj(args.mmproj_outfile)

    if args.asr_dir is not None:
        asr_outfile = args.asr_outfile or args.outfile.with_name("asr-text-q8.gguf")
        convert(args.asr_dir, asr_outfile, mmproj=False)
        check_asr_text(asr_outfile)

    return 0


if __name__ == "__main__":
    sys.exit(main())
