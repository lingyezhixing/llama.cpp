#!/usr/bin/env python3

import os
import sys
from pathlib import Path

# load the local gguf package
if "NO_LOCAL_GGUF" not in os.environ and (Path(__file__).parent.parent.parent / 'gguf-py').exists():
    sys.path.insert(0, str(Path(__file__).parent.parent))

import gguf


def test_aligner_key_names():
    assert gguf.Keys.Classifier.OUT_COUNT.format(arch="qwen3vl") == "qwen3vl.classifier.out_count"
    assert gguf.Keys.ASR.TIMESTAMP_TOKEN_ID.format(arch="qwen3vl") == "qwen3vl.asr.timestamp_token_id"
    assert gguf.Keys.ASR.TIMESTAMP_SEGMENT_TIME_MS.format(arch="qwen3vl") == "qwen3vl.asr.timestamp_segment_time_ms"


def test_aligner_writer_methods(tmp_path):
    writer = gguf.GGUFWriter(tmp_path / "aligner.gguf", "qwen3vl")
    writer.add_classifier_out_count(5000)
    writer.add_asr_timestamp_token_id(151705)
    writer.add_asr_timestamp_segment_time_ms(80.0)

    kv = writer.kv_data[0]
    assert kv["qwen3vl.classifier.out_count"].value == 5000
    assert kv["qwen3vl.classifier.out_count"].type == gguf.GGUFValueType.UINT32
    assert kv["qwen3vl.asr.timestamp_token_id"].value == 151705
    assert kv["qwen3vl.asr.timestamp_token_id"].type == gguf.GGUFValueType.UINT32
    assert kv["qwen3vl.asr.timestamp_segment_time_ms"].value == 80.0
    assert kv["qwen3vl.asr.timestamp_segment_time_ms"].type == gguf.GGUFValueType.FLOAT32


def test_qwen3vl_has_cls_out():
    assert gguf.MODEL_TENSOR.CLS_OUT in gguf.MODEL_TENSORS[gguf.MODEL_ARCH.QWEN3VL]
    assert gguf.TENSOR_NAMES[gguf.MODEL_TENSOR.CLS_OUT] == "cls.output"
