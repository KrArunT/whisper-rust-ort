#!/usr/bin/env python3
import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List

import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic

from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
from transformers import AutoProcessor


MODEL_FILES = [
    "encoder_model.onnx",
    "decoder_model.onnx",
    "decoder_with_past_model.onnx",
]


def ensure_exported(model_id: str, onnx_dir: Path) -> None:
    if all((onnx_dir / name).is_file() for name in MODEL_FILES):
        return
    onnx_dir.mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained(model_id)
    model = ORTModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        export=True,
        use_cache=True,
        provider="CPUExecutionProvider",
        session_options=ort.SessionOptions(),
    )
    model.save_pretrained(onnx_dir)
    processor.tokenizer.save_pretrained(onnx_dir)


def copy_configs(src: Path, dst: Path) -> None:
    for name in ["config.json", "generation_config.json", "tokenizer.json"]:
        path = src / name
        if path.is_file():
            shutil.copy2(path, dst / name)


def ort_opt_level(level: str) -> ort.GraphOptimizationLevel:
    mapping = {
        "o1": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        "o2": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        "o3": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
        "o4": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    }
    return mapping[level]


def write_metadata(dst: Path, meta: Dict[str, object]) -> None:
    with (dst / "optimization_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)


def optimize_models(src: Path, dst: Path, level: str) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for name in MODEL_FILES:
        in_path = src / name
        out_path = dst / name
        if not in_path.is_file():
            raise FileNotFoundError(f"Missing ONNX model: {in_path}")
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort_opt_level(level)
        sess_opts.optimized_model_filepath = str(out_path)
        ort.InferenceSession(str(in_path), sess_options=sess_opts, providers=["CPUExecutionProvider"])


def quantize_models(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for name in MODEL_FILES:
        in_path = src / name
        out_path = dst / name
        if not in_path.is_file():
            raise FileNotFoundError(f"Missing ONNX model: {in_path}")
        quantize_dynamic(
            model_input=str(in_path),
            model_output=str(out_path),
            weight_type=QuantType.QInt8,
            op_types_to_quantize=["MatMul", "Gemm"],
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="openai/whisper-base")
    ap.add_argument("--onnx_dir", default="whisper-base-with-past")
    ap.add_argument("--out_dir", default="models/whisper-base-optimized")
    ap.add_argument("--opt_levels", default="o1,o2,o3,o4")
    ap.add_argument("--isas", default="avx2,avx512,vnni")
    ap.add_argument("--export_onnx", action="store_true")
    ap.add_argument("--quantize", action="store_true")
    args = ap.parse_args()

    onnx_dir = Path(args.onnx_dir)
    out_root = Path(args.out_dir)
    opt_levels = [s.strip().lower() for s in args.opt_levels.split(",") if s.strip()]
    isas = [s.strip().lower() for s in args.isas.split(",") if s.strip()]

    if args.export_onnx:
        ensure_exported(args.model_id, onnx_dir)

    for level in opt_levels:
        if level not in {"o1", "o2", "o3", "o4"}:
            raise ValueError(f"Unsupported opt level: {level}")

        opt_dir = out_root / f"{level}"
        optimize_models(onnx_dir, opt_dir, level)
        copy_configs(onnx_dir, opt_dir)
        write_metadata(
            opt_dir,
            {
                "opt_level": level,
                "graph_optimization_level": ort_opt_level(level).name,
                "note": "o4 maps to ORT_ENABLE_ALL",
                "isa_target": "baseline",
            },
        )

        if args.quantize:
            for isa in isas:
                int8_dir = out_root / f"{level}_int8_{isa}"
                quantize_models(opt_dir, int8_dir)
                copy_configs(onnx_dir, int8_dir)
                write_metadata(
                    int8_dir,
                    {
                        "opt_level": level,
                        "graph_optimization_level": ort_opt_level(level).name,
                        "note": "o4 maps to ORT_ENABLE_ALL; ISA is a label only (depends on ORT build/CPU)",
                        "isa_target": isa,
                        "quantization": "dynamic_int8_matmul_gemm",
                    },
                )

    print(f"Optimized models written to: {out_root}")


if __name__ == "__main__":
    main()
