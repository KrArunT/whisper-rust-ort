#!/usr/bin/env python3
import argparse
import os
import shutil
from pathlib import Path

from onnxruntime.quantization import quantize_dynamic, QuantType


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.is_file():
        shutil.copy2(src, dst)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_dir", default="whisper-base-with-past")
    ap.add_argument("--dst_dir", default="whisper-base-with-past-int8")
    args = ap.parse_args()

    src = Path(args.src_dir)
    dst = Path(args.dst_dir)
    if not src.is_dir():
        raise SystemExit(f"Missing source dir: {src}")

    dst.mkdir(parents=True, exist_ok=True)

    for name in ["config.json", "generation_config.json", "tokenizer.json"]:
        copy_if_exists(src / name, dst / name)

    for name in ["encoder_model.onnx", "decoder_model.onnx", "decoder_with_past_model.onnx"]:
        in_path = src / name
        out_path = dst / name
        if not in_path.is_file():
            raise SystemExit(f"Missing ONNX file: {in_path}")
        print(f"Quantizing {in_path} -> {out_path}")
        quantize_dynamic(
            model_input=str(in_path),
            model_output=str(out_path),
            weight_type=QuantType.QInt8,
            op_types_to_quantize=["MatMul", "Gemm"],
        )

    print("DONE")
    print("Output dir:", dst)


if __name__ == "__main__":
    main()
