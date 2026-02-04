#!/usr/bin/env python3
import argparse
from pathlib import Path

import onnxruntime as ort
from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
from transformers import AutoProcessor


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="openai/whisper-base")
    ap.add_argument("--onnx_dir", default="whisper-base-with-past")
    args = ap.parse_args()

    onnx_dir = Path(args.onnx_dir)
    onnx_dir.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(args.model_id)
    model = ORTModelForSpeechSeq2Seq.from_pretrained(
        args.model_id,
        export=True,
        use_cache=True,
        provider="CPUExecutionProvider",
        session_options=ort.SessionOptions(),
    )
    model.save_pretrained(onnx_dir)
    processor.tokenizer.save_pretrained(onnx_dir)

    print(f"Exported ONNX to {onnx_dir}")


if __name__ == "__main__":
    main()
