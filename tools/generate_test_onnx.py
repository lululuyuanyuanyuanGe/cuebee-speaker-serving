"""Generate a tiny shape-compatible ONNX model for native runtime smoke tests."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def build_model(destination: Path) -> None:
    generator = np.random.default_rng(20260630)
    weights = generator.normal(0.0, 0.1, size=(80, 192)).astype(np.float32)
    graph = helper.make_graph(
        [
            helper.make_node("ReduceMean", ["features"], ["pooled"], axes=[1], keepdims=0),
            helper.make_node("MatMul", ["pooled", "weights"], ["embeddings"]),
        ],
        "cuebee_native_worker_smoke",
        [helper.make_tensor_value_info("features", TensorProto.FLOAT, ["batch", "time", 80])],
        [helper.make_tensor_value_info("embeddings", TensorProto.FLOAT, ["batch", 192])],
        [numpy_helper.from_array(weights, name="weights")],
    )
    model = helper.make_model(
        graph,
        producer_name="cuebee-speaker-serving-tests",
        opset_imports=[helper.make_opsetid("", 13)],
    )
    model.ir_version = 10
    onnx.checker.check_model(model)
    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    build_model(args.destination)


if __name__ == "__main__":
    main()
