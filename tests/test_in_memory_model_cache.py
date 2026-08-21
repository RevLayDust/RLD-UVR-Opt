"""Regression coverage for in-memory FP16 conversion and session eviction."""

from __future__ import annotations

import gc
import tempfile
import unittest
import weakref
from pathlib import Path
from unittest import mock

import numpy as np
import onnx
from onnx import TensorProto, helper

import separate
from gui_data.constants import MODEL_PRECISION_FP16, MODEL_PRECISION_FP32
from separate import ModelCacheManager, SmartPrecisionPolicy, create_onnx_fp16_session, create_onnx_session


class CachedModel:
    pass


class WorkerOwner:
    def execute(self):
        return "done"


class InMemoryONNXTests(unittest.TestCase):
    def test_fp16_conversion_uses_memory_and_does_not_touch_stale_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_path = root / "tiny.onnx"
            stale_fp16_path = root / "tiny.fp16.onnx"
            graph = helper.make_graph(
                [helper.make_node("Identity", ["input"], ["output"])],
                "in-memory-fp16",
                [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 2])],
                [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2])],
            )
            model = helper.make_model(
                graph,
                opset_imports=[helper.make_operatorsetid("", 17)],
                ir_version=10,
            )
            onnx.save(model, source_path)
            stale_fp16_path.write_bytes(b"legacy-file-must-remain-untouched")
            files_before = {path.name: path.read_bytes() for path in root.iterdir()}

            session = create_onnx_fp16_session(source_path, ["CPUExecutionProvider"])

            files_after = {path.name: path.read_bytes() for path in root.iterdir()}
            self.assertEqual(session.get_inputs()[0].type, "tensor(float16)")
            output = session.run(None, {"input": np.ones((1, 2), dtype=np.float16)})[0]
            self.assertEqual(output.dtype, np.float16)
            self.assertEqual(files_after, files_before)

    def test_fp16_failure_falls_back_to_original_fp32_path(self):
        policy = SmartPrecisionPolicy(MODEL_PRECISION_FP16)
        expected_session = object()
        with mock.patch.object(
            separate,
            "create_onnx_fp16_session",
            side_effect=RuntimeError("synthetic conversion failure"),
        ), mock.patch.object(
            separate.ort,
            "InferenceSession",
            return_value=expected_session,
        ) as inference_session:
            result = create_onnx_session("source.onnx", ["CPUExecutionProvider"], policy)

        self.assertIs(result, expected_session)
        inference_session.assert_called_once_with("source.onnx", providers=["CPUExecutionProvider"])

    def test_onnx_worker_does_not_retain_last_session_method(self):
        owner = WorkerOwner()
        owner_reference = weakref.ref(owner)

        result = separate.ONNX_WORKER.execute(owner.execute)
        separate.ONNX_WORKER.queue.join()
        del owner
        gc.collect()

        self.assertEqual(result, "done")
        self.assertIsNone(owner_reference())


class SingleActiveModelCacheTests(unittest.TestCase):
    def test_same_model_and_precision_reuses_session(self):
        manager = ModelCacheManager()
        session = CachedModel()
        manager.set_model("model-a.onnx", session, precision_mode=MODEL_PRECISION_FP16)

        cached = manager.get_model("model-a.onnx", precision_mode=MODEL_PRECISION_FP16)

        self.assertIs(cached, session)
        self.assertEqual(len(manager.cache), 1)

    def test_model_change_releases_previous_cached_session(self):
        manager = ModelCacheManager()
        previous = CachedModel()
        previous_reference = weakref.ref(previous)
        manager.set_model("model-a.onnx", previous, precision_mode=MODEL_PRECISION_FP16)
        del previous

        cached = manager.get_model("model-b.onnx", precision_mode=MODEL_PRECISION_FP16)

        self.assertIsNone(cached)
        self.assertIsNone(previous_reference())
        self.assertEqual(manager.cache, {})

    def test_precision_change_releases_previous_cached_session(self):
        manager = ModelCacheManager()
        previous = CachedModel()
        previous_reference = weakref.ref(previous)
        manager.set_model("model-a.onnx", previous, precision_mode=MODEL_PRECISION_FP32)
        del previous

        cached = manager.get_model("model-a.onnx", precision_mode=MODEL_PRECISION_FP16)

        self.assertIsNone(cached)
        self.assertIsNone(previous_reference())
        self.assertEqual(manager.cache, {})


if __name__ == "__main__":
    unittest.main()
