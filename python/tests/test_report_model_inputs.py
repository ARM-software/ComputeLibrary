# SPDX-FileCopyrightText: 2026 Yusuf Efe
#
# SPDX-License-Identifier: MIT

"""Run with: python -m unittest discover -s python/tests -v."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import flatbuffers
import tflite

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/report-model-ops/report_model_ops.py"


class ModelInputTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(prefix="acl model inputs ")
        self.addCleanup(self.temp_dir.cleanup)
        self.directory = Path(self.temp_dir.name)
        self.config = self.directory / "build_config.json"

    def run_report(self, models):
        return subprocess.run(
            [sys.executable, str(SCRIPT), "-m", *map(str, models), "-c", str(self.config)],
            capture_output=True,
            text=True,
            timeout=30,
        )

    def make_supported_model(self):
        """Serialize the operator metadata inspected by the reporter, without running inference."""
        builder = flatbuffers.Builder(128)
        tflite.OperatorCodeStart(builder)
        tflite.OperatorCodeAddBuiltinCode(builder, tflite.BuiltinOperator.ADD)
        code = tflite.OperatorCodeEnd(builder)
        tflite.ModelStartOperatorCodesVector(builder, 1)
        builder.PrependUOffsetTRelative(code)
        codes = builder.EndVector()
        tflite.ModelStart(builder)
        tflite.ModelAddVersion(builder, 3)
        tflite.ModelAddOperatorCodes(builder, codes)
        model = tflite.ModelEnd(builder)
        builder.Finish(model, file_identifier=b"TFL3")
        path = self.directory / "supported.tflite"
        path.write_bytes(bytes(builder.Output()))
        return path

    def assert_invalid_input(self, result, path):
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(str(path), result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        self.assertNotIn("WARNING:root:None", result.stderr)
        self.assertNotIn("=== Supported Operators", result.stderr)
        self.assertIn("error:", result.stderr)

    def test_missing_model_exits_cleanly_without_creating_config(self):
        missing = self.directory / "missing.tflite"
        self.assert_invalid_input(self.run_report([missing]), missing)
        self.assertFalse(self.config.exists())

    def test_unrecognized_model_files_exit_cleanly(self):
        for contents in [b"", b"text file", b"\x00\x00\x00\x00\xff\xfe\xfd\xfc"]:
            with self.subTest(contents=contents):
                path = self.directory / "unsupported.tflite"
                path.write_bytes(contents)
                self.assert_invalid_input(self.run_report([path]), path)
                self.assertFalse(self.config.exists())

    def test_directory_is_rejected_without_traceback(self):
        self.assert_invalid_input(self.run_report([self.directory]), self.directory)
        self.assertFalse(self.config.exists())

    def test_invalid_model_does_not_produce_partial_config(self):
        supported = self.make_supported_model()
        missing = self.directory / "missing.tflite"
        for models in [[supported, missing], [missing, supported]]:
            with self.subTest(models=models):
                self.assert_invalid_input(self.run_report(models), missing)
                self.assertFalse(self.config.exists())

    def test_invalid_model_preserves_existing_config(self):
        original = b'{"operators": ["Conv2d"]}\n'
        self.config.write_bytes(original)
        missing = self.directory / "missing.tflite"
        self.assert_invalid_input(self.run_report([self.make_supported_model(), missing]), missing)
        self.assertEqual(self.config.read_bytes(), original)

    def test_supported_model_still_generates_config(self):
        result = self.run_report([self.make_supported_model()])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("WARNING", result.stderr)
        self.assertEqual(
            json.loads(self.config.read_text()),
            {"operators": ["Add"], "data_types": [], "data_layouts": ["nhwc"]},
        )


if __name__ == "__main__":
    unittest.main()
