# SPDX-FileCopyrightText: 2026 Yusuf Efe
#
# SPDX-License-Identifier: MIT

"""Regression tests for reporting unsupported TFLite operators.

Run from the repository root after installing python/requirements.txt:
    python -m unittest discover -s python/tests -v
"""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import flatbuffers
import tflite

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from utils.tflite_helpers import tflite_op2acl


class OperatorMappingTest(unittest.TestCase):
    def test_supported_operators_keep_their_mapping(self):
        for operator, expected in [("ADD", "Add"), ("CONV_2D", "Conv2d"), ("RELU", "Activation")]:
            with self.subTest(operator=operator):
                self.assertEqual(tflite_op2acl(operator), expected)

    def test_unsupported_operators_raise_value_error_with_operator_name(self):
        for operator in ["CUSTOM", "IF", "WHILE", "ONE_HOT", "UNKNOWN_OPERATOR"]:
            with self.subTest(operator=operator):
                with self.assertRaisesRegex(ValueError, f"Operator {operator} does not exist in ComputeLibrary"):
                    tflite_op2acl(operator)


class ModelReportTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.directory = Path(self.temp_dir.name)

    def make_model(self, filename, operators):
        """Serialize the operator and tensor metadata consumed by the reporter.

        These fixtures exercise model inspection, not inference execution.
        """
        builder = flatbuffers.Builder(256)
        codes = []
        for operator in operators:
            opcode = getattr(tflite.BuiltinOperator, operator)
            tflite.OperatorCodeStart(builder)
            tflite.OperatorCodeAddBuiltinCode(builder, opcode)
            tflite.OperatorCodeAddDeprecatedBuiltinCode(builder, opcode)
            codes.append(tflite.OperatorCodeEnd(builder))

        tflite.ModelStartOperatorCodesVector(builder, len(codes))
        for code in reversed(codes):
            builder.PrependUOffsetTRelative(code)
        operator_codes = builder.EndVector()

        tflite.TensorStart(builder)
        tflite.TensorAddType(builder, tflite.TensorType.FLOAT32)
        tensor = tflite.TensorEnd(builder)
        tflite.SubGraphStartTensorsVector(builder, 1)
        builder.PrependUOffsetTRelative(tensor)
        tensors = builder.EndVector()
        tflite.SubGraphStart(builder)
        tflite.SubGraphAddTensors(builder, tensors)
        subgraph = tflite.SubGraphEnd(builder)
        tflite.ModelStartSubgraphsVector(builder, 1)
        builder.PrependUOffsetTRelative(subgraph)
        subgraphs = builder.EndVector()

        tflite.ModelStart(builder)
        tflite.ModelAddVersion(builder, 3)
        tflite.ModelAddOperatorCodes(builder, operator_codes)
        tflite.ModelAddSubgraphs(builder, subgraphs)
        model = tflite.ModelEnd(builder)
        builder.Finish(model, file_identifier=b"TFL3")

        path = self.directory / filename
        path.write_bytes(bytes(builder.Output()))
        return path

    def run_report(self, models):
        config = self.directory / "build_config.json"
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPTS / "report-model-ops" / "report_model_ops.py"),
                "-m",
                *map(str, models),
                "-c",
                str(config),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        self.assertTrue(config.is_file())
        return json.loads(config.read_text()), result.stderr

    def test_supported_models_generate_build_configuration(self):
        model = self.make_model("supported.tflite", ["ADD", "CONV_2D"])
        config, stderr = self.run_report([model])
        self.assertEqual(set(config["operators"]), {"Add", "Conv2d"})
        self.assertEqual(config["data_types"], ["fp32"])
        self.assertEqual(config["data_layouts"], ["nhwc"])
        self.assertNotIn("WARNING", stderr)

    def test_mixed_models_report_unsupported_operators_and_keep_supported_ones(self):
        unsupported = ["CUSTOM", "IF", "WHILE", "ONE_HOT"]
        mixed = self.make_model("mixed.tflite", ["ADD", *unsupported])
        supported = self.make_model("supported.tflite", ["CONV_2D"])
        config, stderr = self.run_report([mixed, supported])
        self.assertEqual(set(config["operators"]), {"Add", "Conv2d"})
        self.assertEqual(config["data_types"], ["fp32"])
        self.assertEqual(config["data_layouts"], ["nhwc"])
        self.assertIn("=== Unsupported Operators", stderr)
        for operator in unsupported:
            self.assertIn(f"Operator {operator} does not have ComputeLibrary mapping", stderr)

    def test_unsupported_only_model_does_not_emit_placeholder_operator(self):
        model = self.make_model("unsupported.tflite", ["ONE_HOT"])
        config, stderr = self.run_report([model])
        self.assertEqual(config["operators"], [])
        self.assertIn("=== Unsupported Operators", stderr)
        self.assertIn("Operator ONE_HOT does not have ComputeLibrary mapping", stderr)


if __name__ == "__main__":
    unittest.main()
