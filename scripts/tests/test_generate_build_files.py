# SPDX-FileCopyrightText: 2026 Yusuf Efe
#
# SPDX-License-Identifier: MIT

"""Run with: python -m unittest discover -s scripts/tests -p test_generate_build_files.py -v."""

from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "generate_build_files.py"


class GenerateBuildFilesTest(unittest.TestCase):
    def generate(self, *flags):
        with tempfile.TemporaryDirectory(prefix="acl build ") as directory:
            root = Path(directory)
            shutil.copyfile(ROOT / "filelist.json", root / "filelist.json")
            graph = root / "src" / "graph"
            graph.mkdir(parents=True)
            (graph / "Graph.cpp").touch()
            result = subprocess.run(
                [sys.executable, str(SCRIPT), *flags], cwd=root, capture_output=True, text=True
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            return {
                name: (root / "src" / name).read_bytes()
                for name in ("CMakeLists.txt", "BUILD.bazel")
                if (root / "src" / name).exists()
            }

    def test_combined_generation_matches_separate_invocations(self):
        cmake = self.generate("--cmake")
        bazel = self.generate("--bazel")
        self.assertEqual(set(cmake), {"CMakeLists.txt"})
        self.assertEqual(set(bazel), {"BUILD.bazel"})
        for flags in (("--bazel", "--cmake"), ("--cmake", "--bazel")):
            with self.subTest(flags=flags):
                self.assertEqual(self.generate(*flags), {**cmake, **bazel})

    def test_bazel_utility_label_stays_out_of_cmake(self):
        outputs = self.generate("--bazel", "--cmake")
        self.assertIn(b"//utils:CommonGraphOptions.cpp", outputs["BUILD.bazel"])
        self.assertNotIn(b"//utils:CommonGraphOptions.cpp", outputs["CMakeLists.txt"])
        self.assertIn(b"Graph.cpp", outputs["BUILD.bazel"])
        self.assertIn(b"Graph.cpp", outputs["CMakeLists.txt"])


if __name__ == "__main__":
    unittest.main()
