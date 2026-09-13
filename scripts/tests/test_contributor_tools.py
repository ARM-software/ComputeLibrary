# SPDX-FileCopyrightText: 2026 Yusuf Efe
#
# SPDX-License-Identifier: MIT

"""Run with: python -m unittest discover -s scripts/tests -v."""

from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[1]
ROOT = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS))

import generate_android_bp
import format_doxygen
import include_functions_kernels


class ContributorToolsTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(prefix="acl tools ")
        self.addCleanup(self.temp_dir.cleanup)
        self.directory = Path(self.temp_dir.name)

    def test_android_sources_use_relative_posix_paths_and_exclude_graph(self):
        paths = [
            "src/core/Utils.cpp",
            "src/graph/Graph.cpp",
            "src/core/NEON/kernels/sve/kernel.cpp",
            "examples/example.cpp",
            "src/core/CL/cl_kernels/kernel.cl",
            "src/core/CL/cl_kernels/helpers.h",
        ]
        for path in paths:
            file = self.directory / path
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text("")
        sources, kernels = generate_android_bp.list_all_files(str(self.directory))
        self.assertEqual(sources, ["src/core/Utils.cpp"])
        self.assertEqual(set(kernels), {"src/core/CL/cl_kernels/kernel.cl", "src/core/CL/cl_kernels/helpers.h"})

    def test_umbrella_header_uses_filename_without_duplicating_directory(self):
        folder = self.directory / "kernels"
        folder.mkdir()
        (folder / "Alpha.h").write_text("")
        (folder / "Beta.h").write_text("")
        includes = include_functions_kernels.create_include_list(folder.as_posix())
        self.assertEqual(
            includes, [f'#include "{folder.as_posix()}/Alpha.h"\n', f'#include "{folder.as_posix()}/Beta.h"\n']
        )

    def test_umbrella_header_writes_lf_line_endings(self):
        header = self.directory / "header.h"
        include_functions_kernels.write_file(header, ["first\n", "second\n"])
        self.assertEqual(header.read_bytes(), b"first\nsecond\n")

    def test_doxygen_formatter_preserves_lf_line_endings(self):
        source = self.directory / "source.cpp"
        content = b"/* A comment */\nint value;\n"
        source.write_bytes(content)
        format_doxygen.main(str(source))
        self.assertEqual(source.read_bytes(), content)

    def test_header_guard_checker_preserves_utf8_and_lf(self):
        header = self.directory / "header.h"
        content = "// Unicode: π\n#ifndef ACL_HEADER_H\n#define ACL_HEADER_H\n#endif // ACL_HEADER_H\n".encode("utf-8")
        header.write_bytes(content)
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPTS / "check_header_guards.py"),
                "header.h",
                "--extensions=h",
                "--comment_style=double_slash",
                "--prefix=ACL",
                "--add_extension",
            ],
            cwd=self.directory,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(header.read_bytes(), content)

    def test_graph_build_sources_match_repository_paths(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; sys.path.insert(0, 'scripts'); "
                "import generate_build_files; "
                "print('\\n'.join(generate_build_files.gather_sources()[0]))",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        sources = result.stdout.splitlines()
        self.assertTrue(sources)
        self.assertTrue(all("\\" not in path and (ROOT / "src" / path).is_file() for path in sources))

    def test_android_generator_emits_utf8_and_lf(self):
        source = self.directory / "src/core/Utils.cpp"
        source.parent.mkdir(parents=True)
        source.write_text("")
        output = self.directory / "Generated_Android.bp"
        subprocess.run(
            [
                sys.executable,
                str(SCRIPTS / "generate_android_bp.py"),
                "--folder",
                str(self.directory),
                "--output_file",
                str(output),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        data = output.read_bytes()
        self.assertNotIn(b"\r\n", data)
        self.assertIn("Copyright ©", data.decode("utf-8"))

    def test_android_check_accepts_repository_path_with_spaces(self):
        source = self.directory / "src/core/Utils.cpp"
        source.parent.mkdir(parents=True)
        source.write_text("")
        sources, kernels = generate_android_bp.list_all_files(str(self.directory))
        (self.directory / "Android.bp").write_bytes(
            generate_android_bp.generate_bp_file(sources, kernels).encode("utf-8")
        )
        result = subprocess.run(
            [sys.executable, str(SCRIPTS / "format_code.py"), "--check_android_bp", "--folder", str(self.directory)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    @unittest.skipUnless(shutil.which("bash"), "Bash is required for the style check")
    def test_style_check_accepts_valid_file_and_rejects_invalid_type(self):
        source = self.directory / "style.cpp"
        (self.directory / "tests/validation/CL").mkdir(parents=True)
        for declaration, expected_success in [("unsigned int value;", True), ("uint value;", False)]:
            with self.subTest(declaration=declaration):
                source.write_text("// SPDX-" + "License-Identifier: MIT\n" + declaration + "\n")
                # The existing shell script expects its positional file list without spaces.
                result = subprocess.run(
                    [shutil.which("bash"), str(SCRIPTS / "check_bad_style.sh"), source.name],
                    cwd=self.directory,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                self.assertEqual(result.returncode == 0, expected_success, result.stdout + result.stderr)
                self.assertEqual(result.stderr, "")
                if not expected_success:
                    self.assertIn("Use 'unsigned int' instead", result.stdout)

    @unittest.skipUnless(shutil.which("bash"), "Bash is required for the style check")
    def test_style_check_rejects_missing_license_and_accepts_readme_exception(self):
        (self.directory / "tests/validation/CL").mkdir(parents=True)
        for name, expected_success in [("unlicensed.cpp", False), ("README.md", True)]:
            with self.subTest(name=name):
                (self.directory / name).write_text("A file without license metadata.\n")
                result = subprocess.run(
                    [shutil.which("bash"), str(SCRIPTS / "check_bad_style.sh"), name],
                    cwd=self.directory,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                self.assertEqual(result.returncode == 0, expected_success, result.stdout + result.stderr)
                self.assertEqual(result.stderr, "")
                if not expected_success:
                    self.assertIn("MIT Copyright header missing", result.stdout)


if __name__ == "__main__":
    unittest.main()
