#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright (c) 2023 Arm Limited.
#
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Generates build files for either bazel or cmake experimental builds using filelist.json
Usage
    python scripts/generate_build_files.py --bazel
    python scripts/generate_build_files.py --cmake

Writes generated file to the bazel BUILD file located under src/ if using --bazel flag.
Writes generated file to the CMake CMakeLists.txt file located under src/ if using --cmake flag.
"""

import argparse
import json
import glob


def get_operator_backend_files(filelist, operators, backend='', techs=[], attrs=[]):
    files = {"common": []}

    # Early return if filelist is empty
    if backend not in filelist:
        return files

    # Iterate over operators and create the file lists to compiler
    for operator in operators:
        if operator in filelist[backend]['operators']:
            files['common'] += filelist[backend]['operators'][operator]["files"]["common"]
            for tech in techs:
                if tech in filelist[backend]['operators'][operator]["files"]:
                    # Add tech as a key to dictionary if not there
                    if tech not in files:
                        files[tech] = []

                    # Add tech files to the tech file list
                    tech_files = filelist[backend]['operators'][operator]["files"][tech]
                    files[tech] += tech_files.get('common', [])
                    for attr in attrs:
                        files[tech] += tech_files.get(attr, [])

    # Remove duplicates if they exist
    return {k: list(set(v)) for k, v in files.items()}


def collect_operators(filelist, operators, backend=''):
    ops = set()
    for operator in operators:
        if operator in filelist[backend]['operators']:
            ops.add(operator)
            if 'deps' in filelist[backend]['operators'][operator]:
                ops.update(filelist[backend]['operators'][operator]['deps'])
        else:
            print("Operator {0} is unsupported on {1} backend!".format(
                operator, backend))

    return ops


def resolve_operator_dependencies(filelist, operators, backend=''):
    resolved_operators = collect_operators(filelist, operators, backend)

    are_ops_resolved = False
    while not are_ops_resolved:
        resolution_pass = collect_operators(
            filelist, resolved_operators, backend)
        if len(resolution_pass) != len(resolved_operators):
            resolved_operators.update(resolution_pass)
        else:
            are_ops_resolved = True

    return resolved_operators

def get_template_header():
    return """# Copyright (c) 2023 Arm Limited.
#
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE."""

def build_from_template_bazel(srcs_graph, srcs_sve, srcs_sve2, srcs_core):

    line_separator = '",\n\t"'

    template = f"""{get_template_header()}

filegroup(
        name = "arm_compute_graph_srcs",
        srcs = ["{line_separator.join(srcs_graph)}"]  +
    glob(["**/*.h",
    "**/*.hpp",
    "**/*.inl"]),
		visibility = ["//visibility:public"]
)

filegroup(
        name = "arm_compute_sve2_srcs",
        srcs = ["{line_separator.join(srcs_sve2)}"]  +
    glob(["**/*.h",
    "**/*.hpp",
    "**/*.inl"]),
		visibility = ["//visibility:public"]
)

filegroup(
        name = "arm_compute_sve_srcs",
        srcs = ["{line_separator.join(srcs_sve)}"]  +
    glob(["**/*.h",
    "**/*.hpp",
    "**/*.inl"]),
		visibility = ["//visibility:public"]
)

filegroup(
        name = "arm_compute_srcs",
        srcs = ["{line_separator.join(srcs_core)}"]  +
    glob(["**/*.h",
    "**/*.hpp",
    "**/*.inl"]),
		visibility = ["//visibility:public"]
)
"""

    return template


def build_from_template_cmake(srcs_graph, srcs_sve, srcs_sve2, srcs_core):

    line_separator = '\n\t'

    template = f"""{get_template_header()}

target_sources(
    arm_compute_graph
    PRIVATE
    {line_separator.join(srcs_graph)}
)

target_sources(
    arm_compute_sve
    PRIVATE
    {line_separator.join(srcs_sve)}
)

target_sources(
    arm_compute_sve2
    PRIVATE
    {line_separator.join(srcs_sve2)}
)

target_sources(
    arm_compute
    PRIVATE
    {line_separator.join(srcs_core)}
)"""
    return template


def gather_sources():

    # Source file list
    with open("filelist.json") as fp:
        filelist = json.load(fp)

    # Common backend files
    lib_files = filelist['common']

    # TODO Add Fixed format GEMM kernels ?

    # Logging files
    lib_files += filelist['logging']

    # C API files
    lib_files += filelist['c_api']['common']
    lib_files += filelist['c_api']['operators']

    # Scheduler infrastructure
    lib_files += filelist['scheduler']['single']
    # Add both cppthreads and omp sources for now
    lib_files += filelist['scheduler']['threads']
    lib_files += filelist['scheduler']['omp']

    # Graph files
    graph_files = glob.glob('src/graph/*.cpp')
    graph_files += glob.glob('src/graph/*/*.cpp')

    lib_files_sve = []
    lib_files_sve2 = []

    # -------------------------------------
    # NEON files
    lib_files += filelist['cpu']['common']
    simd = ['neon', 'sve', 'sve2']

    # Get attributes
    data_types = ["qasymm8", "qasymm8_signed", "qsymm16",
                  "fp16", "fp32", "integer"]
    data_layouts = ["nhwc", "nchw"]
    fixed_format_kernels = ["fixed_format_kernels"]
    attrs = data_types + data_layouts + \
        fixed_format_kernels + ["estate64"]

    # Setup data-type and data-layout files to include
    cpu_operators = filelist['cpu']['operators'].keys()
    cpu_ops_to_build = resolve_operator_dependencies(
        filelist, cpu_operators, 'cpu')
    cpu_files = get_operator_backend_files(
        filelist, cpu_ops_to_build, 'cpu', simd, attrs)

    # Shared among ALL CPU files
    lib_files += cpu_files.get('common', [])

    # Arm® Neon™ specific files
    lib_files += cpu_files.get('neon', [])

    # SVE files only
    lib_files_sve = cpu_files.get('sve', [])

    # SVE2 files only
    lib_files_sve2 = cpu_files.get('sve2', [])

    graph_files += glob.glob('src/graph/backends/NEON/*.cpp')

    # -------------------------------------

    graph_files = sorted([path.replace("src/", "") for path in graph_files])
    lib_files_sve = sorted([path.replace("src/", "") for path in lib_files_sve])
    lib_files_sve2 = sorted([path.replace("src/", "") for path in lib_files_sve2])
    lib_files = sorted([path.replace("src/", "") for path in lib_files])

    return graph_files, lib_files_sve, lib_files_sve2, lib_files


if "__main__" in __name__:

    parser = argparse.ArgumentParser()
    parser.add_argument("--bazel", action="store_true")
    parser.add_argument("--cmake", action="store_true")
    args = parser.parse_args()

    graph_files, lib_files_sve, lib_files_sve2, lib_files = gather_sources()

    if args.bazel:
        bazel_build_string = build_from_template_bazel(
            graph_files, lib_files_sve, lib_files_sve2, lib_files)
        with open("src/BUILD.bazel", "w") as fp:
            fp.write(bazel_build_string)

    if args.cmake:
        cmake_build_string = build_from_template_cmake(
            graph_files, lib_files_sve, lib_files_sve2, lib_files)
        with open("src/CMakeLists.txt", "w") as fp:
            fp.write(cmake_build_string)

    if not args.cmake and not args.bazel:
        print("Supply either --bazel or --cmake flag to generate build files for corresponding build")
