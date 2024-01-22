#!/usr/bin/env python3

# Copyright (c) 2023-2024 Arm Limited.
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

import argparse
import os
from jinja2 import Template
import datetime

# Paths to exclude
excluded_paths = ["build",
                  "compute_kernel_writer/",
                  "src/dynamic_fusion/runtime/gpu/cl/ckw_driver/",
                  "src/dynamic_fusion/sketch/gpu/ckw_driver/",
                  "docs/",
                  "documentation/",
                  "examples/",
                  "opencl-1.2-stubs/",
                  "release_repository/",
                  "opengles-3.1-stubs/",
                  "scripts/",
                  "tests/",
                  "/GLES_COMPUTE/",
                  "/graph/",
                  "/sve/",
                  "/SVE/",
                  "/sve2/",
                  "/SVE2/"
                  ]

excluded_files = ["TracePoint.cpp"]

# Android bp template to render
year = datetime.datetime.now().year

bp_tm = Template(
"""//
// Copyright © 2020-""" + str(year) + """ Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

// OpenCL sources are NOT required by ArmNN or its Android NNAPI driver and are used for CI purposes only.
opencl_srcs = [
        {% for cl_src in cl_srcs -%}
            "{{ cl_src }}",
        {% endfor %}
]

bootstrap_go_package {
    name: "arm_compute_library_nn_driver",
    pkgPath: "arm_compute_library_nn_driver",
    deps: [
        "blueprint",
        "blueprint-pathtools",
        "blueprint-proptools",
        "soong",
        "soong-android",
        "soong-cc",
    ],
    srcs: [
        "scripts/arm_compute_library_nn_driver.go",
    ],
    pluginFor: [ "soong_build" ],
}

arm_compute_library_defaults {
       name: "acl-default-cppflags",
       cppflags: [
            "-std=c++14",
            "-fexceptions",
            "-DBOOST_NO_AUTO_PTR",
            "-DEMBEDDED_KERNELS",
            "-DARM_COMPUTE_ASSERTS_ENABLED",
            "-DARM_COMPUTE_CPP_SCHEDULER",
            "-DENABLE_NEON",
            "-DARM_COMPUTE_ENABLE_NEON",
            "-Wno-unused-parameter",
            "-DNO_DOT_IN_TOOLCHAIN",
            "-Wno-implicit-fallthrough",
            "-fPIC"
    ],
    rtti: true,
}

cc_library_static {
    name: "arm_compute_library",
    defaults: ["acl-default-cppflags"],
    proprietary: true,
    local_include_dirs: ["build/android-arm64v8a/src/core",
                         "build/android-arm64v8a/src/core/CL",
                         "src/core/common",
                         "src/core/helpers",
                         "src/core/NEON/kernels/arm_gemm",
                         "src/core/NEON/kernels/assembly",
                         "src/core/NEON/kernels/convolution/common",
                         "src/core/NEON/kernels/convolution/winograd",
                         "src/cpu/kernels/assembly"],
    export_include_dirs: [".", "./include"],
    srcs: [
        {% for src in srcs -%}
            "{{ src }}",
        {% endfor %}
    ],
    arch: {
        arm: {
            srcs: [
                {% for arm_src in arm_srcs -%}
                    "{{ arm_src }}",
                {% endfor %}
            ],
        },
        arm64: {
            srcs: [
                {% for arm64_src in arm64_srcs -%}
                    "{{ arm64_src }}",
                {% endfor %}
            ],
        },
    },
    rtti: true,
}
""")


def generate_bp_file(cpp_files, opencl_files):
    arm_files = [f for f in cpp_files if "a32_" in f]
    arm64_files = [f for f in cpp_files if any(a64 in f for a64 in ["a64_", "sve_", 'sme_', 'sme2_'])]
    gen_files = [x for x in cpp_files if x not in arm_files + arm64_files]

    arm_files.sort()
    arm64_files.sort()
    gen_files.sort()
    opencl_files.sort()

    bp_file = bp_tm.render(srcs=gen_files,
                           arm_srcs=arm_files,
                           arm64_srcs=arm64_files,
                           cl_srcs=opencl_files)
    return bp_file


def list_all_files(repo_path):
    """ Gets the list of files to include to the Android.bp

    :param repo_path: Path of the repository
    :return: The filtered list of useful filess
    """
    if not repo_path.endswith('/'):
        repo_path = repo_path + "/"

    # Get cpp files
    cpp_files = []
    cl_files = []
    for path, subdirs, files in os.walk(repo_path):
        for file in files:
            if file.endswith(".cpp"):
                cpp_files.append(os.path.join(path, file))
            elif file.endswith(".cl"):
                cl_files.append(os.path.join(path, file))
            # Include CL headers
            if "src/core/CL/cl_kernels" in path and file.endswith(".h"):
                cl_files.append(os.path.join(path, file))
    # Filter out unused cpp files
    filtered_cpp_files = []
    for cpp_file in cpp_files:
        if any(ep in cpp_file for ep in excluded_paths) or any(ef in cpp_file for ef in excluded_files):
            continue
        filtered_cpp_files.append(cpp_file.replace(repo_path, ""))
    # Filter out unused cl files
    filtered_cl_files = []
    for cl_file in cl_files:
        if any(ep in cl_file for ep in excluded_paths):
            continue
        filtered_cl_files.append(cl_file.replace(repo_path, ""))

    return filtered_cpp_files, filtered_cl_files


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser('Generate Android.bp file for ComputeLibrary')
    parser.add_argument('--folder', default=".", metavar="folder", dest='folder', type=str, required=False, help='Compute Library source path')
    parser.add_argument('--output_file', metavar="output_file", default='Android.bp', type=str, required=False, help='Specify Android bp output file')
    args = parser.parse_args()

    cpp_files, opencl_files = list_all_files(args.folder)
    bp_file = generate_bp_file(cpp_files, opencl_files)

    with open(args.output_file, 'w') as f:
        f.write(bp_file)
