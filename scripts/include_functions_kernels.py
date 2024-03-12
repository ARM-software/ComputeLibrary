#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2017-2018, 2020-2021, 2023 Arm Limited.
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
import glob
import collections
import os

armcv_path = "arm_compute"
src_path ="src"

Target = collections.namedtuple('Target', 'name prefix basepath')

core_targets = [
    Target("NEON", "NE", src_path),             # Arm® Neon™ kernels are under src
    Target("CL", "CL", src_path),               # CL kernels are under src
    Target("CPP", "CPP", armcv_path)            # CPP kernels are under arm_compute
    ]

# All functions are under arm_compute
runtime_targets = [
    Target("NEON", "NE", armcv_path),
    Target("CL", "CL", armcv_path),
    Target("CPP", "CPP", armcv_path)
    ]

core_path = "/core/"
runtime_path = "/runtime/"
include_str = "#include \""

def read_file(file):
    with open(file, "r") as f:
        lines = f.readlines()
    return lines


def write_file(file, lines):
    with open(file, "w") as f:
        for line in lines:
            f.write(line)


def remove_existing_includes(lines):
    first_pos = next(i for i, line in enumerate(lines) if include_str in line)
    return [x for x in lines if not x.startswith(include_str)], first_pos


def add_updated_includes(lines, pos, includes):
    lines[pos:pos] = includes
    return lines


def create_include_list(folder):
    files_path = folder + "/*.h"
    files = glob.glob(files_path)
    updated_files = [include_str + folder + "/" + x.rsplit('/',1)[1] + "\"\n" for x in files]
    updated_files.sort(key=lambda x: x.lower())
    return updated_files


def include_components(target, path, header_prefix, folder, subfolders=None):
    for t in target:
        target_path = t.basepath + path +  t.name + "/"
        components_file = target_path + t.prefix + header_prefix
        if os.path.exists(components_file):
            include_list = create_include_list(target_path + folder)
            for s in subfolders or []:
                include_list += create_include_list( target_path + folder + "/" + s)
            include_list.sort(key=lambda x: x.lower())
            lines = read_file(components_file)
            lines, first_pos = remove_existing_includes(lines)
            lines = add_updated_includes(lines, first_pos, include_list)
            write_file(components_file, lines)


if __name__ == "__main__":
    # Include kernels
    include_components(core_targets, core_path, "Kernels.h", "kernels", ["arm32", "arm64"])

    # Include functions
    include_components(runtime_targets, runtime_path, "Functions.h", "functions")
