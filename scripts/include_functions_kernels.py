#!/usr/bin/env python
#FIXME: Remove this file before the release

import glob
import collections
import os

Target = collections.namedtuple('Target', 'name prefix')

targets = [Target("NEON", "NE"), Target("CL", "CL"), Target("CPP", "CPP"), Target("GLES_COMPUTE", "GC")]

armcv_path = "arm_compute"
core_path = armcv_path + "/core/"
runtime_path = armcv_path + "/runtime/"
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
    updated_files.sort()
    return updated_files


def include_components(path, header_prefix, folder, subfolders=None):
    for t in targets:
        target_path = path +  t.name + "/"
        components_file = target_path + t.prefix + header_prefix
        if os.path.exists(components_file):
            include_list = create_include_list(target_path + folder)
            for s in subfolders or []:
                include_list += create_include_list( target_path + folder + "/" + s)
            include_list.sort()
            lines = read_file(components_file)
            lines, first_pos = remove_existing_includes(lines)
            lines = add_updated_includes(lines, first_pos, include_list)
            write_file(components_file, lines)


if __name__ == "__main__":
    # Include kernels
    include_components(core_path, "Kernels.h", "kernels", ["arm32", "arm64"])

    # Include functions
    include_components(runtime_path, "Functions.h", "functions")
