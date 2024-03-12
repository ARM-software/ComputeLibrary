#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
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

import argparse
from typing import List, Tuple
import logging
import re

logger = logging.getLogger("check_header_guards")

def find_code_boundaries(lines: List[str]) -> (int, int):
    inside_comment : bool = False

    start = len(lines)
    end = -1
    line_num = 0
    for line in lines:
        stripped_line : str = line.strip()
        if stripped_line.startswith("/*"):    # block comment start
            inside_comment = True

        if not inside_comment and not stripped_line.startswith("//") and stripped_line != "":
            start = min(line_num, start)
            end = line_num

        if inside_comment and stripped_line.endswith("*/"):
            inside_comment = False

        line_num += 1

    return start, end


def is_define(line: str) -> bool:
    return line.strip().startswith("#define")

def is_endif(line: str) -> bool:
    return line.strip().startswith("#endif")

def is_ifndef(line: str) -> bool:
    return line.strip().startswith("#ifndef")

# Strips the given line from // and /* */ blocks
def strip_comments(line: str) -> str:
    line = re.sub(r"/\*.*\*/", "", line)
    line = re.sub(r"//.*", "", line)
    return line.strip()

# If the line
#   1) startswith #ifndef
#   2) is all uppercase
#   3) does not start with double underscore, i.e. __
# Then
#   It "looks" like a header guard
def looks_like_header_guard(line: str) -> bool:
    sline = line.strip()
    guard_candidate = strip_comments(sline[len("#ifndef"):])

    return is_ifndef(sline) and not guard_candidate.startswith("__") and guard_candidate.isupper()


def fix_header_guard(lines: List[str], expected_header_guard: str, comment_style: str) -> Tuple[List[str], bool]:
    start_line, next_line, last_line = "", "", ""
    start_index, last_index = find_code_boundaries(lines)
    guards_updated: bool = True

    if start_index < len(lines):
        # if not, the file is full of comments
        start_line = lines[start_index]

    if start_index + 1 < len(lines):
        # if not, the file has only one line of code
        next_line = lines[start_index + 1]

    if last_index < len(lines) and last_index > start_index + 1:
        # if not, either the file is full of comments OR it has less than three code lines
        last_line = lines[last_index]

    expected_start_line = f"#ifndef {expected_header_guard}\n"
    expected_next_line = f"#define {expected_header_guard}\n"

    if comment_style == 'double_slash':
        expected_last_line = f"#endif // {expected_header_guard}\n"
    elif comment_style == 'slash_asterix':
        expected_last_line = f"#endif /* {expected_header_guard} */\n"

    empty_line = "\n"

    if looks_like_header_guard(start_line) and is_define(next_line) and is_endif(last_line):
        # modify the current header guard if necessary
        lines = lines[:start_index] + [expected_start_line, expected_next_line] + \
            lines[start_index+2:last_index] + [expected_last_line] + lines[last_index+1:]

        guards_updated = (start_line != expected_start_line) or (next_line != expected_next_line) \
            or (last_line != expected_last_line)
    else:
        # header guard could not be detected, add header guards
        lines = lines[:start_index] + [empty_line, expected_start_line, expected_next_line] + \
            [empty_line] + lines[start_index:] + [empty_line, expected_last_line]


    return lines, guards_updated


def find_expected_header_guard(filepath: str, prefix: str, add_extension: str, drop_outermost_subdir: str) -> str:
    if drop_outermost_subdir:
        arr : List[str] = filepath.split("/")
        arr = arr[min(1, len(arr)-1):]
        filepath = "/".join(arr)

    if not add_extension:
        filepath = ".".join(filepath.split(".")[:-1])

    guard = filepath.replace("/", "_").replace(".", "_").upper()    # snake case full path
    return prefix + "_" + guard


def skip_file(filepath: str, extensions: List[str], exclude: List[str], include: List[str]) -> bool:
    extension = filepath.split(".")[-1]

    if extension.lower() not in extensions:
        return True

    if exclude and any([filepath.startswith(exc) for exc in exclude]):
        print(exclude)
        return True

    if include:
        return not any([filepath.startswith(inc) for inc in include])

    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description="Header Guard Checker. It adds full path snake case header guards with or without extension.",
    )

    parser.add_argument("files", type=str, nargs="+", help="Files to check the header guards")
    parser.add_argument("--extensions", type=str, help="Comma separated list of extensions to run the checks. \
        If the input file does not have any of the extensions, it'll be skipped", required=True)
    parser.add_argument("--comment_style", choices=['double_slash', 'slash_asterix'], required=True)
    parser.add_argument("--exclude", type=str, help="Comma separated list of paths to exclude from header guard checks", default="")
    parser.add_argument("--include", type=str, help="Comma separated list of paths to include. Defaults to empty string, \
        which means all the paths are included", default="")
    parser.add_argument("--prefix", help="Prefix to apply to header guards", required=True)
    parser.add_argument("--add_extension", action="store_true", help="If true, it adds the file extension to the end of the guard")
    parser.add_argument("--drop_outermost_subdir", action="store_true", help="If true, it'll not use the outermost folder in the path. \
        This is intended for using in subdirs with different rules")

    args = parser.parse_args()

    files = args.files
    extensions = args.extensions.split(",")
    exclude = args.exclude.split(",") if args.exclude != '' else []
    include = args.include.split(",") if args.include != '' else []
    prefix = args.prefix
    add_extension = args.add_extension
    drop_outermost_subdir = args.drop_outermost_subdir
    comment_style = args.comment_style

    logging_level = logging.INFO
    logging.basicConfig(level=logging_level)

    retval = 0
    for file in files:
        if skip_file(file, extensions, exclude, include):
            logger.info(f"File {file} is SKIPPED")
            continue

        expected_header_guard : str = find_expected_header_guard(file, prefix, add_extension, drop_outermost_subdir)

        with open(file, "r") as fd:
            lines: List = fd.readlines()

        new_lines, guards_updated = fix_header_guard(lines, expected_header_guard, comment_style)

        with open(file, "w") as fd:
            fd.writelines([f"{line}" for line in new_lines])

        if guards_updated:
            logger.info("File has been modified")
            retval = 1

    exit(retval)
