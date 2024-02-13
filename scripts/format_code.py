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
import datetime
import difflib
import filecmp
import logging
import os
import re
import subprocess
import sys

from modules.Shell import Shell

logger = logging.getLogger("format_code")

# List of directories to exclude
exceptions = [
    "src/core/NEON/kernels/assembly/gemm",
    "src/core/NEON/kernels/assembly/arm",
    "/winograd/",
    "/convolution/",
    "/arm_gemm/",
    "/arm_conv/",
    "SConscript",
    "SConstruct"
]

def adjust_copyright_year(copyright_years, curr_year):
    ret_copyright_year = str()
    # Read last year in the Copyright
    last_year = int(copyright_years[-4:])
    if last_year == curr_year:
        ret_copyright_year = copyright_years
    elif last_year == (curr_year - 1):
        # Create range if latest year on the copyright is the previous
        if len(copyright_years) > 4 and copyright_years[-5] == "-":
            # Range already exists, update year to current
            ret_copyright_year = copyright_years[:-5] + "-" + str(curr_year)
        else:
            # Create a new range
            ret_copyright_year = copyright_years + "-" + str(curr_year)
    else:
        ret_copyright_year = copyright_years + ", " + str(curr_year)
    return ret_copyright_year

def check_copyright( filename ):
    f = open(filename, "r")
    content = f.readlines()
    f.close()
    f = open(filename, "w")
    year = datetime.datetime.now().year
    ref = open("scripts/copyright_mit.txt","r").readlines()

    # Need to handle python files separately
    if("SConstruct" in filename or "SConscript" in filename):
        start = 2
        if("SConscript" in filename):
            start = 3
        m = re.match(r"(# Copyright \(c\) )(.*\d{4})( [Arm|ARM].*)", content[start])
        line = m.group(1)

        if m.group(2): # Is there a year already?
            # Yes: adjust accordingly
            line += adjust_copyright_year(m.group(2), year)
        else:
            # No: add current year
            line += str(year)
        line += m.group(3).replace("ARM", "Arm")
        if("SConscript" in filename):
            f.write('#!/usr/bin/python\n')

        f.write('# -*- coding: utf-8 -*-\n\n')
        f.write(line+"\n")
        # Copy the rest of the file's content:
        f.write("".join(content[start + 1:]))
        f.close()

        return

    # This only works until year 9999
    m = re.match(r"(.*Copyright \(c\) )(.*\d{4})( [Arm|ARM].*)", content[1])
    start =len(ref)+2
    if content[0] != "/*\n" or not m:
        start = 0
        f.write("/*\n * Copyright (c) %d Arm Limited.\n" % year)
    else:
        logger.debug("Found Copyright start")
        logger.debug("\n\t".join([ g or "" for g in m.groups()]))
        line = m.group(1)

        if m.group(2): # Is there a year already?
            # Yes: adjust accordingly
            line += adjust_copyright_year(m.group(2), year)
        else:
            # No: add current year
            line += str(year)
        line += m.group(3).replace("ARM", "Arm")
        f.write("/*\n"+line+"\n")
        logger.debug(line)
    # Write out the rest of the Copyright header:
    for i in range(1, len(ref)):
        line = ref[i]
        f.write(" *")
        if line.rstrip() != "":
            f.write(" %s" % line)
        else:
            f.write("\n")
    f.write(" */\n")
    # Copy the rest of the file's content:
    f.write("".join(content[start:]))
    f.close()

def check_license(filename):
    """
    Check that the license file is up-to-date
    """
    f = open(filename, "r")
    content = f.readlines()
    f.close()

    f = open(filename, "w")
    f.write("".join(content[:2]))

    year = datetime.datetime.now().year
    # This only works until year 9999
    m = re.match(r"(.*Copyright \(c\) )(.*\d{4})( [Arm|ARM].*)", content[2])

    if not m:
        f.write("Copyright (c) {} Arm Limited\n".format(year))
    else:
        updated_year = adjust_copyright_year(m.group(2), year)
        f.write("Copyright (c) {} Arm Limited\n".format(updated_year))

    # Copy the rest of the file's content:
    f.write("".join(content[3:]))
    f.close()


class OtherChecksRun:
    def __init__(self, folder, error_diff=False, strategy="all"):
        self.folder = folder
        self.error_diff=error_diff
        self.strategy = strategy

    def error_on_diff(self, msg):
        retval = 0
        if self.error_diff:
            diff = self.shell.run_single_to_str("git diff")
            if len(diff) > 0:
                retval = -1
                logger.error(diff)
                logger.error("\n"+msg)
        return retval

    def run(self):
        retval = 0
        self.shell = Shell()
        self.shell.save_cwd()
        this_dir = os.path.dirname(__file__)
        self.shell.cd(self.folder)
        self.shell.prepend_env("PATH","%s/../bin" % this_dir)

        to_check = ""
        if self.strategy != "all":
            to_check, skip_copyright = FormatCodeRun.get_files(self.folder, self.strategy)
            #FIXME: Exclude shaders!

        logger.info("Running ./scripts/format_doxygen.py")
        logger.debug(self.shell.run_single_to_str("./scripts/format_doxygen.py %s" % " ".join(to_check)))
        retval = self.error_on_diff("Doxygen comments badly formatted (check above diff output for more details) try to run ./scripts/format_doxygen.py on your patch and resubmit")
        if retval == 0:
            logger.info("Running ./scripts/include_functions_kernels.py")
            logger.debug(self.shell.run_single_to_str("python ./scripts/include_functions_kernels.py"))
            retval = self.error_on_diff("Some kernels or functions are not included in their corresponding master header (check above diff output to see which includes are missing)")
        if retval == 0:
            try:
                logger.info("Running ./scripts/check_bad_style.sh")
                logger.debug(self.shell.run_single_to_str("./scripts/check_bad_style.sh"))
                #logger.debug(self.shell.run_single_to_str("./scripts/check_bad_style.sh %s" % " ".join(to_check)))
            except subprocess.CalledProcessError as e:
                logger.error("Command %s returned:\n%s" % (e.cmd, e.output))
                retval -= 1

        if retval != 0:
            raise Exception("format-code failed with error code %d" % retval)

class FormatCodeRun:
    @staticmethod
    def get_files(folder, strategy="git-head"):
        shell = Shell()
        shell.cd(folder)
        skip_copyright = False
        if strategy == "git-head":
            cmd = "git diff-tree --no-commit-id --name-status -r HEAD | grep \"^[AMRT]\" | cut -f 2"
        elif strategy == "git-diff":
            cmd = "git diff --name-status --cached -r HEAD | grep \"^[AMRT]\" | rev | cut -f 1 | rev"
        else:
            cmd = "git ls-tree -r HEAD --name-only"
            # Skip copyright checks when running on all files because we don't know when they were last modified
            # Therefore we can't tell if their copyright dates are correct
            skip_copyright = True

        grep_folder = "grep -e \"^\\(arm_compute\\|src\\|examples\\|tests\\|utils\\|support\\)/\""
        grep_extension = "grep -e \"\\.\\(cpp\\|h\\|hh\\|inl\\|cl\\|cs\\|hpp\\)$\""
        list_files = shell.run_single_to_str(cmd+" | { "+ grep_folder+" | "+grep_extension + " || true; }")
        to_check = [ f for f in list_files.split("\n") if len(f) > 0]

        # Check for scons files as they are excluded from the above list
        list_files = shell.run_single_to_str(cmd+" | { grep -e \"SC\" || true; }")
        to_check += [ f for f in list_files.split("\n") if len(f) > 0]

        return (to_check, skip_copyright)

    def __init__(self, files, folder, error_diff=False, skip_copyright=False):
        self.files = files
        self.folder = folder
        self.skip_copyright = skip_copyright
        self.error_diff=error_diff

    def error_on_diff(self, msg):
        retval = 0
        if self.error_diff:
            diff = self.shell.run_single_to_str("git diff")
            if len(diff) > 0:
                retval = -1
                logger.error(diff)
                logger.error("\n"+msg)
        return retval

    def run(self):
        if len(self.files) < 1:
            logger.debug("No file: early exit")
        retval = 0
        self.shell = Shell()
        self.shell.save_cwd()
        this_dir = os.path.dirname(__file__)
        try:
            self.shell.cd(self.folder)
            self.shell.prepend_env("PATH","%s/../bin" % this_dir)

            for f in self.files:
                if not self.skip_copyright:
                    check_copyright(f)

                skip_this_file = False
                for e in exceptions:
                    if e in f:
                        logger.warning("Skipping '%s' file: %s" % (e,f))
                        skip_this_file = True
                        break
                if skip_this_file:
                    continue

                logger.info("Formatting %s" % f)

            check_license("LICENSE")

        except subprocess.CalledProcessError as e:
            retval = -1
            logger.error(e)
            logger.error("OUTPUT= %s" % e.output)

        retval += self.error_on_diff("See above for clang-tidy errors")

        if retval != 0:
            raise Exception("format-code failed with error code %d" % retval)

class GenerateAndroidBP:
    def __init__(self, folder):
        self.folder = folder
        self.bp_output_file = "Generated_Android.bp"

    def run(self):
        retval = 0
        self.shell = Shell()
        self.shell.save_cwd()
        this_dir = os.path.dirname(__file__)

        logger.debug("Running Android.bp check")
        try:
            self.shell.cd(self.folder)
            cmd = "%s/generate_android_bp.py --folder %s --output_file %s" % (this_dir, self.folder, self.bp_output_file)
            output = self.shell.run_single_to_str(cmd)
            if len(output) > 0:
                logger.info(output)
        except subprocess.CalledProcessError as e:
            retval = -1
            logger.error(e)
            logger.error("OUTPUT= %s" % e.output)

        # Compare the genereated file with the one in the review
        if not filecmp.cmp(self.bp_output_file, self.folder + "/Android.bp"):
            is_mismatched = True

            with open(self.bp_output_file, 'r') as generated_file:
                with open(self.folder + "/Android.bp", 'r') as review_file:
                    diff = list(difflib.unified_diff(generated_file.readlines(), review_file.readlines(),
                                                     fromfile='Generated_Android.bp', tofile='Android.bp'))

                    # If the only mismatch in Android.bp file is the copyright year,
                    # the content of the file is considered unchanged and we don't need to update
                    # the copyright year. This will resolve the issue that emerges every new year.
                    num_added_lines = 0
                    num_removed_lines = 0
                    last_added_line = ""
                    last_removed_line = ""
                    expect_add_line = False

                    for line in diff:
                        if line.startswith("-") and not line.startswith("---"):
                            num_removed_lines += 1
                            if num_removed_lines > 1:
                                break
                            last_removed_line = line
                            expect_add_line = True
                        elif line.startswith("+") and not line.startswith("+++"):
                            num_added_lines += 1
                            if num_added_lines > 1:
                                break
                            if expect_add_line:
                                last_added_line = line
                        else:
                            expect_add_line = False

                    if num_added_lines == 1 and num_removed_lines == 1:
                        re_copyright = re.compile("^(?:\+|\-)// Copyright © ([0-9]+)\-([0-9]+) Arm Ltd. All rights reserved.\n$")
                        generated_matches = re_copyright.search(last_removed_line)
                        review_matches = re_copyright.search(last_added_line)

                        if generated_matches is not None and review_matches is not None:
                            if generated_matches.group(1) == review_matches.group(1) and \
                               int(generated_matches.group(2)) > int(review_matches.group(2)):
                                is_mismatched = False

                    if is_mismatched:
                        logger.error("Lines with '-' need to be added to Android.bp")
                        logger.error("Lines with '+' need to be removed from Android.bp")

                        for line in diff:
                            logger.error(line.rstrip())
            if is_mismatched:
                raise Exception("Android bp file is not updated")

        if retval != 0:
            raise Exception("generate Android bp file failed with error code %d" % retval)

def run_fix_code_formatting( files="git-head", folder=".", num_threads=1, error_on_diff=True):
    try:
        retval = 0

        # Genereate Android.bp file and test it
        gen_android_bp = GenerateAndroidBP(folder)
        gen_android_bp.run()

        to_check, skip_copyright = FormatCodeRun.get_files(folder, files)
        other_checks = OtherChecksRun(folder,error_on_diff, files)
        other_checks.run()

        logger.debug(to_check)
        num_files = len(to_check)
        per_thread = max( num_files / num_threads,1)
        start=0
        logger.info("Files to format:\n\t%s" % "\n\t".join(to_check))

        for i in range(num_threads):
            if i == num_threads -1:
                end = num_files
            else:
                end= min(start+per_thread, num_files)
            sub = to_check[start:end]
            logger.debug("[%d] [%d,%d] %s" % (i, start, end, sub))
            start = end
            format_code_run = FormatCodeRun(sub, folder, skip_copyright=skip_copyright)
            format_code_run.run()

        return retval
    except Exception as e:
        logger.error("Exception caught in run_fix_code_formatting: %s" % e)
        return -1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description="Build & run pre-commit tests",
    )

    file_sources=["git-diff","git-head","all"]
    parser.add_argument("-D", "--debug", action='store_true', help="Enable script debugging output")
    parser.add_argument("--error_on_diff", action='store_true', help="Show diff on error and stop")
    parser.add_argument("--files", nargs='?', metavar="source", choices=file_sources, help="Which files to run fix_code_formatting on, choices=%s" % file_sources, default="git-head")
    parser.add_argument("--folder", metavar="path", help="Folder in which to run fix_code_formatting", default=".")

    args = parser.parse_args()

    logging_level = logging.INFO
    if args.debug:
        logging_level = logging.DEBUG

    logging.basicConfig(level=logging_level)

    logger.debug("Arguments passed: %s" % str(args.__dict__))

    exit(run_fix_code_formatting(args.files, args.folder, 1, error_on_diff=args.error_on_diff))
