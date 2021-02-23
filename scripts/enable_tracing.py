#
# Copyright (c) 2020 Arm Limited.
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
#
#
#!/usr/bin/env python
import re
import os
import sys
import argparse
import fnmatch
import logging

import json
import glob

logger = logging.getLogger("acl_tracing")

# Returns the files matching the given pattern
def find(path, pattern):
    matches = []
    for root, dirnames, filenames, in os.walk(path):
        for filename in fnmatch.filter(filenames, pattern):
            matches.append(os.path.join(root,filename))
    return matches

# Returns the class name (Core or Runtime) and arguments of the given function
def get_class_and_args(function):
    decl = " ".join(function_signature)
    m = re.match("void ([^:]+)::configure\(([^)]*)\)", decl)
    if m:
            assert m, "Can't parse '%s'" % line
            class_name = m.group(1)
            args = m.group(2)
            #Remove comments:
            args = re.sub("\/\*.*?\*\/","",args)
            #Remove templates
            args = re.sub("<.*?>","",args)
            logger.debug(args)
            arg_names = []
            for arg in args.split(","):
                m = re.match(".*?([^ &*]+)$", arg.strip())
                arg_names.append(m.group(1))
                logger.debug("  %s" % m.group(1))
            return (class_name, arg_names)
    else:
        return ('','')

# Adds the tracepoints to the source file for the given function
def do_insert_tracing(source, function, fd):
    logger.debug("Full signature = %s" % " ".join(function_signature))
    class_name, arg_names = get_class_and_args(function)
    if len(arg_names):
        assert len(arg_names), "No argument to configure for %s ?" % class_name
        spaces = re.match("([ ]*)void", function[0]).group(1)
        fd.write("%s    ARM_COMPUTE_CREATE_TRACEPOINT(%s, \"%s\", this, TracePoint::Args()" % (spaces, source, class_name))
        for arg in arg_names:
            fd.write("<<%s" % arg)
        fd.write(");\n")
    else:
        print('Failed to get class name in %s ' % " ".join(function_signature))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description="Post process JSON benchmark files",
    )

    parser.add_argument("-D", "--debug", action='store_true', help="Enable script debugging output")
    args = parser.parse_args()
    logging_level = logging.INFO
    if args.debug:
        logging_level = logging.DEBUG
    logging.basicConfig(level=logging_level)
    logger.debug("Arguments passed: %s" % str(args.__dict__))
    for f in find("src","*.cpp"):
        logger.debug(f)
        fd = open(f,'r+')
        lines = fd.readlines()
        contains_configure = False
        for line in lines:
            if re.search(r"void.*::configure\(",line):
                contains_configure = True
                break
        if not contains_configure:
            continue
        fd.seek(0)
        fd.truncate()
        function_signature = None
        insert_tracing = False
        start = True
        for line in lines:
            write = True
            if start:
                if not (line.startswith("/*") or line.startswith(" *") or line.startswith("#") or len(line.strip()) == 0):
                    start = False
                    fd.write("#include \"arm_compute/core/TracePoint.h\"\n")
            elif not function_signature:
                if re.search(r"void.*::configure\(",line):
                    function_signature = [ line.rstrip() ]
            else:
                if re.search("[ ]*{$", line):
                    insert_tracing = True
                else:
                    function_signature.append(line.rstrip())
            if write:
                fd.write(line)
            if insert_tracing:
                if "/core/" in f:
                    source = "TracePoint::Layer::CORE"
                elif "/runtime/" in f:
                    source = "TracePoint::Layer::RUNTIME"
                else:
                    assert "Can't find layer for file %s" %f
                do_insert_tracing(source, function_signature, fd)
                insert_tracing = False
                function_signature = None
