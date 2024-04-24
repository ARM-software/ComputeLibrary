#!/usr/bin/env python

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

import os
import logging
import subprocess

logger = logging.getLogger("Shell")

class Shell:
    def __init__(self, is_interactive=False):
        self.line=""
        self.env=os.environ.copy()
        self.initial_path = self.env["PATH"]
        self.save_cwd()
        self.is_interactive = is_interactive

    def reset_path(self):
        self.env["PATH"]=self.initial_path

    def set_env(self, key, value):
        self.env[key] = value

    def append_env(self, key, value):
        logger.debug("Appending '%s' to '%s'" % (value, key))
        if key not in list(self.env.keys()):
            self.set_env(key,value)
        else:
            self.env[key] += ":"+value
    def prepend_env(self, key, value):
        logger.debug("Prepending '%s' to '%s'" % (value, key))
        if key not in list(self.env.keys()):
            self.set_env(key,value)
        else:
            self.env[key] = value+":"+self.env[key]
    def run(self, cmd):
        if isinstance(cmd, list):
            for c in cmd:
                self.run_single(c)
        else:
            self.run_single(cmd)
    def run_to_str(self, cmd):
        out = ""
        if isinstance(cmd, list):
            for c in cmd:
                out += self.run_single_to_str(c)
        else:
            out = self.run_single_to_str(cmd)
        return out
    def cd(self, dirname):
        os.chdir(dirname)

    def save_cwd(self):
        self.cwd = os.getcwd()

    def restore_cwd(self):
        self.cd( self.cwd )

    def run_single_interactive(self,cmd):
        subprocess.check_call(cmd, env=self.env,stderr=subprocess.STDOUT, shell=True)
        logger.debug("%s returned" % cmd)

    def run_single(self,cmd):
        if self.is_interactive:
            self.run_single_interactive(cmd)
        else:
            self.run_single_to_str(cmd)

    def run_single_to_str_no_output_check(self,cmd):
        try:
            out = subprocess.check_output(cmd, env=self.env, stderr=subprocess.STDOUT, shell=True)
        except subprocess.CalledProcessError as cpe:
            out = cpe.output
        if (len(out.strip()) > 0):
            logger.debug(out)
        logger.debug("%s returned" % cmd)
        return out

    def run_single_to_str(self,cmd):
        out = subprocess.check_output(cmd, env=self.env, stderr=subprocess.STDOUT, shell=True).decode('utf-8')
        if (len(out.strip()) > 0):
            logger.debug(out)
        logger.debug("%s returned" % cmd)
        return out
