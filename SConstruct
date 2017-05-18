# Copyright (c) 2016, 2017 ARM Limited.
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

vars = Variables("scons")
vars.AddVariables(
    BoolVariable("debug", "Debug", False),
    BoolVariable("asserts", "Enable asserts (this flag is forced to 1 for debug=1)", False),
    EnumVariable("arch", "Target Architecture", "armv7a", allowed_values=("armv7a", "arm64-v8a", "arm64-v8.2-a", "x86_32", "x86_64")),
    EnumVariable("os", "Target OS", "linux", allowed_values=("linux", "android", "bare_metal")),
    EnumVariable("build", "Build type", "cross_compile", allowed_values=("native", "cross_compile")),
    BoolVariable("examples", "Build example programs", False),
    BoolVariable("Werror", "Enable/disable the -Werror compilation flag", True),
    BoolVariable("opencl", "Enable OpenCL support", True),
    BoolVariable("neon", "Enable Neon support", False),
    BoolVariable("embed_kernels", "Embed OpenCL kernels in library binary", False),
    BoolVariable("set_soname", "Set the library's soname and shlibversion (requires SCons 2.4 or above)", False),
    BoolVariable("openmp", "Enable OpenMP backend", False),
    BoolVariable("cppthreads", "Enable C++11 threads backend", True),
    PathVariable("build_dir", "Specify sub-folder for the build", ".", PathVariable.PathIsDirCreate),
    ("extra_cxx_flags", "Extra CXX flags to be appended to the build command", "")
)

env = Environment(platform='posix', variables = vars, ENV = os.environ)

Help(vars.GenerateHelpText(env))

Export('vars')
Export('env')

if not GetOption("help"):
    SConscript('sconscript', variant_dir='#build/%s/arm_compute' % env['build_dir'], duplicate=0)
