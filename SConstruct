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

import SCons
import os
import subprocess

def version_at_least(version, required):
    end = min(len(version), len(required))

    for i in range(0, end, 2):
        if int(version[i]) < int(required[i]):
            return False
        elif int(version[i]) > int(required[i]):
            return True

    return True

vars = Variables("scons")
vars.AddVariables(
    BoolVariable("debug", "Debug", False),
    BoolVariable("asserts", "Enable asserts (this flag is forced to 1 for debug=1)", False),
    BoolVariable("logging", "Logging (this flag is forced to 1 for debug=1)", False),
    EnumVariable("arch", "Target Architecture", "armv7a", allowed_values=("armv7a", "arm64-v8a", "arm64-v8.2-a", "x86_32", "x86_64")),
    EnumVariable("os", "Target OS", "linux", allowed_values=("linux", "android", "bare_metal")),
    EnumVariable("build", "Build type", "cross_compile", allowed_values=("native", "cross_compile")),
    BoolVariable("examples", "Build example programs", True),
    BoolVariable("Werror", "Enable/disable the -Werror compilation flag", True),
    BoolVariable("standalone", "Builds the tests as standalone executables, links statically with libgcc, libstdc++ and libarm_compute", False),
    BoolVariable("opencl", "Enable OpenCL support", True),
    BoolVariable("neon", "Enable Neon support", False),
    BoolVariable("gles_compute", "Enable OpenGL ES Compute Shader support", False),
    BoolVariable("embed_kernels", "Embed OpenCL kernels and OpenGL ES compute shaders in library binary", True),
    BoolVariable("set_soname", "Set the library's soname and shlibversion (requires SCons 2.4 or above)", False),
    BoolVariable("openmp", "Enable OpenMP backend", False),
    BoolVariable("cppthreads", "Enable C++11 threads backend", True),
    PathVariable("build_dir", "Specify sub-folder for the build", ".", PathVariable.PathAccept),
    ("extra_cxx_flags", "Extra CXX flags to be appended to the build command", "")
)

env = Environment(platform="posix", variables=vars, ENV = os.environ)
env.Append(LIBPATH = ["#build/%s" % env['build_dir']])

SConsignFile('build/.%s' % env['build_dir'])

Help(vars.GenerateHelpText(env))

if env['neon'] and 'x86' in env['arch']:
    print "Cannot compile NEON for x86"
    Exit(1)

if env['set_soname'] and not version_at_least(SCons.__version__, "2.4"):
    print "Setting the library's SONAME / SHLIBVERSION requires SCons 2.4 or above"
    print "Update your version of SCons or use set_soname=0"
    Exit(1)

if env['os'] == 'bare_metal':
    if env['cppthreads'] or env['openmp']:
         print("ERROR: OpenMP and C++11 threads not supported in bare_metal. Use cppthreads=0 openmp=0")
         Exit(1)

env.Append(CXXFLAGS = ['-Wno-deprecated-declarations','-Wall','-DARCH_ARM',
         '-Wextra','-Wno-unused-parameter','-pedantic','-Wdisabled-optimization','-Wformat=2',
         '-Winit-self','-Wstrict-overflow=2','-Wswitch-default',
         '-fpermissive','-std=gnu++11','-Wno-vla','-Woverloaded-virtual',
         '-Wctor-dtor-privacy','-Wsign-promo','-Weffc++','-Wno-format-nonliteral','-Wno-overlength-strings','-Wno-strict-overflow'])

env.Append(CPPDEFINES = ['_GLIBCXX_USE_NANOSLEEP'])

default_cpp_compiler = 'g++' if env['os'] != 'android' else 'clang++'
default_c_compiler = 'gcc' if env['os'] != 'android' else 'clang'
cpp_compiler = os.environ.get('CXX', default_cpp_compiler)
c_compiler = os.environ.get('CC', default_c_compiler)

if env['os'] == 'android' and ( cpp_compiler != 'clang++' or c_compiler != 'clang'):
    print "WARNING: Only clang is officially supported to build the Compute Library for Android"

if cpp_compiler == 'clang++':
    env.Append(CXXFLAGS = ['-Wno-format-nonliteral','-Wno-deprecated-increment-bool','-Wno-vla-extension','-Wno-mismatched-tags'])
else:
    env.Append(CXXFLAGS = ['-Wlogical-op','-Wnoexcept','-Wstrict-null-sentinel'])

if env['cppthreads']:
    env.Append(CPPDEFINES = [('ARM_COMPUTE_CPP_SCHEDULER', 1)])

if env['openmp']:
    if cpp_compiler == 'clang++':
        print "Clang does not support OpenMP. Use scheduler=cpp."
        Exit(1)

    env.Append(CPPDEFINES = [('ARM_COMPUTE_OPENMP_SCHEDULER', 1)])
    env.Append(CXXFLAGS = ['-fopenmp'])
    env.Append(LINKFLAGS = ['-fopenmp'])

prefix = ""
if env['arch'] == 'armv7a':
    env.Append(CXXFLAGS = ['-march=armv7-a', '-mthumb', '-mfpu=neon'])

    if env['os'] == 'linux':
        prefix = "arm-linux-gnueabihf-"
        env.Append(CXXFLAGS = ['-mfloat-abi=hard'])
    elif env['os'] == 'bare_metal':
        prefix = "arm-eabi-"
        env.Append(CXXFLAGS = ['-mfloat-abi=hard'])
    elif env['os'] == 'android':
        prefix = "arm-linux-androideabi-"
        env.Append(CXXFLAGS = ['-mfloat-abi=softfp'])
elif env['arch'] == 'arm64-v8a':
    env.Append(CXXFLAGS = ['-march=armv8-a'])
    env.Append(CPPDEFINES = ['ARM_COMPUTE_AARCH64_V8A'])
    if env['os'] == 'linux':
        prefix = "aarch64-linux-gnu-"
    elif env['os'] == 'bare_metal':
        prefix = "aarch64-elf-"
    elif env['os'] == 'android':
        prefix = "aarch64-linux-android-"
elif env['arch'] == 'arm64-v8.2-a':
    env.Append(CPPDEFINES = ['ARM_COMPUTE_AARCH64_V8_2'])

    if cpp_compiler == 'clang++':
        env.Append(CXXFLAGS = ['-fno-integrated-as'])

    if env['os'] == 'linux':
        prefix = "aarch64-linux-gnu-"
    elif env['os'] == 'bare_metal':
        prefix = "aarch64-elf-"
    elif env['os'] == 'android':
        prefix = "aarch64-linux-android-"
elif env['arch'] == 'x86_32':
    env.Append(CCFLAGS = ['-m32'])
    env.Append(LINKFLAGS = ['-m32'])
elif env['arch'] == 'x86_64':
    env.Append(CCFLAGS = ['-m64'])
    env.Append(LINKFLAGS = ['-m64'])

if env['build'] == 'native':
    prefix = ""

env['CC'] = prefix + c_compiler
env['CXX'] = prefix + cpp_compiler
env['LD'] = prefix + "ld"
env['AS'] = prefix + "as"
env['AR'] = prefix + "ar"
env['RANLIB'] = prefix + "ranlib"

if not GetOption("help"):
    try:
        compiler_ver = subprocess.check_output(env['CXX'].split() + ["-dumpversion"]).strip()
    except OSError:
        print("ERROR: Compiler '%s' not found" % env['CXX'])
        Exit(1)

    if cpp_compiler == 'g++':
        if env['arch'] == 'arm64-v8.2-a' and not version_at_least(compiler_ver, '6.2.1'):
            print "GCC 6.2.1 or newer is required to compile armv8.2-a code"
            Exit(1)
        elif env['arch'] == 'arm64-v8a' and not version_at_least(compiler_ver, '4.9'):
            print "GCC 4.9 or newer is required to compile NEON code for AArch64"
            Exit(1)

        if version_at_least(compiler_ver, '6.1'):
            env.Append(CXXFLAGS = ['-Wno-ignored-attributes'])

        if compiler_ver == '4.8.3':
            env.Append(CXXFLAGS = ['-Wno-array-bounds'])

if env['standalone']:
    env.Append(CXXFLAGS = ['-fPIC'])
    env.Append(LINKFLAGS = ['-static-libgcc','-static-libstdc++'])
    if env['cppthreads']:
        env.Append(LINKFLAGS = ['-lpthread'])

if env['Werror']:
    env.Append(CXXFLAGS = ['-Werror'])

if env['os'] == 'android':
    env.Append(CPPDEFINES = ['ANDROID'])
    env.Append(LINKFLAGS = ['-pie', '-static-libstdc++'])
elif env['os'] == 'bare_metal':
    env.Append(LINKFLAGS = ['-static'])
    env.Append(LINKFLAGS = ['-specs=rdimon.specs'])
    env.Append(CXXFLAGS = ['-fPIC'])
    env.Append(CPPDEFINES = ['NO_MULTI_THREADING'])
    env.Append(CPPDEFINES = ['BARE_METAL'])

if env['opencl']:
    if env['os'] in ['bare_metal'] or env['standalone']:
        print("Cannot link OpenCL statically, which is required on bare metal")
        Exit(1)

if env['opencl'] or env['gles_compute']:
    if env['embed_kernels']:
        env.Append(CPPDEFINES = ['EMBEDDED_KERNELS'])

if env['debug']:
    env['asserts'] = True
    env['logging'] = True
    env.Append(CXXFLAGS = ['-O0','-g','-gdwarf-2'])
    env.Append(CPPDEFINES = ['ARM_COMPUTE_DEBUG_ENABLED'])
else:
    env.Append(CXXFLAGS = ['-O3','-ftree-vectorize'])

if env['asserts']:
    env.Append(CPPDEFINES = ['ARM_COMPUTE_ASSERTS_ENABLED'])
    env.Append(CXXFLAGS = ['-fstack-protector-strong'])

if env['logging']:
    env.Append(CPPDEFINES = ['ARM_COMPUTE_LOGGING_ENABLED'])

env.Append(CPPPATH = ['#/include', "#"])
env.Append(CXXFLAGS = env['extra_cxx_flags'])

Export('vars')
Export('env')
Export('version_at_least')

if env['opencl']:
    SConscript("./opencl-1.2-stubs/SConscript", variant_dir="build/%s/opencl-1.2-stubs" % env['build_dir'], duplicate=0)

if env['gles_compute'] and env['os'] != 'android':
    env.Append(CPPPATH = ['#/include/linux'])
    env.Append(LIBPATH = ["#build/%s/opengles-3.1-stubs" % env['build_dir']])
    SConscript("./opengles-3.1-stubs/SConscript", variant_dir="build/%s/opengles-3.1-stubs" % env['build_dir'], duplicate=0)

SConscript('./SConscript', variant_dir='#build/%s' % env['build_dir'], duplicate=0)

if env['examples'] and env['os'] != 'bare_metal':
    SConscript('./examples/SConscript', variant_dir='#build/%s/examples' % env['build_dir'], duplicate=0)

if env['os'] != 'bare_metal':
    SConscript('./tests/SConscript', variant_dir='#build/%s/tests' % env['build_dir'], duplicate=0)
