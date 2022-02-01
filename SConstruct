# -*- coding: utf-8 -*-

# Copyright (c) 2016-2021 Arm Limited.
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
import json
import os
import subprocess
import sys

def version_at_least(version, required):

    version_list = version.split('.')
    required_list = required.split('.')
    end = min(len(version_list), len(required_list))
    for i in range(0, end):
        if int(version_list[i]) < int(required_list[i]):
            return False
        elif int(version_list[i]) > int(required_list[i]):
            return True

    return True

def read_build_config_json(build_config):
    build_config_contents = {}
    custom_types = []
    custom_layouts = []
    if os.path.isfile(build_config):
        with open(build_config) as f:
            try:
                build_config_contents = json.load(f)
            except:
                print("Warning: Build configuration file is of invalid JSON format!")
    else:
        try:
            build_config_contents = json.loads(build_config)
        except:
            print("Warning: Build configuration string is of invalid JSON format!")
    if build_config_contents:
        custom_types = build_config_contents.get("data_types", [])
        custom_layouts = build_config_contents.get("data_layouts", [])
    return custom_types, custom_layouts

def update_data_type_layout_flags(env, data_types, data_layouts):
    # Manage data-types
    if any(i in data_types for i in ['all', 'fp16']):
        env.Append(CXXFLAGS = ['-DENABLE_FP16_KERNELS'])
    if any(i in data_types for i in ['all', 'fp32']):
        env.Append(CXXFLAGS = ['-DENABLE_FP32_KERNELS'])
    if any(i in data_types for i in ['all', 'qasymm8']):
        env.Append(CXXFLAGS = ['-DENABLE_QASYMM8_KERNELS'])
    if any(i in data_types for i in ['all', 'qasymm8_signed']):
        env.Append(CXXFLAGS = ['-DENABLE_QASYMM8_SIGNED_KERNELS'])
    if any(i in data_types for i in ['all', 'qsymm16']):
        env.Append(CXXFLAGS = ['-DENABLE_QSYMM16_KERNELS'])
    if any(i in data_types for i in ['all', 'integer']):
        env.Append(CXXFLAGS = ['-DENABLE_INTEGER_KERNELS'])

    # Manage data-layouts
    if any(i in data_layouts for i in ['all', 'nhwc']):
        env.Append(CXXFLAGS = ['-DENABLE_NHWC_KERNELS'])
    if any(i in data_layouts for i in ['all', 'nchw']):
        env.Append(CXXFLAGS = ['-DENABLE_NCHW_KERNELS'])

    return env


vars = Variables("scons")
vars.AddVariables(
    BoolVariable("debug", "Debug", False),
    BoolVariable("asserts", "Enable asserts (this flag is forced to 1 for debug=1)", False),
    BoolVariable("logging", "Enable Logging", False),
    EnumVariable("arch", "Target Architecture", "armv7a",
                  allowed_values=("armv7a", "armv7a-hf", "arm64-v8a", "arm64-v8.2-a", "arm64-v8.2-a-sve", "arm64-v8.2-a-sve2", "x86_32", "x86_64",
                                  "armv8a", "armv8.2-a", "armv8.2-a-sve", "armv8.6-a", "armv8.6-a-sve", "armv8.6-a-sve2", "armv8r64", "x86")),
    EnumVariable("estate", "Execution State", "auto", allowed_values=("auto", "32", "64")),
    EnumVariable("os", "Target OS", "linux", allowed_values=("linux", "android", "tizen", "macos", "bare_metal", "openbsd")),
    EnumVariable("build", "Build type", "cross_compile", allowed_values=("native", "cross_compile", "embed_only")),
    BoolVariable("examples", "Build example programs", True),
    BoolVariable("gemm_tuner", "Build gemm_tuner programs", True),
    BoolVariable("Werror", "Enable/disable the -Werror compilation flag", True),
    BoolVariable("multi_isa", "Build Multi ISA binary version of library. Note works only for armv8.2-a", False),
    BoolVariable("standalone", "Builds the tests as standalone executables, links statically with libgcc, libstdc++ and libarm_compute", False),
    BoolVariable("opencl", "Enable OpenCL support", True),
    BoolVariable("neon", "Enable Arm® Neon™ support", False),
    BoolVariable("embed_kernels", "Embed OpenCL kernels and OpenGL ES compute shaders in library binary", True),
    BoolVariable("compress_kernels", "Compress embedded OpenCL kernels in library binary. Note embed_kernels should be enabled", False),
    BoolVariable("set_soname", "Set the library's soname and shlibversion (requires SCons 2.4 or above)", False),
    BoolVariable("openmp", "Enable OpenMP backend", False),
    BoolVariable("cppthreads", "Enable C++11 threads backend", True),
    PathVariable("build_dir", "Specify sub-folder for the build", ".", PathVariable.PathAccept),
    PathVariable("install_dir", "Specify sub-folder for the install", "", PathVariable.PathAccept),
    BoolVariable("exceptions", "Enable/disable C++ exception support", True),
    BoolVariable("high_priority", "Generate a library containing only the high priority operators", False),
    PathVariable("linker_script", "Use an external linker script", "", PathVariable.PathAccept),
    PathVariable("external_tests_dir", "Add examples, benchmarks and tests to the tests suite", "", PathVariable.PathAccept),
    ListVariable("custom_options", "Custom options that can be used to turn on/off features", "none", ["disable_mmla_fp"]),
    ListVariable("data_type_support", "Enable a list of data types to support", "all", ["qasymm8", "qasymm8_signed", "qsymm16", "fp16", "fp32", "integer"]),
    ListVariable("data_layout_support", "Enable a list of data layout to support", "all", ["nhwc", "nchw"]),
    ("toolchain_prefix", "Override the toolchain prefix", ""),
    ("compiler_prefix", "Override the compiler prefix", ""),
    ("extra_cxx_flags", "Extra CXX flags to be appended to the build command", ""),
    ("extra_link_flags", "Extra LD flags to be appended to the build command", ""),
    ("compiler_cache", "Command to prefix to the C and C++ compiler (e.g ccache)", ""),
    ("specs_file", "Specs file to use (e.g. rdimon.specs)", ""),
    ("build_config", "Operator/Data-type/Data-layout configuration to use for tailored ComputeLibrary builds. Can be a JSON file or a JSON formatted string", "")
)


env = Environment(platform="posix", variables=vars, ENV = os.environ)
build_path = env['build_dir']
# If build_dir is a relative path then add a #build/ prefix:
if not env['build_dir'].startswith('/'):
    SConsignFile('build/%s/.scons' % build_path)
    build_path = "#build/%s" % build_path
else:
    SConsignFile('%s/.scons' % build_path)

install_path = env['install_dir']
#If the install_dir is a relative path then assume it's from inside build_dir
if not env['install_dir'].startswith('/') and install_path != "":
    install_path = "%s/%s" % (build_path, install_path)

env.Append(LIBPATH = [build_path])
Export('env')
Export('vars')

def install_lib( lib ):
    # If there is no install folder, then there is nothing to do:
    if install_path == "":
        return lib
    return env.Install( "%s/lib/" % install_path, lib)
def install_bin( bin ):
    # If there is no install folder, then there is nothing to do:
    if install_path == "":
        return bin
    return env.Install( "%s/bin/" % install_path, bin)
def install_include( inc ):
    if install_path == "":
        return inc
    return env.Install( "%s/include/" % install_path, inc)

Export('install_lib')
Export('install_bin')

Help(vars.GenerateHelpText(env))

if env['linker_script'] and env['os'] != 'bare_metal':
    print("Linker script is only supported for bare_metal builds")
    Exit(1)

if env['build'] == "embed_only":
    SConscript('./SConscript', variant_dir=build_path, duplicate=0)
    Return()

if env['neon'] and 'x86' in env['arch']:
    print("Cannot compile Arm® Neon™ for x86")
    Exit(1)

if env['set_soname'] and not version_at_least(SCons.__version__, "2.4"):
    print("Setting the library's SONAME / SHLIBVERSION requires SCons 2.4 or above")
    print("Update your version of SCons or use set_soname=0")
    Exit(1)

if env['os'] == 'bare_metal':
    if env['cppthreads'] or env['openmp']:
         print("ERROR: OpenMP and C++11 threads not supported in bare_metal. Use cppthreads=0 openmp=0")
         Exit(1)

if env['opencl'] and env['embed_kernels'] and env['compress_kernels'] and env['os'] not in ['android']:
    print("Compressed kernels are supported only for android builds")
    Exit(1)

if not env['exceptions']:
    if env['opencl']:
         print("ERROR: OpenCL is not supported when building without exceptions. Use opencl=0")
         Exit(1)

    env.Append(CPPDEFINES = ['ARM_COMPUTE_EXCEPTIONS_DISABLED'])
    env.Append(CXXFLAGS = ['-fno-exceptions'])

env.Append(CXXFLAGS = ['-Wall','-DARCH_ARM',
         '-Wextra','-pedantic','-Wdisabled-optimization','-Wformat=2',
         '-Winit-self','-Wstrict-overflow=2','-Wswitch-default',
         '-std=c++14','-Woverloaded-virtual', '-Wformat-security',
         '-Wctor-dtor-privacy','-Wsign-promo','-Weffc++','-Wno-overlength-strings'])

env.Append(CPPDEFINES = ['_GLIBCXX_USE_NANOSLEEP'])

default_cpp_compiler = 'g++' if env['os'] not in ['android', 'macos', 'openbsd'] else 'clang++'
default_c_compiler = 'gcc' if env['os'] not in ['android', 'macos', 'openbsd'] else 'clang'
cpp_compiler = os.environ.get('CXX', default_cpp_compiler)
c_compiler = os.environ.get('CC', default_c_compiler)

if env['os'] == 'android' and ( 'clang++' not in cpp_compiler or 'clang' not in c_compiler ):
    print( "WARNING: Only clang is officially supported to build the Compute Library for Android")

if 'clang++' in cpp_compiler:
    env.Append(CXXFLAGS = ['-Wno-vla-extension'])
elif 'armclang' in cpp_compiler:
    pass
else:
    env.Append(CXXFLAGS = ['-Wlogical-op','-Wnoexcept','-Wstrict-null-sentinel'])

if cpp_compiler == 'g++':
    # Don't strip comments that could include markers
    env.Append(CXXFLAGS = ['-C'])

if env['cppthreads']:
    env.Append(CPPDEFINES = [('ARM_COMPUTE_CPP_SCHEDULER', 1)])

if env['openmp']:
    env.Append(CPPDEFINES = [('ARM_COMPUTE_OPENMP_SCHEDULER', 1)])
    env.Append(CXXFLAGS = ['-fopenmp'])
    env.Append(LINKFLAGS = ['-fopenmp'])

# Validate and define state
if env['estate'] == 'auto':
    if 'v7a' in env['arch']:
        env['estate'] = '32'
    else:
        env['estate'] = '64'

# Map legacy arch
if 'arm64' in env['arch']:
    env['estate'] = '64'

if 'v7a' in env['estate'] and env['estate'] == '64':
    print("ERROR: armv7a architecture has only 32-bit execution state")
    Exit(1)

if 'sve' in env['arch']:
    env.Append(CPPDEFINES = ['ENABLE_SVE', 'ARM_COMPUTE_ENABLE_SVE'])
    if 'sve2' in env['arch']:
        env.Append(CPPDEFINES = ['ARM_COMPUTE_ENABLE_SVE2'])
else:
    env.Append(CPPDEFINES = ['ENABLE_NEON', 'ARM_COMPUTE_ENABLE_NEON'])

# Add architecture specific flags
prefix = ""
if env['multi_isa']:
    # assert arch version is v8
    if 'v8' not in env['arch']:
        print("Currently Multi ISA binary is only supported for arm v8 family")
        Exit(1)

    if 'v8.6-a' in env['arch']:
        if "disable_mmla_fp" not in env['custom_options']:
            env.Append(CPPDEFINES = ['ARM_COMPUTE_ENABLE_SVEF32MM'])

else: # NONE "multi_isa" builds

    if 'v7a' in env['arch']:
        env.Append(CXXFLAGS = ['-march=armv7-a', '-mthumb', '-mfpu=neon'])
        if (env['os'] == 'android' or env['os'] == 'tizen') and not 'hf' in env['arch']:
            env.Append(CXXFLAGS = ['-mfloat-abi=softfp'])
        else:
            env.Append(CXXFLAGS = ['-mfloat-abi=hard'])
    elif 'v8' in env['arch']:
        # Preserve the V8 archs for non-multi-ISA variants
        if 'sve2' in env['arch']:
            env.Append(CXXFLAGS = ['-march=armv8.2-a+sve2+fp16+dotprod'])
        elif 'sve' in env['arch']:
            env.Append(CXXFLAGS = ['-march=armv8.2-a+sve+fp16+dotprod'])
        elif 'armv8r64' in env['arch']:
            env.Append(CXXFLAGS = ['-march=armv8.4-a'])
        elif 'v8.' in env['arch']:
            env.Append(CXXFLAGS = ['-march=armv8.2-a+fp16']) # explicitly enable fp16 extension otherwise __ARM_FEATURE_FP16_VECTOR_ARITHMETIC is undefined
        else:
            env.Append(CXXFLAGS = ['-march=armv8-a'])

        if 'v8.6-a' in env['arch']:
            env.Append(CPPDEFINES = ['ARM_COMPUTE_ENABLE_I8MM', 'ARM_COMPUTE_ENABLE_BF16'])
            if "disable_mmla_fp" not in env['custom_options']:
                env.Append(CPPDEFINES = ['ARM_COMPUTE_ENABLE_SVEF32MM'])
        if 'v8.' in env['arch']:
            env.Append(CPPDEFINES = ['ARM_COMPUTE_ENABLE_FP16'])

    elif 'x86' in env['arch']:
        if env['estate'] == '32':
            env.Append(CCFLAGS = ['-m32'])
            env.Append(LINKFLAGS = ['-m32'])
        else:
            env.Append(CXXFLAGS = ['-fPIC'])
            env.Append(CCFLAGS = ['-m64'])
            env.Append(LINKFLAGS = ['-m64'])

# Define toolchain
prefix = ""
if 'x86' not in env['arch']:
    if env['estate'] == '32':
        if env['os'] == 'linux':
            prefix = "arm-linux-gnueabihf-" if 'v7' in env['arch'] else "armv8l-linux-gnueabihf-"
        elif env['os'] == 'bare_metal':
            prefix = "arm-eabi-"
        elif env['os'] == 'android':
            prefix = "arm-linux-androideabi-"
        elif env['os'] == 'tizen':
            prefix = "armv7l-tizen-linux-gnueabi-"
    elif env['estate'] == '64' and 'v8' in env['arch']:
        if env['os'] == 'linux':
            prefix = "aarch64-linux-gnu-"
        elif env['os'] == 'bare_metal':
            prefix = "aarch64-elf-"
        elif env['os'] == 'android':
            prefix = "aarch64-linux-android-"
        elif env['os'] == 'tizen':
            prefix = "aarch64-tizen-linux-gnu-"

if env['build'] == 'native':
    prefix = ""

if env["toolchain_prefix"] != "":
    prefix = env["toolchain_prefix"]

compiler_prefix = prefix
if env["compiler_prefix"] != "":
    compiler_prefix = env["compiler_prefix"]

env['CC'] = env['compiler_cache']+ " " + compiler_prefix + c_compiler
env['CXX'] = env['compiler_cache']+ " " + compiler_prefix + cpp_compiler
env['LD'] = prefix + "ld"
env['AS'] = prefix + "as"
env['AR'] = prefix + "ar"
env['RANLIB'] = prefix + "ranlib"

if not GetOption("help"):
    try:
        compiler_ver = subprocess.check_output(env['CXX'].split() + ["-dumpversion"]).decode().strip()
    except OSError:
        print("ERROR: Compiler '%s' not found" % env['CXX'])
        Exit(1)

    if 'armclang' in cpp_compiler:
        pass
    elif 'clang++' not in cpp_compiler:
        if env['arch'] == 'arm64-v8.2-a' and not version_at_least(compiler_ver, '6.2.1'):
            print("GCC 6.2.1 or newer is required to compile armv8.2-a code")
            Exit(1)
        elif env['arch'] == 'arm64-v8a' and not version_at_least(compiler_ver, '4.9'):
            print("GCC 4.9 or newer is required to compile Arm® Neon™ code for AArch64")
            Exit(1)

        if version_at_least(compiler_ver, '6.1'):
            env.Append(CXXFLAGS = ['-Wno-ignored-attributes'])

        if compiler_ver == '4.8.3':
            env.Append(CXXFLAGS = ['-Wno-array-bounds'])

        if not version_at_least(compiler_ver, '7.0.0') and env['os'] == 'bare_metal':
            env.Append(LINKFLAGS = ['-fstack-protector-strong'])

if env['high_priority'] and env['build_config']:
    print("The high priority library cannot be built in conjunction with a user-specified build configuration")
    Exit(1)

if not env['high_priority'] and not env['build_config']:
    env.Append(CPPDEFINES = ['ARM_COMPUTE_GRAPH_ENABLED'])

data_types = []
data_layouts = []

# Set correct data types / layouts to build
if env['high_priority']:
    data_types = ['all']
    data_layouts = ['all']
elif env['build_config']:
    data_types, data_layouts = read_build_config_json(env['build_config'])
else:
    data_types = env['data_type_support']
    data_layouts = env['data_layout_support']

env = update_data_type_layout_flags(env, data_types, data_layouts)

if env['standalone']:
    env.Append(CXXFLAGS = ['-fPIC'])
    env.Append(LINKFLAGS = ['-static-libgcc','-static-libstdc++'])

if env['Werror']:
    env.Append(CXXFLAGS = ['-Werror'])

if env['os'] == 'android':
    env.Append(CPPDEFINES = ['ANDROID'])
    env.Append(LINKFLAGS = ['-pie', '-static-libstdc++', '-ldl'])
elif env['os'] == 'bare_metal':
    env.Append(LINKFLAGS = ['-static'])
    env.Append(CXXFLAGS = ['-fPIC'])
    if env['specs_file'] == "":
        env.Append(LINKFLAGS = ['-specs=rdimon.specs'])
    env.Append(CPPDEFINES = ['NO_MULTI_THREADING'])
    env.Append(CPPDEFINES = ['BARE_METAL'])
if env['os'] == 'linux' and env['arch'] == 'armv7a':
    env.Append(CXXFLAGS = [ '-Wno-psabi' ])

if env['specs_file'] != "":
    env.Append(LINKFLAGS = ['-specs='+env['specs_file']])

if env['neon']:
    env.Append(CPPDEFINES = ['ARM_COMPUTE_CPU_ENABLED'])

if env['opencl']:
    env.Append(CPPDEFINES = ['ARM_COMPUTE_OPENCL_ENABLED'])
    if env['os'] in ['bare_metal'] or env['standalone']:
        print("Cannot link OpenCL statically, which is required for bare metal / standalone builds")
        Exit(1)

if env["os"] not in ["android", "bare_metal"] and (env['opencl'] or env['cppthreads']):
    env.Append(LIBS = ['pthread'])

if env['os'] == 'openbsd':
    env.Append(LIBS = ['c'])
    env.Append(CXXFLAGS = ['-fPIC'])

if env['opencl']:
    if env['embed_kernels']:
        env.Append(CPPDEFINES = ['EMBEDDED_KERNELS'])
    if env['compress_kernels']:
        env.Append(CPPDEFINES = ['ARM_COMPUTE_COMPRESSED_KERNELS'])
        env.Append(LIBS = ['z'])

if env['debug']:
    env['asserts'] = True
    env.Append(CXXFLAGS = ['-O0','-g','-gdwarf-2'])
    env.Append(CPPDEFINES = ['ARM_COMPUTE_DEBUG_ENABLED'])
else:
    env.Append(CXXFLAGS = ['-O3'])

if env['asserts']:
    env.Append(CPPDEFINES = ['ARM_COMPUTE_ASSERTS_ENABLED'])
    env.Append(CXXFLAGS = ['-fstack-protector-strong'])

if env['logging']:
    env.Append(CPPDEFINES = ['ARM_COMPUTE_LOGGING_ENABLED'])

env.Append(CPPPATH = ['#/include', "#"])
env.Append(CXXFLAGS = env['extra_cxx_flags'])
env.Append(LINKFLAGS = env['extra_link_flags'])

Default( install_include("arm_compute"))
Default( install_include("support"))
Default( install_include("utils"))
for dirname in os.listdir("./include"):
    Default( install_include("include/%s" % dirname))

Export('version_at_least')

SConscript('./SConscript', variant_dir=build_path, duplicate=0)

if env['examples'] and (env['build_config'] or env['high_priority']):
    print("WARNING: Building examples for selected operators not supported. Use examples=0")
    Return()

if env['examples'] and env['exceptions']:
    if env['os'] == 'bare_metal' and env['arch'] == 'armv7a':
        print("WARNING: Building examples for bare metal and armv7a is not supported. Use examples=0")
        Return()
    SConscript('./examples/SConscript', variant_dir='%s/examples' % build_path, duplicate=0)

if env['exceptions']:
    if env['build_config'] or env['high_priority']:
        print("WARNING: Building tests for selected operators not supported")
        Return()
    if env['os'] == 'bare_metal' and env['arch'] == 'armv7a':
        print("WARNING: Building tests for bare metal and armv7a is not supported")
        Return()
    SConscript('./tests/SConscript', variant_dir='%s/tests' % build_path, duplicate=0)

# Unknown variables are not allowed
# Note: we must delay the call of UnknownVariables until after
# we have applied the Variables object to the construction environment
unknown = vars.UnknownVariables()
if unknown:
    print("Unknown variables: %s" % " ".join(unknown.keys()))
    Exit(1)
