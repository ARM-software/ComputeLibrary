# -*- coding: utf-8 -*-

# Copyright (c) 2016-2023 Arm Limited.
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
    EnumVariable("arch", "Target Architecture. The x86_32 and x86_64 targets can only be used with neon=0 and opencl=1.", "armv7a",
                  allowed_values=("armv7a", "armv7a-hf", "arm64-v8a", "arm64-v8.2-a", "arm64-v8.2-a-sve", "arm64-v8.2-a-sve2", "x86_32", "x86_64",
                                  "armv8a", "armv8.2-a", "armv8.2-a-sve", "armv8.6-a", "armv8.6-a-sve", "armv8.6-a-sve2", "armv8.6-a-sve2-sme2", "armv8r64", "x86")),
    EnumVariable("estate", "Execution State", "auto", allowed_values=("auto", "32", "64")),
    EnumVariable("os", "Target OS. With bare metal selected, only Arm® Neon™ (not OpenCL) can be used, static libraries get built and Neon™'s multi-threading support is disabled.", "linux", allowed_values=("linux", "android", "tizen", "macos", "bare_metal", "openbsd","windows")),
    EnumVariable("build", "Either build directly on your device (native) or cross compile from your desktop machine (cross-compile). In both cases make sure the compiler is available in your path.", "cross_compile", allowed_values=("native", "cross_compile", "embed_only")),
    BoolVariable("examples", "Build example programs", True),
    BoolVariable("gemm_tuner", "Build gemm_tuner programs", True),
    BoolVariable("Werror", "Enable/disable the -Werror compilation flag", True),
    BoolVariable("multi_isa", "Build Multi ISA binary version of library. Works for armv8a without the support for FP16 vector arithmetic. Use armv8.2-a or beyond to enable FP16 vector arithmetic support", False),
    BoolVariable("standalone", "Builds the tests as standalone executables, links statically with libgcc, libstdc++ and libarm_compute", False),
    BoolVariable("opencl", "Enable OpenCL support", True),
    BoolVariable("neon", "Enable Arm® Neon™ support", False),
    BoolVariable("embed_kernels", "Enable if you want the OpenCL kernels to be built in the library's binaries instead of being read from separate '.cl' / '.cs' files. If embed_kernels is set to 0 then the application can set the path to the folder containing the OpenCL kernel files by calling CLKernelLibrary::init(). By default the path is set to './cl_kernels'.", True),
    BoolVariable("compress_kernels", "Compress embedded OpenCL kernels in library binary using zlib. Useful for reducing the binary size. embed_kernels should be enabled", False),
    BoolVariable("set_soname", "If enabled the library will contain a SONAME and SHLIBVERSION and some symlinks will automatically be created between the objects. (requires SCons 2.4 or above)", False),
    BoolVariable("openmp", "Enable OpenMP backend. Only works when building with g++ and not clang++", False),
    BoolVariable("cppthreads", "Enable C++11 threads backend", True),
    PathVariable("build_dir", "Specify sub-folder for the build", ".", PathVariable.PathAccept),
    PathVariable("install_dir", "Specify sub-folder for the install", "", PathVariable.PathAccept),
    BoolVariable("exceptions", "Enable/disable C++ exception support", True),
    BoolVariable("high_priority", "Generate a library containing only the high priority operators", False),
    PathVariable("linker_script", "Use an external linker script", "", PathVariable.PathAccept),
    PathVariable("external_tests_dir", """Add examples, benchmarks and tests to the tests suite from an external path. In order to use this option, the external tests directory must have the following structure: 
    EXTERNAL_TESTS_DIR:
    └── tests
        ├── benchmark
        │   ├── CL
        │   ├── datasets
        │   ├── fixtures
        │   └── Neon
        └── validation
            ├── CL
            ├── datasets
            ├── fixtures
            └── Neon\n""", "", PathVariable.PathAccept),
    BoolVariable("experimental_dynamic_fusion", "Build the experimental dynamic fusion files. This option also enables opencl=1 on which it has a direct dependency.", False),
    BoolVariable("fixed_format_kernels", "Enable fixed format kernels for GEMM", False),
    BoolVariable("mapfile", "Generate a map file", False),
    ListVariable("custom_options", "Custom options that can be used to turn on/off features", "none", ["disable_mmla_fp"]),
    ListVariable("data_type_support", "Enable a list of data types to support", "all", ["qasymm8", "qasymm8_signed", "qsymm16", "fp16", "fp32", "integer"]),
    ListVariable("data_layout_support", "Enable a list of data layout to support", "all", ["nhwc", "nchw"]),
    ("toolchain_prefix", "Override the toolchain prefix; used by all toolchain components: compilers, linker, assembler etc. If unspecified, use default(auto) prefixes; if passed an empty string '' prefixes would be disabled", "auto"),
    ("compiler_prefix", "Override the compiler prefix; used by just compilers (CC,CXX); further overrides toolchain_prefix for compilers; this is for when the compiler prefixes are different from that of the linkers, archivers etc. If unspecified, this is the same as toolchain_prefix; if passed an empty string '' prefixes would be disabled", "auto"),
    ("extra_cxx_flags", "Extra CXX flags to be appended to the build command", ""),
    ("extra_link_flags", "Extra LD flags to be appended to the build command", ""),
    ("compiler_cache", "Command to prefix to the C and C++ compiler (e.g ccache)", ""),
    ("specs_file", "Specs file to use (e.g. rdimon.specs)", ""),
    ("build_config", "Operator/Data-type/Data-layout configuration to use for tailored ComputeLibrary builds. Can be a JSON file or a JSON formatted string", "")
)

if version_at_least(SCons.__version__, "4.0"):
    vars.Add(BoolVariable("export_compile_commands", "Export compile_commands.json file.", False))


env = Environment(variables=vars, ENV = os.environ)


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

# Export compile_commands.json file
if env.get("export_compile_commands", False):
    env.Tool("compilation_db")
    env.CompilationDatabase("%s/compile_commands.json" % build_path)

if 'armv7a' in env['arch'] and env['os'] == 'android':
    print("WARNING: armv7a on Android is no longer maintained")

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

if env['experimental_dynamic_fusion']:
    # Dynamic Fusion on GPU has a direct dependency on OpenCL and Compute Kernel Writer
    env['opencl'] = 1

if env['opencl'] and env['embed_kernels'] and env['compress_kernels'] and env['os'] not in ['android']:
    print("Compressed kernels are supported only for android builds")
    Exit(1)

if not env['exceptions']:
    if env['opencl']:
         print("ERROR: OpenCL is not supported when building without exceptions. Use opencl=0")
         Exit(1)

    env.Append(CPPDEFINES = ['ARM_COMPUTE_EXCEPTIONS_DISABLED'])
    env.Append(CXXFLAGS = ['-fno-exceptions'])

env.Append(CXXFLAGS = ['-DARCH_ARM',
         '-Wextra','-Wdisabled-optimization','-Wformat=2',
         '-Winit-self','-Wstrict-overflow=2','-Wswitch-default',
         '-Woverloaded-virtual', '-Wformat-security',
         '-Wctor-dtor-privacy','-Wsign-promo','-Weffc++','-Wno-overlength-strings'])

if not 'windows' in env['os']:
    env.Append(CXXFLAGS = ['-Wall','-std=c++14', '-pedantic' ])

env.Append(CPPDEFINES = ['_GLIBCXX_USE_NANOSLEEP'])

cpp_tool = {'linux': 'g++', 'android' : 'clang++',
             'tizen': 'g++', 'macos':'clang++',
             'bare_metal':'g++', 'openbsd':'g++','windows':'clang-cl'}

c_tool = {'linux':'gcc', 'android': 'clang', 'tizen':'gcc',
          'macos':'clang','bare_metal':'gcc',
          'openbsd':'gcc','windows':'clang-cl'}

default_cpp_compiler = cpp_tool[env['os']]
default_c_compiler = c_tool[env['os']]
cpp_compiler = os.environ.get('CXX', default_cpp_compiler)
c_compiler = os.environ.get('CC', default_c_compiler)

if env['os'] == 'android' and ( 'clang++' not in cpp_compiler or 'clang' not in c_compiler ):
    print( "WARNING: Only clang is officially supported to build the Compute Library for Android")

if 'clang++' in cpp_compiler:
    env.Append(CXXFLAGS = ['-Wno-vla-extension'])
elif 'armclang' in cpp_compiler:
    pass
elif not 'windows' in env['os']:
        env.Append(CXXFLAGS = ['-Wlogical-op','-Wnoexcept','-Wstrict-null-sentinel','-Wno-misleading-indentation'])

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

env.Append(CPPDEFINES = ['ENABLE_NEON', 'ARM_COMPUTE_ENABLE_NEON'])

if 'sve' in env['arch']:
    env.Append(CPPDEFINES = ['ENABLE_SVE', 'ARM_COMPUTE_ENABLE_SVE'])
    if 'sve2' in env['arch']:
        env.Append(CPPDEFINES = ['ARM_COMPUTE_ENABLE_SVE2'])

if 'sme' in env['arch']:
    env.Append(CPPDEFINES = ['ENABLE_SME', 'ARM_COMPUTE_ENABLE_SME'])
    if 'sme2' in env['arch']:
       env.Append(CPPDEFINES = ['ARM_COMPUTE_ENABLE_SME2'])

# Add architecture specific flags
if env['multi_isa']:
    # assert arch version is v8
    if 'v8' not in env['arch']:
        print("Currently Multi ISA binary is only supported for arm v8 family")
        Exit(1)

    if 'v8a' in env['arch']:
        print("INFO: multi_isa armv8-a architecture build doesn't enable __ARM_FEATURE_FP16_VECTOR_ARITHMETIC. Use armv8.2-a or beyond to enable FP16 vector arithmetic support")
        env.Append(CXXFLAGS = ['-march=armv8-a']) # note: this will disable fp16 extension __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    else:
        if 'v8.6-a' in env['arch']:
            if "disable_mmla_fp" not in env['custom_options']:
                env.Append(CPPDEFINES = ['ARM_COMPUTE_ENABLE_SVEF32MM'])

        env.Append(CXXFLAGS = ['-march=armv8.2-a+fp16']) # explicitly enable fp16 extension otherwise __ARM_FEATURE_FP16_VECTOR_ARITHMETIC is undefined

else: # NONE "multi_isa" builds

    if 'v7a' in env['arch']:
        env.Append(CXXFLAGS = ['-march=armv7-a', '-mthumb', '-mfpu=neon'])
        if (env['os'] == 'android' or env['os'] == 'tizen') and not 'hf' in env['arch']:
            env.Append(CXXFLAGS = ['-mfloat-abi=softfp'])
        else:
            env.Append(CXXFLAGS = ['-mfloat-abi=hard'])
    elif 'v8.6-a' in env['arch']:
        if 'armv8.6-a-sve2' in env['arch']:
            env.Append(CXXFLAGS = ['-march=armv8.6-a+sve2'])
        elif 'armv8.6-a-sve' == env['arch']:
            env.Append(CXXFLAGS = ['-march=armv8.6-a+sve'])
        elif 'armv8.6-a' == env['arch']:
            env.Append(CXXFLAGS = ['-march=armv8.6-a+fp16'])

        env.Append(CPPDEFINES = ['ARM_COMPUTE_ENABLE_I8MM', 'ARM_COMPUTE_ENABLE_BF16','ARM_COMPUTE_ENABLE_FP16'])
        if "disable_mmla_fp" not in env['custom_options']:
            env.Append(CPPDEFINES = ['ARM_COMPUTE_ENABLE_SVEF32MM'])
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
# The reason why we distinguish toolchain_prefix from compiler_prefix is for cases where the linkers/archivers use a
# different prefix than the compilers. An example is the NDK r20 toolchain
auto_toolchain_prefix = ""
if 'x86' not in env['arch']:
    if env['estate'] == '32':
        if env['os'] == 'linux':
            auto_toolchain_prefix = "arm-linux-gnueabihf-" if 'v7' in env['arch'] else "armv8l-linux-gnueabihf-"
        elif env['os'] == 'bare_metal':
            auto_toolchain_prefix = "arm-eabi-"
        elif env['os'] == 'android':
            auto_toolchain_prefix = "arm-linux-androideabi-"
        elif env['os'] == 'tizen':
            auto_toolchain_prefix = "armv7l-tizen-linux-gnueabi-"
    elif env['estate'] == '64' and 'v8' in env['arch']:
        if env['os'] == 'linux':
            auto_toolchain_prefix = "aarch64-linux-gnu-"
        elif env['os'] == 'bare_metal':
            auto_toolchain_prefix = "aarch64-elf-"
        elif env['os'] == 'android':
            auto_toolchain_prefix = "aarch64-linux-android-"
        elif env['os'] == 'tizen':
            auto_toolchain_prefix = "aarch64-tizen-linux-gnu-"

if env['build'] == 'native' or env["toolchain_prefix"] == "":
    toolchain_prefix = ""
elif env["toolchain_prefix"] == "auto":
    toolchain_prefix = auto_toolchain_prefix
else:
    toolchain_prefix = env["toolchain_prefix"]

if env['build'] == 'native' or env["compiler_prefix"] == "":
    compiler_prefix = ""
elif env["compiler_prefix"] == "auto":
    compiler_prefix = toolchain_prefix
else:
    compiler_prefix = env["compiler_prefix"]

env['CC'] = env['compiler_cache']+ " " + compiler_prefix + c_compiler
env['CXX'] = env['compiler_cache']+ " " + compiler_prefix + cpp_compiler
env['LD'] = toolchain_prefix + "ld"
env['AS'] = toolchain_prefix + "as"

if env['os'] == 'windows':
    env['AR'] = "llvm-lib"
    env['RANLIB'] = "llvm-ranlib"
else:
    env['AR'] = toolchain_prefix + "ar"

env['RANLIB'] = toolchain_prefix + "ranlib"

print("Using compilers:")
print("CC", env['CC'])
print("CXX", env['CXX'])

"""Build the Compute Kernel Writer subproject"""
if env['experimental_dynamic_fusion']:
    # Strip ccache prefix from CC and CXX to obtain only the target triple
    CKW_CC = env['CC'].replace(env['compiler_cache'] + " ", "")
    CKW_CXX = env['CXX'].replace(env['compiler_cache'] + " ", "")
    CKW_CCACHE = 1 if env['compiler_cache'] else 0

    CKW_BUILD_TYPE = "Debug" if env['debug'] else "Release"

    CKW_ENABLE_OPENCL = env['opencl']
    CKW_ENABLE_ASSERTS = env['debug'] or env['asserts']

    CKW_PROJECT_DIR = Dir('.').path + "/compute_kernel_writer"
    CKW_INCLUDE_DIR = CKW_PROJECT_DIR + "/include"
    CKW_BUILD_DIR = build_path.replace("#", "")

    CKW_CMAKE_CMD = "CC={CKW_CC} CXX={CKW_CXX} cmake -G \"Unix Makefiles\" " \
                    "-S {CKW_PROJECT_DIR} -B {CKW_BUILD_DIR} " \
                    "-DCMAKE_BUILD_TYPE={CKW_BUILD_TYPE} " \
                    "-DCKW_ENABLE_OPENCL={CKW_ENABLE_OPENCL} " \
                    "-DCKW_ENABLE_ASSERTS={CKW_ENABLE_ASSERTS} " \
                    "-DCKW_CCACHE={CKW_CCACHE} ".format(CKW_CC=CKW_CC,
                                                        CKW_CXX=CKW_CXX,
                                                        CKW_PROJECT_DIR=CKW_PROJECT_DIR,
                                                        CKW_BUILD_DIR=CKW_BUILD_DIR,
                                                        CKW_BUILD_TYPE=CKW_BUILD_TYPE,
                                                        CKW_ENABLE_OPENCL=CKW_ENABLE_OPENCL,
                                                        CKW_ENABLE_ASSERTS=CKW_ENABLE_ASSERTS,
                                                        CKW_CCACHE=CKW_CCACHE
                                                        )

    # Configure CKW static objects with -fPIC (CMAKE_POSITION_INDEPENDENT_CODE) option to enable linking statically to ACL
    CKW_CMAKE_CONFIGURE_STATIC = CKW_CMAKE_CMD + "-DBUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON"
    CKW_CMAKE_BUILD = "cmake --build {CKW_BUILD_DIR} -j{NUM_JOBS}".format(CKW_BUILD_DIR=CKW_BUILD_DIR,
                                                                          NUM_JOBS=GetOption('num_jobs')
                                                                          )

    # Build Compute Kernel Writer Static Library
    subprocess.check_call(CKW_CMAKE_CONFIGURE_STATIC, stderr=subprocess.STDOUT, shell=True)
    subprocess.check_call(CKW_CMAKE_BUILD, stderr=subprocess.STDOUT, shell=True)

    # Let ACL know where to find CKW headers
    env.Append(CPPPATH = CKW_INCLUDE_DIR)

if not GetOption("help"):
    try:
        if env['os'] == 'windows':
            compiler_ver = subprocess.check_output("clang++ -dumpversion").decode().strip()
        else:
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

    # Add Android NDK toolchain specific flags
    if 'clang++' in cpp_compiler and env['os'] == 'android':
        # For NDK >= r21, clang 9 or above is used
        if version_at_least(compiler_ver, '9.0.0'):
            env['ndk_above_r21'] = True

            if env['openmp']:
                env.Append(LINKFLAGS = ['-static-openmp'])

        # For NDK >= r23, clang 12 or above is used. This condition detects NDK < r23
        if not version_at_least(compiler_ver, '12.0.0'):
            # System assembler is deprecated and integrated assembler is preferred since r23.
            # However integrated assembler has always been suppressed for NDK < r23.
            # Thus for backward compatibility, we include this flag only for NDK < r23
            env.Append(CXXFLAGS = ['-no-integrated-as'])

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
    if not 'windows' in env['os']:
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
if env['os'] == 'windows':
    env.Append(CXXFLAGS = [ '/std:c++14','/EHa'])
    env.Append(CXXFLAGS = [ '-Wno-c++98-compat', '-Wno-covered-switch-default','-Wno-c++98-compat-pedantic'])
    env.Append(CXXFLAGS = [ '-Wno-shorten-64-to-32', '-Wno-sign-conversion','-Wno-documentation'])
    env.Append(CXXFLAGS = [ '-Wno-extra-semi-stmt', '-Wno-float-equal','-Wno-implicit-int-conversion'])
    env.Append(CXXFLAGS = [ '-Wno-documentation-pedantic', '-Wno-extra-semi','-Wno-shadow-field-in-constructor'])
    env.Append(CXXFLAGS = [ '-Wno-float-conversion', '-Wno-switch-enum','-Wno-comma'])
    env.Append(CXXFLAGS = [ '-Wno-implicit-float-conversion', '-Wno-deprecated-declarations','-Wno-old-style-cast'])
    env.Append(CXXFLAGS = [ '-Wno-zero-as-null-pointer-constant', '-Wno-inconsistent-missing-destructor-override'])
    env.Append(CXXFLAGS = [ '-Wno-asm-operand-widths'])


if env['specs_file'] != "":
    env.Append(LINKFLAGS = ['-specs='+env['specs_file']])

if env['neon']:
    env.Append(CPPDEFINES = ['ARM_COMPUTE_CPU_ENABLED'])

if env['opencl']:
    env.Append(CPPDEFINES = ['ARM_COMPUTE_OPENCL_ENABLED'])
    if env['os'] in ['bare_metal'] or env['standalone']:
        print("Cannot link OpenCL statically, which is required for bare metal / standalone builds")
        Exit(1)

if env["os"] not in ["windows","android", "bare_metal"] and (env['opencl'] or env['cppthreads']):
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
    if not 'windows' in env['os']:
        env.Append(CXXFLAGS = ['-O0','-g','-gdwarf-2'])
    else:
        env.Append(CXXFLAGS = ['-Z7','-MTd','-fms-compatibility','-fdelayed-template-parsing'])
        env.Append(LINKFLAGS = ['-DEBUG'])
 
    env.Append(CPPDEFINES = ['ARM_COMPUTE_DEBUG_ENABLED'])
else:
    if not 'windows' in env['os']:
        env.Append(CXXFLAGS = ['-O3'])
    else:
        # on windows we use clang-cl which does not support the option -O3
        env.Append(CXXFLAGS = ['-O2'])

if env['asserts']:
    env.Append(CPPDEFINES = ['ARM_COMPUTE_ASSERTS_ENABLED'])
    if not 'windows' in env['os']:
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
