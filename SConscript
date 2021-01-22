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
import collections
import os.path
import re
import subprocess
import zlib
import base64
import string
import json

VERSION = "v0.0-unreleased"
LIBRARY_VERSION_MAJOR = 23
LIBRARY_VERSION_MINOR =  0
LIBRARY_VERSION_PATCH =  0
SONAME_VERSION = str(LIBRARY_VERSION_MAJOR) + "." + str(LIBRARY_VERSION_MINOR) + "." + str(LIBRARY_VERSION_PATCH)

Import('env')
Import('vars')
Import('install_lib')

def build_bootcode_objs(sources):
    arm_compute_env.Append(ASFLAGS = "-I bootcode/")
    obj = arm_compute_env.Object(sources)
    obj = install_lib(obj)
    Default(obj)
    return obj

def build_sve_objs(sources):
    tmp_env = arm_compute_env.Clone()
    tmp_env.Append(CXXFLAGS = "-march=armv8.2-a+sve+fp16")
    obj = tmp_env.SharedObject(sources)
    obj = install_lib(obj)
    Default(obj)
    return obj

def build_library(name, build_env, sources, static=False, libs=[]):
    if static:
        obj = build_env.StaticLibrary(name, source=sources, LIBS = arm_compute_env["LIBS"] + libs)
    else:
        if env['set_soname']:
            obj = build_env.SharedLibrary(name, source=sources, SHLIBVERSION = SONAME_VERSION, LIBS = arm_compute_env["LIBS"] + libs)
        else:
            obj = build_env.SharedLibrary(name, source=sources, LIBS = arm_compute_env["LIBS"] + libs)

    obj = install_lib(obj)
    Default(obj)
    return obj

def remove_incode_comments(code):
    def replace_with_empty(match):
        s = match.group(0)
        if s.startswith('/'):
            return " "
        else:
            return s

    comment_regex = re.compile(r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"', re.DOTALL | re.MULTILINE)
    return re.sub(comment_regex, replace_with_empty, code)

def resolve_includes(target, source, env):
    # File collection
    FileEntry = collections.namedtuple('FileEntry', 'target_name file_contents')

    # Include pattern
    pattern = re.compile("#include \"(.*)\"")

    # Get file contents
    files = []
    for i in range(len(source)):
        src = source[i]
        dst = target[i]
        contents = src.get_contents().decode('utf-8')
        contents = remove_incode_comments(contents).splitlines()
        entry = FileEntry(target_name=dst, file_contents=contents)
        files.append((os.path.basename(src.get_path()),entry))

    # Create dictionary of tupled list
    files_dict = dict(files)

    # Check for includes (can only be files in the same folder)
    final_files = []
    for file in files:
        done = False
        tmp_file = file[1].file_contents
        while not done:
            file_count = 0
            updated_file = []
            for line in tmp_file:
                found = pattern.search(line)
                if found:
                    include_file = found.group(1)
                    data = files_dict[include_file].file_contents
                    updated_file.extend(data)
                else:
                    updated_file.append(line)
                    file_count += 1

            # Check if all include are replaced.
            if file_count == len(tmp_file):
                done = True

            # Update temp file
            tmp_file = updated_file

        # Append and prepend string literal identifiers and add expanded file to final list
        entry = FileEntry(target_name=file[1].target_name, file_contents=tmp_file)
        final_files.append((file[0], entry))

    # Write output files
    for file in final_files:
        with open(file[1].target_name.get_path(), 'w+') as out_file:
            file_to_write = "\n".join( file[1].file_contents )
            if env['compress_kernels']:
                file_to_write = zlib.compress(file_to_write, 9).encode("base64").replace("\n", "")
            file_to_write = "R\"(" + file_to_write + ")\""
            out_file.write(file_to_write)

def create_version_file(target, source, env):
# Generate string with build options library version to embed in the library:
    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"])
    except (OSError, subprocess.CalledProcessError):
        git_hash="unknown"

    build_info = "\"arm_compute_version=%s Build options: %s Git hash=%s\"" % (VERSION, vars.args, git_hash.strip())
    with open(target[0].get_path(), "w") as fd:
        fd.write(build_info)

arm_compute_env = env.Clone()
version_file = arm_compute_env.Command("src/core/arm_compute_version.embed", "", action=create_version_file)
arm_compute_env.AlwaysBuild(version_file)

default_cpp_compiler = 'g++' if env['os'] not in ['android', 'macos'] else 'clang++'
cpp_compiler = os.environ.get('CXX', default_cpp_compiler)

# Generate embed files
generate_embed = [ version_file ]
if env['opencl'] and env['embed_kernels']:
    cl_files = Glob('src/core/CL/cl_kernels/*.cl')
    cl_files += Glob('src/core/CL/cl_kernels/*.h')

    embed_files = [ f.get_path()+"embed" for f in cl_files ]
    arm_compute_env.Append(CPPPATH =[Dir("./src/core/CL/").path] )

    generate_embed.append(arm_compute_env.Command(embed_files, cl_files, action=resolve_includes))

Default(generate_embed)
if env["build"] == "embed_only":
    Return()

# Append version defines for semantic versioning
arm_compute_env.Append(CPPDEFINES = [('ARM_COMPUTE_VERSION_MAJOR', LIBRARY_VERSION_MAJOR),
                                     ('ARM_COMPUTE_VERSION_MINOR', LIBRARY_VERSION_MINOR),
                                     ('ARM_COMPUTE_VERSION_PATCH', LIBRARY_VERSION_PATCH)])

# Don't allow undefined references in the libraries:
undefined_flag = '-Wl,-undefined,error' if 'macos' in arm_compute_env["os"] else '-Wl,--no-undefined'
arm_compute_env.Append(LINKFLAGS=[undefined_flag])
arm_compute_env.Append(CPPPATH =[Dir("./src/core/").path] )

arm_compute_env.Append(LIBS = ['dl'])

with (open(Dir('#').path + '/filelist.json')) as fp:
    filelist = json.load(fp)

core_files = Glob('src/core/*.cpp')
core_files += Glob('src/core/CPP/*.cpp')
core_files += Glob('src/core/CPP/kernels/*.cpp')
core_files += Glob('src/core/helpers/*.cpp')
core_files += Glob('src/core/utils/*.cpp')
core_files += Glob('src/core/utils/helpers/*.cpp')
core_files += Glob('src/core/utils/io/*.cpp')
core_files += Glob('src/core/utils/quantization/*.cpp')
core_files += Glob('src/core/utils/misc/*.cpp')
if env["logging"]:
    core_files += Glob('src/core/utils/logging/*.cpp')

runtime_files = Glob('src/runtime/*.cpp')
runtime_files += Glob('src/runtime/CPP/ICPPSimpleFunction.cpp')
runtime_files += Glob('src/runtime/CPP/functions/*.cpp')

# C API files
runtime_files += filelist['c_api']['cpu']

if env['opencl']:
    runtime_files += filelist['c_api']['gpu']

# Common backend files
core_files += filelist['common']

runtime_files += Glob('src/runtime/CPP/SingleThreadScheduler.cpp')

graph_files = Glob('src/graph/*.cpp')
graph_files += Glob('src/graph/*/*.cpp')

if env['cppthreads']:
     runtime_files += Glob('src/runtime/CPP/CPPScheduler.cpp')

if env['openmp']:
     runtime_files += Glob('src/runtime/OMP/OMPScheduler.cpp')

if env['opencl']:
    cl_kernel_hp_files = ['src/core/gpu/cl/kernels/gemm/ClGemmHelpers.cpp',
                          'src/core/gpu/cl/kernels/gemm/native/ClGemmDefaultConfigNativeBifrost.cpp',
                          'src/core/gpu/cl/kernels/gemm/native/ClGemmDefaultConfigNativeMidgard.cpp',
                          'src/core/gpu/cl/kernels/gemm/native/ClGemmDefaultConfigNativeValhall.cpp',
                          'src/core/gpu/cl/kernels/gemm/reshaped/ClGemmDefaultConfigReshapedBifrost.cpp',
                          'src/core/gpu/cl/kernels/gemm/reshaped/ClGemmDefaultConfigReshapedValhall.cpp',
                          'src/core/gpu/cl/kernels/gemm/reshaped_only_rhs/ClGemmDefaultConfigReshapedRhsOnlyBifrost.cpp',
                          'src/core/gpu/cl/kernels/gemm/reshaped_only_rhs/ClGemmDefaultConfigReshapedRhsOnlyValhall.cpp',
                          ]
    core_files += cl_kernel_hp_files
    core_files += Glob('src/core/CL/*.cpp')
    core_files += Glob('src/core/gpu/cl/*.cpp')

    runtime_files += Glob('src/runtime/CL/*.cpp')
    runtime_files += Glob('src/runtime/CL/functions/*.cpp')
    runtime_files += Glob('src/runtime/CL/gemm/*.cpp')
    runtime_files += Glob('src/runtime/CL/tuners/*.cpp')
    runtime_files += Glob('src/runtime/gpu/cl/*.cpp')
    runtime_files += Glob('src/runtime/gpu/cl/operators/*.cpp')
    runtime_files += Glob('src/runtime/CL/mlgo/*.cpp')
    runtime_files += Glob('src/runtime/CL/gemm_auto_heuristics/*.cpp')

    runtime_files += Glob('src/gpu/cl/*.cpp')
    graph_files += Glob('src/graph/backends/CL/*.cpp')

    core_files += filelist['gpu']['core']['kernels']['high_priority'] + filelist['gpu']['core']['kernels']['all']

sve_o = []
core_files_sve = []
if env['neon']:
    core_files += Glob('src/core/NEON/*.cpp')
    core_files += Glob('src/core/NEON/kernels/*.cpp')

    core_files += Glob('src/core/NEON/kernels/arm_gemm/*.cpp')

    # build winograd/depthwise sources for either v7a / v8a
    core_files += Glob('src/core/NEON/kernels/convolution/*/*.cpp')
    core_files += Glob('src/core/NEON/kernels/convolution/winograd/*/*.cpp')
    arm_compute_env.Append(CPPPATH = ["src/core/NEON/kernels/convolution/common/",
                                      "src/core/NEON/kernels/convolution/winograd/",
                                      "src/core/NEON/kernels/convolution/depthwise/",
                                      "src/core/NEON/kernels/assembly/",
                                      "arm_compute/core/NEON/kernels/assembly/",
                                      "src/core/cpu/kernels/assembly/",])

    graph_files += Glob('src/graph/backends/NEON/*.cpp')

    if env['estate'] == '32':
        core_files += Glob('src/core/NEON/kernels/arm_gemm/kernels/a32_*/*.cpp')

    if env['estate'] == '64':
        core_files += Glob('src/core/NEON/kernels/assembly/*.cpp')
        core_files += Glob('src/core/NEON/kernels/arm_conv/depthwise/*.cpp')
        core_files += Glob('src/core/NEON/kernels/arm_conv/depthwise/kernels/cpp_*/*.cpp')
        core_files += Glob('src/core/NEON/kernels/arm_conv/depthwise/interleaves/8b_mla.cpp')
        core_files += Glob('src/core/NEON/kernels/arm_conv/pooling/*.cpp')
        core_files += Glob('src/core/NEON/kernels/arm_conv/pooling/kernels/cpp_*/*.cpp')

        core_files += Glob('src/core/NEON/kernels/arm_gemm/kernels/a64_*/*.cpp')
        core_files += Glob('src/core/NEON/kernels/arm_conv/depthwise/interleaves/a64_*.cpp')
        core_files += Glob('src/core/NEON/kernels/arm_conv/depthwise/kernels/a64_*/*.cpp')
        core_files += Glob('src/core/NEON/kernels/arm_conv/pooling/kernels/a64_*/*.cpp')
        if "sve" in env['arch'] or env['fat_binary']:
            core_files_sve += filelist['cpu']['core']['sve']['all']
            core_files_sve += Glob('src/core/NEON/kernels/arm_gemm/kernels/sve_*/*.cpp')
            core_files += Glob('src/core/NEON/kernels/arm_conv/depthwise/interleaves/sve_*.cpp')
            core_files += Glob('src/core/NEON/kernels/arm_conv/depthwise/kernels/sve_*/*.cpp')
            core_files_sve += Glob('src/core/NEON/kernels/arm_conv/pooling/kernels/sve_*/*.cpp')

    if any(i in env['data_layout_support'] for i in ['all', 'nchw']):
        core_files += filelist['cpu']['core']['neon']['nchw']

    if any(i in env['data_type_support'] for i in ['all', 'fp16']):
        if not "sve" in env['arch'] or env['fat_binary']:
            core_files += filelist['cpu']['core']['neon']['fp16']
        if "sve" in env['arch'] or env['fat_binary']:
            core_files_sve += filelist['cpu']['core']['sve']['fp16']
    if any(i in env['data_type_support'] for i in ['all', 'fp32']):
        if not "sve" in env['arch'] or env['fat_binary']:
            core_files += filelist['cpu']['core']['neon']['fp32']
        if "sve" in env['arch'] or env['fat_binary']:
            core_files_sve += filelist['cpu']['core']['sve']['fp32']
    if any(i in env['data_type_support'] for i in ['all', 'qasymm8']):
        core_files += filelist['cpu']['core']['neon']['qasymm8']
        core_files_sve += filelist['cpu']['core']['sve']['qasymm8']
    if any(i in env['data_type_support'] for i in ['all', 'qasymm8_signed']):
        core_files += filelist['cpu']['core']['neon']['qasymm8_signed']
        core_files_sve += filelist['cpu']['core']['sve']['qasymm8_signed']
    if any(i in env['data_type_support'] for i in ['all', 'qsymm16']):
        core_files += filelist['cpu']['core']['neon']['qsymm16']
        core_files_sve += filelist['cpu']['core']['sve']['qsymm16']
    if any(i in env['data_type_support'] for i in ['all', 'integer']):
        if not "sve" in env['arch'] or env['fat_binary']:
            core_files += filelist['cpu']['core']['neon']['integer']
        if "sve" in env['arch'] or env['fat_binary']:
            core_files_sve += filelist['cpu']['core']['sve']['integer']

    core_files += Glob('src/core/cpu/kernels/*/*.cpp')
    core_files += filelist['cpu']['core']['kernels']['high_priority'] + filelist['cpu']['core']['kernels']['all']

    runtime_files += Glob('src/runtime/NEON/*.cpp')
    runtime_files += Glob('src/runtime/NEON/functions/*.cpp')
    runtime_files += Glob('src/runtime/NEON/functions/assembly/*.cpp')
    runtime_files += filelist['cpu']['runtime']['all'] + filelist['cpu']['runtime']['operators']['high_priority'] \
                  + filelist['cpu']['runtime']['operators']['all'] + filelist['cpu']['runtime']['operators']['internal']

bootcode_o = []
if env['os'] == 'bare_metal':
    bootcode_files = Glob('bootcode/*.s')
    bootcode_o = build_bootcode_objs(bootcode_files)
Export('bootcode_o')

if (env['fat_binary']):
    sve_o = build_sve_objs(core_files_sve)
    arm_compute_core_a = build_library('arm_compute_core-static', arm_compute_env, core_files + sve_o, static=True)
else:
    arm_compute_core_a = build_library('arm_compute_core-static', arm_compute_env, core_files + core_files_sve, static=True)
Export('arm_compute_core_a')

if env['os'] != 'bare_metal' and not env['standalone']:
    if (env['fat_binary']):
        arm_compute_core_so = build_library('arm_compute_core', arm_compute_env, core_files + sve_o, static=False)
    else:
        arm_compute_core_so = build_library('arm_compute_core', arm_compute_env, core_files + core_files_sve, static=False)
    Export('arm_compute_core_so')

arm_compute_a = build_library('arm_compute-static', arm_compute_env, runtime_files, static=True, libs = [ arm_compute_core_a ])
Export('arm_compute_a')

if env['os'] != 'bare_metal' and not env['standalone']:
    arm_compute_so = build_library('arm_compute', arm_compute_env, runtime_files, static=False, libs = [ "arm_compute_core" ])
    Depends(arm_compute_so, arm_compute_core_so)
    Export('arm_compute_so')

arm_compute_graph_env = arm_compute_env.Clone()

arm_compute_graph_env.Append(CXXFLAGS = ['-Wno-redundant-move', '-Wno-pessimizing-move'])

arm_compute_graph_a = build_library('arm_compute_graph-static', arm_compute_graph_env, graph_files, static=True, libs = [ arm_compute_a])
Export('arm_compute_graph_a')

if env['os'] != 'bare_metal' and not env['standalone']:
    arm_compute_graph_so = build_library('arm_compute_graph', arm_compute_graph_env, graph_files, static=False, libs = [ "arm_compute" , "arm_compute_core"])
    Depends(arm_compute_graph_so, arm_compute_so)
    Export('arm_compute_graph_so')

if env['standalone']:
    alias = arm_compute_env.Alias("arm_compute", [arm_compute_a])
else:
    alias = arm_compute_env.Alias("arm_compute", [arm_compute_a, arm_compute_so])

Default(alias)

if env['standalone']:
    Depends([alias,arm_compute_core_a], generate_embed)
else:
    Depends([alias,arm_compute_core_so, arm_compute_core_a], generate_embed)
