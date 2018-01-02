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
import collections
import os.path
import re
import subprocess

VERSION = "v17.12"
SONAME_VERSION="6.0.0"

Import('env')
Import('vars')

def build_library(name, sources, static=False, libs=[]):
    if static:
        obj = arm_compute_env.StaticLibrary(name, source=sources, LIBS = arm_compute_env["LIBS"] + libs)
    else:
        if env['set_soname']:
            obj = arm_compute_env.SharedLibrary(name, source=sources, SHLIBVERSION = SONAME_VERSION, LIBS = arm_compute_env["LIBS"] + libs)

            symlinks = []
            # Manually delete symlinks or SCons will get confused:
            directory = os.path.dirname(obj[0].path)
            library_prefix = obj[0].path[:-(1 + len(SONAME_VERSION))]
            real_lib = "%s.%s" % (library_prefix, SONAME_VERSION)

            for f in Glob("#%s*" % library_prefix):
                if str(f) != real_lib:
                    symlinks.append("%s/%s" % (directory,str(f)))

            clean = arm_compute_env.Command('clean-%s' % str(obj[0]), [], Delete(symlinks))
            Default(clean)
            Depends(obj, clean)
        else:
            obj = arm_compute_env.SharedLibrary(name, source=sources, LIBS = arm_compute_env["LIBS"] + libs)

    Default(obj)
    return obj

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
        contents = src.get_contents().splitlines()
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
        tmp_file.insert(0, "R\"(\n")
        tmp_file.append("\n)\"")
        entry = FileEntry(target_name=file[1].target_name, file_contents=tmp_file)
        final_files.append((file[0], entry))

    # Write output files
    for file in final_files:
        with open(file[1].target_name.get_path(), 'w+') as out_file:
            out_file.write( "\n".join( file[1].file_contents ))

def create_version_file(target, source, env):
# Generate string with build options library version to embed in the library:
    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"])
    except (OSError, subprocess.CalledProcessError):
        git_hash="unknown"

    version_filename = "%s/arm_compute_version.embed" % Dir("src/core").path
    build_info = "\"arm_compute_version=%s Build options: %s Git hash=%s\"" % (VERSION, vars.args, git_hash.strip())
    with open(target[0].get_path(), "w") as fd:
        fd.write(build_info)


arm_compute_env = env.Clone()
# Don't allow undefined references in the libraries:
arm_compute_env.Append(LINKFLAGS=['-Wl,--no-undefined','-Wl,--no-allow-shlib-undefined'])

generate_embed = [ arm_compute_env.Command("src/core/arm_compute_version.embed", "", action=create_version_file) ]
arm_compute_env.Append(CPPPATH =[Dir("./src/core/").path] )

if env["os"] not in ["android", "bare_metal"]:
    arm_compute_env.Append(LIBS = ['pthread'])

arm_compute_env.Append(LIBS = ['dl'])

core_files = Glob('src/core/*.cpp')
core_files += Glob('src/core/CPP/*.cpp')
core_files += Glob('src/core/CPP/kernels/*.cpp')
core_files += Glob('src/core/utils/*/*.cpp')

runtime_files = Glob('src/runtime/*.cpp')
runtime_files += Glob('src/runtime/CPP/ICPPSimpleFunction.cpp')
runtime_files += Glob('src/runtime/CPP/functions/*.cpp')

# CLHarrisCorners uses the Scheduler to run CPP kernels
runtime_files += Glob('src/runtime/CPP/SingleThreadScheduler.cpp')

if env['cppthreads']:
     runtime_files += Glob('src/runtime/CPP/CPPScheduler.cpp')

if env['openmp']:
     runtime_files += Glob('src/runtime/OMP/OMPScheduler.cpp')

if env['opencl']:
    core_files += Glob('src/core/CL/*.cpp')
    core_files += Glob('src/core/CL/kernels/*.cpp')

    runtime_files += Glob('src/runtime/CL/*.cpp')
    runtime_files += Glob('src/runtime/CL/functions/*.cpp')

    # Generate embed files
    if env['embed_kernels']:
        cl_files = Glob('src/core/CL/cl_kernels/*.cl')
        cl_files += Glob('src/core/CL/cl_kernels/*.h')

        embed_files = [ f.get_path()+"embed" for f in cl_files ]
        arm_compute_env.Append(CPPPATH =[Dir("./src/core/CL/").path] )

        generate_embed.append(arm_compute_env.Command(embed_files, cl_files, action=resolve_includes))

if env['neon']:
    core_files += Glob('src/core/NEON/*.cpp')
    core_files += Glob('src/core/NEON/kernels/*.cpp')

    if env['arch'] == "armv7a":
        core_files += Glob('src/core/NEON/kernels/arm32/*.cpp')

    if "arm64-v8" in env['arch']:
        core_files += Glob('src/core/NEON/kernels/arm64/*.cpp')

    runtime_files += Glob('src/runtime/NEON/*.cpp')
    runtime_files += Glob('src/runtime/NEON/functions/*.cpp')

if env['gles_compute']:
    if env['os'] != 'android':
        arm_compute_env.Append(CPPPATH = ["#opengles-3.1/include", "#opengles-3.1/mali_include"])

    core_files += Glob('src/core/GLES_COMPUTE/*.cpp')
    core_files += Glob('src/core/GLES_COMPUTE/kernels/*.cpp')

    runtime_files += Glob('src/runtime/GLES_COMPUTE/*.cpp')
    runtime_files += Glob('src/runtime/GLES_COMPUTE/functions/*.cpp')

    # Generate embed files
    if env['embed_kernels']:
        cs_files = Glob('src/core/GLES_COMPUTE/cs_shaders/*.cs')
        cs_files += Glob('src/core/GLES_COMPUTE/cs_shaders/*.h')

        embed_files = [ f.get_path()+"embed" for f in cs_files ]
        arm_compute_env.Append(CPPPATH =[Dir("./src/core/GLES_COMPUTE/").path] )

        generate_embed.append(arm_compute_env.Command(embed_files, cs_files, action=resolve_includes))

arm_compute_core_a = build_library('arm_compute_core-static', core_files, static=True)
Export('arm_compute_core_a')

if env['os'] != 'bare_metal' and not env['standalone']:
    arm_compute_core_so = build_library('arm_compute_core', core_files, static=False)
    Export('arm_compute_core_so')

arm_compute_a = build_library('arm_compute-static', runtime_files, static=True, libs = [ arm_compute_core_a ])
Export('arm_compute_a')

if env['os'] != 'bare_metal' and not env['standalone']:
    arm_compute_so = build_library('arm_compute', runtime_files, static=False, libs = [ "arm_compute_core" ])
    Depends(arm_compute_so, arm_compute_core_so)
    Export('arm_compute_so')

if env['neon'] and env['opencl']:
    Import('opencl')
    graph_files = Glob('src/graph/*.cpp')
    graph_files += Glob('src/graph/nodes/*.cpp')
    graph_files += Glob('src/graph/operations/*.cpp')

    graph_files += Glob('src/graph/CL/*.cpp')
    graph_files += Glob('src/graph/NEON/*.cpp')

    shared_graph_objects = [arm_compute_env.SharedObject(f) for f in graph_files]
    static_graph_objects = [arm_compute_env.StaticObject(f) for f in graph_files]

    arm_compute_graph_a = build_library('arm_compute_graph-static', static_graph_objects, static=True, libs = [ arm_compute_a ])
    Export('arm_compute_graph_a')

    arm_compute_env.Append(LIBPATH = ["#build/%s/opencl-1.2-stubs" % env['build_dir']])
    arm_compute_graph_so = build_library('arm_compute_graph', shared_graph_objects, static=False, libs = [ "arm_compute", "arm_compute_core", "OpenCL" ])
    Depends(arm_compute_graph_so, arm_compute_so)
    Depends(arm_compute_graph_so, opencl)
    Export('arm_compute_graph_so')

    graph_alias = arm_compute_env.Alias("arm_compute_graph", [arm_compute_graph_a, arm_compute_graph_so])
    Default(graph_alias)

if env['standalone']:
    alias = arm_compute_env.Alias("arm_compute", [arm_compute_a])
else:
    alias = arm_compute_env.Alias("arm_compute", [arm_compute_a, arm_compute_so])

Default(alias)

Default(generate_embed)

if env['standalone']:
    Depends([alias,arm_compute_core_a], generate_embed)
else:
    Depends([alias,arm_compute_core_so, arm_compute_core_a], generate_embed)
