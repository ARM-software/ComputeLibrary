#!/usr/bin/python
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

import collections
import os.path
import re
import subprocess
import zlib
import json
import codecs

from SCons.Warnings import warn, DeprecatedWarning

warn(DeprecatedWarning,
     "DEPRECATION NOTICE: Legacy libarm_compute_core has been deprecated and is scheduled for removal in 24.02 release."
     " Link your application only to libarm_compute for core library functionality"
     )

VERSION = "v0.0-unreleased"
LIBRARY_VERSION_MAJOR = 33
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


# @brief Create a list of object from a given file list.
#
# @param  arch_info      A dictionary represents the architecture info such as the
#                        compiler flags and defines (filedefs.json).
#
# @param  sources        A list of files to build
#
# @return A list of objects for the corresponding architecture.

def build_obj_list(arch_info, sources, static=False):

    # Clone environment
    tmp_env = arm_compute_env.Clone()

    # Append architecture spec
    if 'cxxflags' in arch_info and len(arch_info['cxxflags']) > 0:
        tmp_env.Append(CXXFLAGS = arch_info['cxxflags'])

    # Build and return objects
    if static:
        objs = tmp_env.StaticObject(sources)
    else:
        objs = tmp_env.SharedObject(sources)

    tmp_env.Default(objs)
    return objs

# @brief Build multi-ISA files with the respective architecture.
#
# @return Two distinct lists:
#         A list of static objects
#         A list of shared objects

def build_lib_objects():
    lib_static_objs = [] # static objects
    lib_shared_objs = [] # shared objects

    arm_compute_env.Append(CPPDEFINES = ['ENABLE_NEON', 'ARM_COMPUTE_ENABLE_NEON',
                           'ENABLE_SVE', 'ARM_COMPUTE_ENABLE_SVE',
                           'ARM_COMPUTE_ENABLE_FP16', 'ARM_COMPUTE_ENABLE_BF16',
                           'ARM_COMPUTE_ENABLE_I8MM', 'ARM_COMPUTE_ENABLE_SVEF32MM'])

    # Build all the common files for the base architecture
    if env['arch'] == 'armv8a':
        lib_static_objs += build_obj_list(filedefs["armv8-a"], lib_files, static=True)
        lib_shared_objs += build_obj_list(filedefs["armv8-a"], lib_files, static=False)
    else:
        lib_static_objs += build_obj_list(filedefs["armv8.2-a"], lib_files, static=True)
        lib_shared_objs += build_obj_list(filedefs["armv8.2-a"], lib_files, static=False)

    # Build the SVE specific files
    lib_static_objs += build_obj_list(filedefs["armv8.2-a-sve"], lib_files_sve, static=True)
    lib_shared_objs += build_obj_list(filedefs["armv8.2-a-sve"], lib_files_sve, static=False)

    # Build the SVE2 specific files
    arm_compute_env.Append(CPPDEFINES = ['ARM_COMPUTE_ENABLE_SVE2'])
    lib_static_objs += build_obj_list(filedefs["armv8.6-a-sve2"], lib_files_sve2, static=True)
    lib_shared_objs += build_obj_list(filedefs["armv8.6-a-sve2"], lib_files_sve2, static=False)

    return lib_static_objs, lib_shared_objs


# The built-in SCons Glob() method does not support recursive searching of directories, thus we implement our own:
def recursive_glob(root_dir, pattern):
    files = []
    regex = re.compile(pattern)

    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            f = os.path.join(dirpath, f)
            if regex.match(f):
                files.append(f)

    return files


def get_ckw_obj_list():
    cmake_obj_dir = os.path.abspath("prototype/CMakeFiles/ckw_prototype.dir/src")
    return recursive_glob(root_dir=cmake_obj_dir, pattern=".*.o$")


def build_library(name, build_env, sources, static=False, libs=[]):
    cloned_build_env = build_env.Clone()
    if env['os'] == 'android' and static == False:
        cloned_build_env["LINKFLAGS"].remove('-pie')
        cloned_build_env["LINKFLAGS"].remove('-static-libstdc++')

    # -- Static Library --
    if static:
        # Recreate the list to avoid mutating the original
        static_sources = list(sources)

        # Dynamic Fusion has a direct dependency on the Compute Kernel Writer (CKW) subproject, therefore we collect the
        # built CKW objects to pack into the Compute Library archive.
        if env['experimental_dynamic_fusion'] and name == "arm_compute-static":
            static_sources += get_ckw_obj_list()

        obj = cloned_build_env.StaticLibrary(name, source=static_sources, LIBS=arm_compute_env["LIBS"] + libs)

    # -- Shared Library --
    else:
        # Always statically link Compute Library against CKW
        if env['experimental_dynamic_fusion'] and name == "arm_compute":
            libs.append('libckw_prototype.a')

        # Add shared library versioning
        if env['set_soname']:
            obj = cloned_build_env.SharedLibrary(name, source=sources, SHLIBVERSION = SONAME_VERSION, LIBS = arm_compute_env["LIBS"] + libs)
        else:
            obj = cloned_build_env.SharedLibrary(name, source=sources, LIBS = arm_compute_env["LIBS"] + libs)

    if env['mapfile']:
        if not 'windows' in env['os'] and not 'macos' in env['os']:
            cloned_build_env['LINKFLAGS'].append('"-Wl,-Map='+ name + '.map"')
        else:
            cloned_build_env['LINKFLAGS'].append('-Wl,-map,' + name + '.map')

    obj = install_lib(obj)
    build_env.Default(obj)
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
                    # Only get the header file name and discard the relative path.
                    # E.g. "src/core/CL/cl_kernels/activation_float_helpers.h" -> "activation_float_helpers.h"
                    include_file = found.group(1).split('/')[-1]
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
                file_to_write = zlib.compress(file_to_write.encode('utf-8'), 9)
                file_to_write = codecs.encode(file_to_write, "base64").decode('utf-8').replace("\n", "")
            file_to_write = "R\"(" + file_to_write + ")\""
            out_file.write(file_to_write)


def create_version_file(target, source, env):
# Generate string with build options library version to embed in the library:
    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"])
    except (OSError, subprocess.CalledProcessError):
        git_hash="unknown"

    build_options = str(vars.args).replace('"', '\\"')
    build_info = "\"arm_compute_version=%s Build options: %s Git hash=%s\"" % (VERSION,build_options, git_hash.strip())
    with open(target[0].get_path(), "w") as fd:
        fd.write(build_info)


def get_attrs_list(env, data_types, data_layouts):
    attrs = []

    # Manage data-types
    if 'all' in data_types:
        attrs += ['fp16', 'fp32', 'integer', 'qasymm8', 'qasymm8_signed', 'qsymm16']
    else:
        if 'fp16' in data_types: attrs += ['fp16']
        if 'fp32' in data_types: attrs += ['fp32']
        if 'integer' in data_types: attrs += ['integer']
        if 'qasymm8' in data_types: attrs += ['qasymm8']
        if 'qasymm8_signed' in data_types: attrs += ['qasymm8_signed']
        if 'qsymm16' in data_types: attrs += ['qsymm16']
    # Manage data-layouts
    if 'all' in data_layouts:
        attrs += ['nhwc', 'nchw']
    else:
        if 'nhwc' in data_layouts: attrs += ['nhwc']
        if 'nchw' in data_layouts: attrs += ['nchw']

    # Manage execution state
    attrs += ['estate32' if (env['estate'] == 'auto' and 'v7a' in env['arch']) or '32' in env['estate'] else 'estate64']

    return attrs


def get_operator_backend_files(filelist, operators, backend='', techs=[], attrs=[]):
    files = { "common" : [] }

    # Early return if filelist is empty
    if backend not in filelist:
        return files

    # Iterate over operators and create the file lists to compiler
    for operator in operators:
        if operator in filelist[backend]['operators']:
            files['common'] += filelist[backend]['operators'][operator]["files"]["common"]
            for tech in techs:
                if tech in filelist[backend]['operators'][operator]["files"]:
                    # Add tech as a key to dictionary if not there
                    if tech not in files:
                        files[tech] = []

                    # Add tech files to the tech file list
                    tech_files = filelist[backend]['operators'][operator]["files"][tech]
                    files[tech] += tech_files.get('common', [])
                    for attr in attrs:
                        files[tech] += tech_files.get(attr, [])

    # Remove duplicates if they exist
    return {k: list(set(v)) for k,v in files.items()}

def collect_operators(filelist, operators, backend=''):
    ops = set()
    for operator in operators:
        if operator in filelist[backend]['operators']:
            ops.add(operator)
            if 'deps' in filelist[backend]['operators'][operator]:
                ops.update(filelist[backend]['operators'][operator]['deps'])
        else:
            print("Operator {0} is unsupported on {1} backend!".format(operator, backend))

    return ops


def resolve_operator_dependencies(filelist, operators, backend=''):
    resolved_operators = collect_operators(filelist, operators, backend)

    are_ops_resolved = False
    while not are_ops_resolved:
        resolution_pass = collect_operators(filelist, resolved_operators, backend)
        if len(resolution_pass) != len(resolved_operators):
            resolved_operators.update(resolution_pass)
        else:
            are_ops_resolved = True

    return resolved_operators

def read_build_config_json(build_config):
    build_config_contents = {}
    custom_operators = []
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
        custom_operators = build_config_contents.get("operators", [])
        custom_types = build_config_contents.get("data_types", [])
        custom_layouts = build_config_contents.get("data_layouts", [])
    return custom_operators, custom_types, custom_layouts

arm_compute_env = env.Clone()
version_file = arm_compute_env.Command("src/core/arm_compute_version.embed", "", action=create_version_file)
arm_compute_env.AlwaysBuild(version_file)

default_cpp_compiler = 'g++' if env['os'] not in ['android', 'macos', 'openbsd'] else 'clang++'
cpp_compiler = os.environ.get('CXX', default_cpp_compiler)

# Generate embed files
generate_embed = [ version_file ]
if env['opencl'] and env['embed_kernels']:

    # Header files
    cl_helper_files = [ 'src/core/CL/cl_kernels/activation_float_helpers.h',
                        'src/core/CL/cl_kernels/activation_quant_helpers.h',
                        'src/core/CL/cl_kernels/gemm_helpers.h',
                        'src/core/CL/cl_kernels/helpers_asymm.h',
                        'src/core/CL/cl_kernels/helpers.h',
                        'src/core/CL/cl_kernels/load_store_utility.h',
                        'src/core/CL/cl_kernels/repeat.h',
                        'src/core/CL/cl_kernels/tile_helpers.h',
                        'src/core/CL/cl_kernels/types.h',
                        'src/core/CL/cl_kernels/warp_helpers.h',
                    ]

    # Common kernels
    cl_files_common = ['src/core/CL/cl_kernels/common/activation_layer.cl',
                       'src/core/CL/cl_kernels/common/activation_layer_quant.cl',
                       'src/core/CL/cl_kernels/common/arg_min_max.cl',
                       'src/core/CL/cl_kernels/common/batchnormalization_layer.cl',
                       'src/core/CL/cl_kernels/common/bounding_box_transform.cl',
                       'src/core/CL/cl_kernels/common/bounding_box_transform_quantized.cl',
                       'src/core/CL/cl_kernels/common/bitwise_op.cl',
                       'src/core/CL/cl_kernels/common/cast.cl',
                       'src/core/CL/cl_kernels/common/comparisons.cl',
                       'src/core/CL/cl_kernels/common/concatenate.cl',
                       'src/core/CL/cl_kernels/common/convolution_layer.cl',
                       'src/core/CL/cl_kernels/common/col2im.cl',
                       'src/core/CL/cl_kernels/common/convert_fc_weights.cl',
                       'src/core/CL/cl_kernels/common/copy_tensor.cl',
                       'src/core/CL/cl_kernels/common/crop_tensor.cl',
                       'src/core/CL/cl_kernels/common/deconvolution_layer.cl',
                       'src/core/CL/cl_kernels/common/dequantization_layer.cl',
                       'src/core/CL/cl_kernels/common/elementwise_operation.cl',
                       'src/core/CL/cl_kernels/common/elementwise_operation_quantized.cl',
                       'src/core/CL/cl_kernels/common/elementwise_unary.cl',
                       'src/core/CL/cl_kernels/common/elementwise_unary_quantized.cl',
                       'src/core/CL/cl_kernels/common/fft_digit_reverse.cl',
                       'src/core/CL/cl_kernels/common/fft.cl',
                       'src/core/CL/cl_kernels/common/fft_scale.cl',
                       'src/core/CL/cl_kernels/common/fill_border.cl',
                       'src/core/CL/cl_kernels/common/floor.cl',
                       'src/core/CL/cl_kernels/common/gather.cl',
                       'src/core/CL/cl_kernels/common/gemm.cl',
                       'src/core/CL/cl_kernels/common/gemm_reshaped_only_rhs_mmul.cl',
                       'src/core/CL/cl_kernels/common/gemm_utils.cl',
                       'src/core/CL/cl_kernels/common/gemmlowp.cl',
                       'src/core/CL/cl_kernels/common/gemmlowp_reshaped_only_rhs_mmul.cl',
                       'src/core/CL/cl_kernels/common/gemv.cl',
                       'src/core/CL/cl_kernels/common/generate_proposals.cl',
                       'src/core/CL/cl_kernels/common/generate_proposals_quantized.cl',
                       'src/core/CL/cl_kernels/common/instance_normalization.cl',
                       'src/core/CL/cl_kernels/common/l2_normalize.cl',
                       'src/core/CL/cl_kernels/common/mat_mul.cl',
                       'src/core/CL/cl_kernels/common/mat_mul_mmul.cl',
                       'src/core/CL/cl_kernels/common/mat_mul_quantized.cl',
                       'src/core/CL/cl_kernels/common/mat_mul_quantized_mmul.cl',
                       'src/core/CL/cl_kernels/common/mean_stddev_normalization.cl',
                       'src/core/CL/cl_kernels/common/memset.cl',
                       'src/core/CL/cl_kernels/common/minmax_layer.cl',
                       'src/core/CL/cl_kernels/common/nonmax.cl',
                       'src/core/CL/cl_kernels/common/pad_layer.cl',
                       'src/core/CL/cl_kernels/common/permute.cl',
                       'src/core/CL/cl_kernels/common/pixelwise_mul_float.cl',
                       'src/core/CL/cl_kernels/common/pixelwise_mul_int.cl',
                       'src/core/CL/cl_kernels/common/qlstm_layer_normalization.cl',
                       'src/core/CL/cl_kernels/common/quantization_layer.cl',
                       'src/core/CL/cl_kernels/common/range.cl',
                       'src/core/CL/cl_kernels/common/reduction_operation.cl',
                       'src/core/CL/cl_kernels/common/reshape_layer.cl',
                       'src/core/CL/cl_kernels/common/reverse.cl',
                       'src/core/CL/cl_kernels/common/roi_align_layer.cl',
                       'src/core/CL/cl_kernels/common/roi_align_layer_quantized.cl',
                       'src/core/CL/cl_kernels/common/roi_pooling_layer.cl',
                       'src/core/CL/cl_kernels/common/select.cl',
                       'src/core/CL/cl_kernels/common/slice_ops.cl',
                       'src/core/CL/cl_kernels/common/softmax_layer.cl',
                       'src/core/CL/cl_kernels/common/stack_layer.cl',
                       'src/core/CL/cl_kernels/common/tile.cl',
                       'src/core/CL/cl_kernels/common/transpose.cl',
                       'src/core/CL/cl_kernels/common/unpooling_layer.cl'
                    ]

    # NCHW kernels
    cl_files_nchw = ['src/core/CL/cl_kernels/nchw/batch_to_space.cl',
                    'src/core/CL/cl_kernels/nchw/batchnormalization_layer.cl',
                    'src/core/CL/cl_kernels/nchw/channel_shuffle.cl',
                    'src/core/CL/cl_kernels/nchw/depth_to_space.cl',
                    'src/core/CL/cl_kernels/nchw/direct_convolution.cl',
                    'src/core/CL/cl_kernels/nchw/dequantization_layer.cl',
                    'src/core/CL/cl_kernels/nchw/im2col.cl',
                    'src/core/CL/cl_kernels/nchw/normalization_layer.cl',
                    'src/core/CL/cl_kernels/nchw/normalize_planar_yuv_layer.cl',
                    'src/core/CL/cl_kernels/nchw/normalize_planar_yuv_layer_quantized.cl',
                    'src/core/CL/cl_kernels/nchw/pooling_layer.cl',
                    'src/core/CL/cl_kernels/nchw/prior_box_layer.cl',
                    'src/core/CL/cl_kernels/nchw/reorg_layer.cl',
                    'src/core/CL/cl_kernels/nchw/scale.cl',
                    'src/core/CL/cl_kernels/nchw/space_to_batch.cl',
                    'src/core/CL/cl_kernels/nchw/space_to_depth.cl',
                    'src/core/CL/cl_kernels/nchw/upsample_layer.cl',
                    'src/core/CL/cl_kernels/nchw/winograd_filter_transform.cl',
                    'src/core/CL/cl_kernels/nchw/winograd_input_transform.cl',
                    'src/core/CL/cl_kernels/nchw/winograd_output_transform.cl'
                ]

    # NHWC kernels
    cl_files_nhwc = ['src/core/CL/cl_kernels/nhwc/batch_to_space.cl',
                    'src/core/CL/cl_kernels/nhwc/batchnormalization_layer.cl',
                    'src/core/CL/cl_kernels/nhwc/channel_shuffle.cl',
                    'src/core/CL/cl_kernels/nhwc/direct_convolution.cl',
                    'src/core/CL/cl_kernels/nhwc/direct_convolution3d.cl',
                    'src/core/CL/cl_kernels/nhwc/depth_to_space.cl',
                    'src/core/CL/cl_kernels/nhwc/dequantization_layer.cl',
                    'src/core/CL/cl_kernels/nhwc/dwc_native_fp_nhwc.cl',
                    'src/core/CL/cl_kernels/nhwc/dwc_native_quantized_nhwc.cl',
                    'src/core/CL/cl_kernels/nhwc/im2col.cl',
                    'src/core/CL/cl_kernels/nhwc/indirect_convolution.cl',
                    'src/core/CL/cl_kernels/nhwc/normalization_layer.cl',
                    'src/core/CL/cl_kernels/nhwc/normalize_planar_yuv_layer.cl',
                    'src/core/CL/cl_kernels/nhwc/normalize_planar_yuv_layer_quantized.cl',
                    'src/core/CL/cl_kernels/nhwc/pooling_layer.cl',
                    'src/core/CL/cl_kernels/nhwc/pooling_3d_layer.cl',
                    'src/core/CL/cl_kernels/nhwc/pooling_3d_layer_quantized.cl',
                    'src/core/CL/cl_kernels/nhwc/pooling_layer_quantized.cl',
                    'src/core/CL/cl_kernels/nhwc/reorg_layer.cl',
                    'src/core/CL/cl_kernels/nhwc/scale.cl',
                    'src/core/CL/cl_kernels/nhwc/space_to_batch.cl',
                    'src/core/CL/cl_kernels/nhwc/space_to_depth.cl',
                    'src/core/CL/cl_kernels/nhwc/transposed_convolution.cl',
                    'src/core/CL/cl_kernels/nhwc/upsample_layer.cl',
                    'src/core/CL/cl_kernels/nhwc/winograd_filter_transform.cl',
                    'src/core/CL/cl_kernels/nhwc/winograd_input_transform.cl',
                    'src/core/CL/cl_kernels/nhwc/winograd_output_transform.cl'
                ]

    cl_files = cl_helper_files + cl_files_common + cl_files_nchw + cl_files_nhwc

    embed_files = [ f+"embed" for f in cl_files ]
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
if not env['thread_sanitizer']:
    arm_compute_env.Append(LINKFLAGS=[undefined_flag])
arm_compute_env.Append(CPPPATH =[Dir("./src/core/").path] )

if env['os'] != 'openbsd':
    if env['os'] == 'windows':
        arm_compute_env.Append(LIBS = [])
    else:
        arm_compute_env.Append(LIBS = ['dl'])


# Load build definitions file
with (open(Dir('#').path + '/filedefs.json')) as fd:
    filedefs = json.load(fd)
    filedefs = filedefs['cpu']['arch']


with (open(Dir('#').path + '/filelist.json')) as fp:
    filelist = json.load(fp)

# Common backend files
lib_files = filelist['common']

# Fixed format GEMM kernels.
if env['fixed_format_kernels']:
    arm_compute_env.Append(CPPDEFINES = ['ARM_COMPUTE_ENABLE_FIXED_FORMAT_KERNELS'])

# Experimental files
# Dynamic fusion
if env['experimental_dynamic_fusion']:
    lib_files += filelist['experimental']['dynamic_fusion']['common']
    lib_files += filelist['experimental']['dynamic_fusion']['template_writer']

if "ACL_INTERNAL_TEST_CKW_IN_DF" in env["extra_cxx_flags"]:
    if not env["experimental_dynamic_fusion"]:
        print("To use ACL_INTERNAL_TEST_CKW_IN_DF experimental_dynamic_fusion must be set to 1")
        Exit(1)
    lib_files += filelist['experimental']['dynamic_fusion']['ckw_driver']

# Logging files
if env["logging"]:
    lib_files += filelist['logging']

# C API files
lib_files += filelist['c_api']['common']
lib_files += filelist['c_api']['operators']

# Scheduler infrastructure
lib_files += filelist['scheduler']['single']
if env['cppthreads']:
     lib_files += filelist['scheduler']['threads']
if env['openmp']:
     lib_files += filelist['scheduler']['omp']

# Graph files
graph_files = Glob('src/graph/*.cpp')
graph_files += Glob('src/graph/*/*.cpp')

# Specify user-defined priority operators
custom_operators = []
custom_types = []
custom_layouts = []

use_custom_ops = env['high_priority'] or env['build_config']

if env['high_priority']:
    custom_operators = filelist['high_priority']
    custom_types = ['all']
    custom_layouts = ['all']

if env['build_config']:
    custom_operators, custom_types, custom_layouts = read_build_config_json(env['build_config'])

if env['opencl']:
    lib_files += filelist['c_api']['gpu']
    lib_files += filelist['gpu']['common']

    cl_operators = custom_operators if use_custom_ops else filelist['gpu']['operators'].keys()
    cl_ops_to_build = resolve_operator_dependencies(filelist, cl_operators, 'gpu')
    lib_files += get_operator_backend_files(filelist, cl_ops_to_build, 'gpu')['common']

    graph_files += Glob('src/graph/backends/CL/*.cpp')


lib_files_sve = []
lib_files_sve2 = []

if env['neon']:
    # build winograd/depthwise sources for either v7a / v8a
    arm_compute_env.Append(CPPPATH = ["src/core/NEON/kernels/arm_gemm",
                                      "src/core/NEON/kernels/convolution/common/",
                                      "src/core/NEON/kernels/convolution/winograd/",
                                      "src/core/NEON/kernels/arm_conv/depthwise/",
                                      "src/core/NEON/kernels/arm_conv/pooling/",
                                      "src/core/NEON/kernels/arm_conv/",
                                      "src/core/NEON/kernels/assembly/",
                                      "arm_compute/core/NEON/kernels/assembly/",
                                      "src/cpu/kernels/assembly/"])

    lib_files += filelist['cpu']['common']

    # Setup SIMD file list to include
    simd = ['neon']
    if env['multi_isa']:
        simd += ['sve', 'sve2']
    else:
        if 'sve' in env['arch']: simd += ['sve']
        if 'sve2' in env['arch']: simd += ['sve2']

    # Get attributes
    if(use_custom_ops):
        attrs = get_attrs_list(env, custom_types, custom_layouts)
    else:
        attrs = get_attrs_list(env, env['data_type_support'], env['data_layout_support'])


    if env['fixed_format_kernels']:
        attrs.append("fixed_format_kernels")

    # Setup data-type and data-layout files to include
    cpu_operators = custom_operators if use_custom_ops else filelist['cpu']['operators'].keys()
    cpu_ops_to_build = resolve_operator_dependencies(filelist, cpu_operators, 'cpu')

    cpu_files = get_operator_backend_files(filelist, cpu_ops_to_build, 'cpu', simd, attrs)

    # Shared among ALL CPU files
    lib_files += cpu_files.get('common', [])

    # Arm® Neon™ specific files
    lib_files += cpu_files.get('neon', [])

    # SVE files only
    lib_files_sve = cpu_files.get('sve', [])

    # SVE2 files only
    lib_files_sve2 = cpu_files.get('sve2', [])

    graph_files += Glob('src/graph/backends/NEON/*.cpp')

# Restrict from building graph API if a reduced operator list has been provided
if use_custom_ops:
    print("WARNING: Graph library requires all operators to be built")
    graph_files = []

# Build bootcode in case of bare-metal
bootcode_o = []
if env['os'] == 'bare_metal':
    bootcode_files = Glob('bootcode/*.s')
    bootcode_o = build_bootcode_objs(bootcode_files)
Export('bootcode_o')


if (env['multi_isa']):
    lib_static_objs, lib_shared_objs = build_lib_objects()


# STATIC library build.
if (env['multi_isa']):
    arm_compute_a = build_library('arm_compute-static', arm_compute_env, lib_static_objs, static=True)
else:
    if 'sve2' in env['arch']:
        lib_files += lib_files_sve
        lib_files += lib_files_sve2
    elif 'sve' in env['arch']:
        lib_files += lib_files_sve

    arm_compute_a = build_library('arm_compute-static', arm_compute_env, lib_files, static=True)

Export('arm_compute_a')

# SHARED library build.
if env['os'] != 'bare_metal' and not env['standalone']:
    if (env['multi_isa']):

        arm_compute_so = build_library('arm_compute', arm_compute_env, lib_shared_objs, static=False)
    else:
        arm_compute_so = build_library('arm_compute', arm_compute_env, lib_files, static=False)

    Export('arm_compute_so')

# Generate dummy core lib for backwards compatibility
if env['os'] == 'macos':
    # macos static library archiver fails if given an empty list of files
    arm_compute_core_a = build_library('arm_compute_core-static', arm_compute_env, lib_files, static=True)
else:
    arm_compute_core_a = build_library('arm_compute_core-static', arm_compute_env, [], static=True)

Export('arm_compute_core_a')

if env['os'] != 'bare_metal' and not env['standalone']:
    arm_compute_core_a_so = build_library('arm_compute_core', arm_compute_env, [], static=False)
    Export('arm_compute_core_a_so')

arm_compute_graph_env = arm_compute_env.Clone()

# Build graph libraries
arm_compute_graph_env.Append(CXXFLAGS = ['-Wno-redundant-move', '-Wno-pessimizing-move'])

arm_compute_graph_a = build_library('arm_compute_graph-static', arm_compute_graph_env, graph_files, static=True)
Export('arm_compute_graph_a')

if env['os'] != 'bare_metal' and not env['standalone']:
    arm_compute_graph_so = build_library('arm_compute_graph', arm_compute_graph_env, graph_files, static=False, libs = [ "arm_compute" ])
    Depends(arm_compute_graph_so, arm_compute_so)
    Export('arm_compute_graph_so')

if env['standalone']:
    alias = arm_compute_env.Alias("arm_compute", [arm_compute_a])
else:
    alias = arm_compute_env.Alias("arm_compute", [arm_compute_a, arm_compute_so])

Default(alias)

if env['standalone']:
    Depends([alias], generate_embed)
else:
    Depends([alias], generate_embed)
