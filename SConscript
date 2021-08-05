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
import json

VERSION = "v0.0-unreleased"
LIBRARY_VERSION_MAJOR = 24
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
    Default(obj)
    return obj

def build_objs(sources):

    obj = arm_compute_env.SharedObject(sources)
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

def get_cpu_runtime_files(operator):
    file_list = []
    operators = filelist['cpu']['operators']

    if "operator" in operators[operator]["files"]:
        file_list += operators[operator]["files"]["operator"]
    return file_list

def get_gpu_runtime_files(operator):
    file_list = []
    operators = filelist['gpu']['operators']

    if "operator" in operators[operator]["files"]:
        file_list += operators[operator]["files"]["operator"]
    return file_list

def get_cpu_kernel_files(operator):

    file_list = []
    file_list_sve = []
    operators = filelist['cpu']['operators']

    if env['estate'] == '64' and "neon" in operators[operator]['files'] and "estate64" in operators[operator]['files']['neon']:
        file_list += operators[operator]['files']['neon']['estate64']
    if env['estate'] == '32' and "neon" in operators[operator]['files'] and "estate32" in operators[operator]['files']['neon']:
        file_list += operators[operator]['files']['neon']['estate32']

    if "kernel" in operators[operator]["files"]:
        file_list += operators[operator]["files"]["kernel"]

    if ("neon" in operators[operator]["files"]):
        if any(i in env['data_type_support'] for i in ['all', 'qasymm8']) and ("qasymm8" in operators[operator]["files"]["neon"]):
            file_list += operators[operator]["files"]["neon"]["qasymm8"]
        if any(i in env['data_type_support'] for i in ['all', 'qasymm8_signed']) and ("qasymm8_signed" in operators[operator]["files"]["neon"]):
            file_list += operators[operator]["files"]["neon"]["qasymm8_signed"]
        if any(i in env['data_type_support'] for i in ['all', 'qsymm16']) and ("qsymm16" in operators[operator]["files"]["neon"]):
            file_list += operators[operator]["files"]["neon"]["qsymm16"]
        if any(i in env['data_type_support'] for i in ['all', 'integer']) and ("integer" in operators[operator]["files"]["neon"]):
                file_list += operators[operator]["files"]["neon"]["integer"]

    if (not "sve" in env['arch'] or env['fat_binary']) and ("neon" in operators[operator]["files"]):
        if any(i in env['data_type_support'] for i in ['all', 'fp16']) and ("fp16" in operators[operator]["files"]["neon"]):
            file_list += operators[operator]["files"]["neon"]["fp16"]
        if any(i in env['data_type_support'] for i in ['all', 'fp32']) and ("fp32" in operators[operator]["files"]["neon"]):
            file_list += operators[operator]["files"]["neon"]["fp32"]
        if any(i in env['data_layout_support'] for i in ['all', 'nchw']) and ("nchw" in operators[operator]["files"]["neon"]):
            file_list += operators[operator]['files']['neon']['nchw']
        if ("all" in operators[operator]["files"]["neon"]):
            file_list += operators[operator]["files"]["neon"]["all"]
    if ("sve" in env['arch'] or env['fat_binary']) and ("sve" in operators[operator]["files"]):
        if any(i in env['data_type_support'] for i in ['all', 'fp16']) and ("fp16" in operators[operator]["files"]["sve"]):
            file_list_sve += operators[operator]["files"]["sve"]["fp16"]
        if any(i in env['data_type_support'] for i in ['all', 'fp32'])  and ("fp32" in operators[operator]["files"]["sve"]):
            file_list_sve += operators[operator]["files"]["sve"]["fp32"]
        if any(i in env['data_type_support'] for i in ['all', 'qasymm8']) and ("qasymm8" in operators[operator]["files"]["sve"]):
            file_list_sve += operators[operator]["files"]["sve"]["qasymm8"]
        if any(i in env['data_type_support'] for i in ['all', 'qasymm8_signed']) and ("qasymm8_signed" in operators[operator]["files"]["sve"]):
            file_list_sve += operators[operator]["files"]["sve"]["qasymm8_signed"]
        if any(i in env['data_type_support'] for i in ['all', 'qsymm16']) and ("qsymm16" in operators[operator]["files"]["sve"]):
            file_list_sve += operators[operator]["files"]["sve"]["qsymm16"]
        if any(i in env['data_type_support'] for i in ['all', 'integer']) and ("integer" in operators[operator]["files"]["sve"]):
            file_list_sve += operators[operator]["files"]["sve"]["integer"]
        if ("all" in operators[operator]["files"]["sve"]):
            file_list_sve += operators[operator]["files"]["sve"]["all"]

    return file_list, file_list_sve

arm_compute_env = env.Clone()
version_file = arm_compute_env.Command("src/core/arm_compute_version.embed", "", action=create_version_file)
arm_compute_env.AlwaysBuild(version_file)

default_cpp_compiler = 'g++' if env['os'] not in ['android', 'macos'] else 'clang++'
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
                        'src/core/CL/cl_kernels/warp_helpers_quantized.h',
                        'src/core/CL/cl_kernels/warp_helpers.h'
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
                       'src/core/CL/cl_kernels/common/col2im.cl',
                       'src/core/CL/cl_kernels/common/convert_fc_weights.cl',
                       'src/core/CL/cl_kernels/common/copy_tensor.cl',
                       'src/core/CL/cl_kernels/common/crop_tensor.cl',
                       'src/core/CL/cl_kernels/common/deconvolution_layer.cl',
                       'src/core/CL/cl_kernels/common/dequantization_layer.cl',
                       'src/core/CL/cl_kernels/common/elementwise_operation.cl',
                       'src/core/CL/cl_kernels/common/elementwise_operation_quantized.cl',
                       'src/core/CL/cl_kernels/common/elementwise_unary.cl',
                       'src/core/CL/cl_kernels/common/fft_digit_reverse.cl',
                       'src/core/CL/cl_kernels/common/fft.cl',
                       'src/core/CL/cl_kernels/common/fft_scale.cl',
                       'src/core/CL/cl_kernels/common/fill_border.cl',
                       'src/core/CL/cl_kernels/common/floor.cl',
                       'src/core/CL/cl_kernels/common/gather.cl',
                       'src/core/CL/cl_kernels/common/gemm.cl',
                       'src/core/CL/cl_kernels/common/gemv.cl',
                       'src/core/CL/cl_kernels/common/gemm_v1.cl',
                       'src/core/CL/cl_kernels/common/gemmlowp.cl',
                       'src/core/CL/cl_kernels/common/generate_proposals.cl',
                       'src/core/CL/cl_kernels/common/generate_proposals_quantized.cl',
                       'src/core/CL/cl_kernels/common/instance_normalization.cl',
                       'src/core/CL/cl_kernels/common/l2_normalize.cl',
                       'src/core/CL/cl_kernels/common/mean_stddev_normalization.cl',
                       'src/core/CL/cl_kernels/common/unpooling_layer.cl',
                       'src/core/CL/cl_kernels/common/memset.cl',
                       'src/core/CL/cl_kernels/common/nonmax.cl',
                       'src/core/CL/cl_kernels/common/minmax_layer.cl',
                       'src/core/CL/cl_kernels/common/pad_layer.cl',
                       'src/core/CL/cl_kernels/common/permute.cl',
                       'src/core/CL/cl_kernels/common/pixelwise_mul_float.cl',
                       'src/core/CL/cl_kernels/common/pixelwise_mul_int.cl',
                       'src/core/CL/cl_kernels/common/qlstm_layer_normalization.cl',
                       'src/core/CL/cl_kernels/common/quantization_layer.cl',
                       'src/core/CL/cl_kernels/common/range.cl',
                       'src/core/CL/cl_kernels/common/reduction_operation.cl',
                       'src/core/CL/cl_kernels/common/pooling_layer.cl',
                       'src/core/CL/cl_kernels/common/reshape_layer.cl',
                       'src/core/CL/cl_kernels/common/convolution_layer.cl',
                       'src/core/CL/cl_kernels/common/reverse.cl',
                       'src/core/CL/cl_kernels/common/roi_align_layer.cl',
                       'src/core/CL/cl_kernels/common/roi_align_layer_quantized.cl',
                       'src/core/CL/cl_kernels/common/roi_pooling_layer.cl',
                       'src/core/CL/cl_kernels/common/select.cl',
                       'src/core/CL/cl_kernels/common/softmax_layer.cl',
                       'src/core/CL/cl_kernels/common/softmax_layer_quantized.cl',
                       'src/core/CL/cl_kernels/common/stack_layer.cl',
                       'src/core/CL/cl_kernels/common/slice_ops.cl',
                       'src/core/CL/cl_kernels/common/tile.cl',
                       'src/core/CL/cl_kernels/common/transpose.cl'
                    ]

    # NCHW kernels
    cl_files_nchw = ['src/core/CL/cl_kernels/nchw/batch_to_space.cl',
                    'src/core/CL/cl_kernels/nchw/batchnormalization_layer.cl',
                    'src/core/CL/cl_kernels/nchw/channel_shuffle.cl',
                    'src/core/CL/cl_kernels/nchw/depth_to_space.cl',
                    'src/core/CL/cl_kernels/nchw/direct_convolution_quantized.cl',
                    'src/core/CL/cl_kernels/nchw/direct_convolution1x1.cl',
                    'src/core/CL/cl_kernels/nchw/direct_convolution3x3.cl',
                    'src/core/CL/cl_kernels/nchw/direct_convolution5x5.cl',
                    'src/core/CL/cl_kernels/nchw/dequantization_layer.cl',
                    'src/core/CL/cl_kernels/nchw/im2col.cl',
                    'src/core/CL/cl_kernels/nchw/normalization_layer.cl',
                    'src/core/CL/cl_kernels/nchw/normalize_planar_yuv_layer.cl',
                    'src/core/CL/cl_kernels/nchw/normalize_planar_yuv_layer_quantized.cl',
                    'src/core/CL/cl_kernels/nchw/pooling_layer.cl',
                    'src/core/CL/cl_kernels/nchw/pooling_layer_quantized.cl',
                    'src/core/CL/cl_kernels/nchw/prior_box_layer.cl',
                    'src/core/CL/cl_kernels/nchw/remap.cl',
                    'src/core/CL/cl_kernels/nchw/reorg_layer.cl',
                    'src/core/CL/cl_kernels/nchw/scale.cl',
                    'src/core/CL/cl_kernels/nchw/scale_quantized.cl',
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
                    'src/core/CL/cl_kernels/nhwc/depth_to_space.cl',
                    'src/core/CL/cl_kernels/nhwc/dequantization_layer.cl',
                    'src/core/CL/cl_kernels/nhwc/dwc_native_fp_nhwc.cl',
                    'src/core/CL/cl_kernels/nhwc/dwc_native_quantized_nhwc.cl',
                    'src/core/CL/cl_kernels/nhwc/im2col.cl',
                    'src/core/CL/cl_kernels/nhwc/normalization_layer.cl',
                    'src/core/CL/cl_kernels/nhwc/normalize_planar_yuv_layer.cl',
                    'src/core/CL/cl_kernels/nhwc/normalize_planar_yuv_layer_quantized.cl',
                    'src/core/CL/cl_kernels/nhwc/pooling_layer.cl',
                    'src/core/CL/cl_kernels/nhwc/pooling_layer_quantized.cl',
                    'src/core/CL/cl_kernels/nhwc/remap.cl',
                    'src/core/CL/cl_kernels/nhwc/reorg_layer.cl',
                    'src/core/CL/cl_kernels/nhwc/scale.cl',
                    'src/core/CL/cl_kernels/nhwc/scale_quantized.cl',
                    'src/core/CL/cl_kernels/nhwc/space_to_batch.cl',
                    'src/core/CL/cl_kernels/nhwc/space_to_depth.cl',
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

runtime_files_hp = Glob('src/runtime/*.cpp')
runtime_files_hp += Glob('src/runtime/CPP/ICPPSimpleFunction.cpp')
runtime_files = Glob('src/runtime/CPP/functions/*.cpp')

# C API files
runtime_files_hp += filelist['c_api']['common']
runtime_files_hp += filelist['c_api']['operators']

if env['opencl']:
    runtime_files_hp += filelist['c_api']['gpu']

# Common backend files
core_files += filelist['common']

# Initialize high priority core files
core_files_hp = core_files
core_files_sve_hp = []
core_files = []

runtime_files_hp += Glob('src/runtime/CPP/SingleThreadScheduler.cpp')

graph_files = Glob('src/graph/*.cpp')
graph_files += Glob('src/graph/*/*.cpp')

if env['cppthreads']:
     runtime_files_hp += Glob('src/runtime/CPP/CPPScheduler.cpp')

if env['openmp']:
     runtime_files_hp += Glob('src/runtime/OMP/OMPScheduler.cpp')

if env['opencl']:
    operators = filelist['gpu']['operators']
    for operator in operators:
        if operator in filelist['gpu']['high_priority']:
            runtime_files_hp += get_gpu_runtime_files(operator)
            if "kernel" in operators[operator]["files"]:
                core_files_hp += operators[operator]["files"]["kernel"]
        else:
            runtime_files += get_gpu_runtime_files(operator)
            if "kernel" in operators[operator]["files"]:
                core_files += operators[operator]["files"]["kernel"]

    runtime_files_hp += filelist['gpu']['common']
    runtime_files += Glob('src/runtime/CL/functions/*.cpp')

    graph_files += Glob('src/graph/backends/CL/*.cpp')

sve_o = []
core_files_sve = []
if env['neon']:
    core_files += Glob('src/core/NEON/*.cpp')

    # build winograd/depthwise sources for either v7a / v8a
    arm_compute_env.Append(CPPPATH = ["src/core/NEON/kernels/convolution/common/",
                                      "src/core/NEON/kernels/convolution/winograd/",
                                      "src/core/NEON/kernels/convolution/depthwise/",
                                      "src/core/NEON/kernels/assembly/",
                                      "arm_compute/core/NEON/kernels/assembly/",
                                      "src/core/cpu/kernels/assembly/",])

    # Load files based on user's options
    operators = filelist['cpu']['operators']
    for operator in operators:
        if operator in filelist['cpu']['high_priority']:
            runtime_files_hp += get_cpu_runtime_files(operator)
            file_list, file_list_sve = get_cpu_kernel_files(operator)
            core_files_hp += file_list
            core_files_sve_hp += file_list_sve
        else:
            runtime_files += get_cpu_runtime_files(operator)
            file_list, file_list_sve = get_cpu_kernel_files(operator)
            core_files += file_list
            core_files_sve += file_list_sve

    runtime_files_hp += filelist['cpu']['common']
    runtime_files_hp += Glob('src/runtime/NEON/*.cpp')
    runtime_files += Glob('src/runtime/NEON/functions/*.cpp')

    graph_files += Glob('src/graph/backends/NEON/*.cpp')

bootcode_o = []
if env['os'] == 'bare_metal':
    bootcode_files = Glob('bootcode/*.s')
    bootcode_o = build_bootcode_objs(bootcode_files)
Export('bootcode_o')

high_priority_o = build_objs(core_files_hp + runtime_files_hp)
high_priority_sve_o = []
if (env['fat_binary']):
    sve_o = build_sve_objs(core_files_sve)
    high_priority_sve_o = build_sve_objs(core_files_sve_hp)
    arm_compute_a = build_library('arm_compute-static', arm_compute_env, core_files + sve_o + high_priority_o + high_priority_sve_o + runtime_files, static=True)
else:
    high_priority_o += build_objs(core_files_sve_hp)
    arm_compute_a = build_library('arm_compute-static', arm_compute_env, core_files + core_files_sve + high_priority_o + runtime_files, static=True)
Export('arm_compute_a')
if env['high_priority']:
    arm_compute_hp_a = build_library('arm_compute_hp-static', arm_compute_env, high_priority_o + high_priority_sve_o, static=True)
    Export('arm_compute_hp_a')

if env['os'] != 'bare_metal' and not env['standalone']:
    if (env['fat_binary']):
        arm_compute_so = build_library('arm_compute', arm_compute_env, core_files + sve_o + high_priority_sve_o + high_priority_o + runtime_files, static=False)
    else:
        arm_compute_so = build_library('arm_compute', arm_compute_env, core_files + core_files_sve + high_priority_o + runtime_files , static=False)

    Export('arm_compute_so')

    if env['high_priority']:
        arm_compute_hp_so = build_library('arm_compute_hp', arm_compute_env, high_priority_sve_o + high_priority_o, static=False)
        Export('arm_compute_hp_so')

# Generate dummy core lib for backwards compatibility
arm_compute_core_a = build_library('arm_compute_core-static', arm_compute_env, [], static=True)
Export('arm_compute_core_a')

if env['os'] != 'bare_metal' and not env['standalone']:
    arm_compute_core_a_so = build_library('arm_compute_core', arm_compute_env, [], static=False)
    Export('arm_compute_core_a_so')

arm_compute_graph_env = arm_compute_env.Clone()

arm_compute_graph_env.Append(CXXFLAGS = ['-Wno-redundant-move', '-Wno-pessimizing-move'])

arm_compute_graph_a = build_library('arm_compute_graph-static', arm_compute_graph_env, graph_files, static=True, libs = [ arm_compute_a])
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
