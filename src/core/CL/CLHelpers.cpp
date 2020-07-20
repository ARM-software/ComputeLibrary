/*
 * Copyright (c) 2016-2020 Arm Limited.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLCoreRuntimeContext.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/CLTypes.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Log.h"
#include "arm_compute/core/Types.h"

#include <utility>
#include <vector>

namespace arm_compute
{
std::string get_cl_type_from_data_type(const DataType &dt)
{
    switch(dt)
    {
        case DataType::U8:
        case DataType::QASYMM8:
            return "uchar";
        case DataType::S8:
        case DataType::QASYMM8_SIGNED:
        case DataType::QSYMM8:
        case DataType::QSYMM8_PER_CHANNEL:
            return "char";
        case DataType::U16:
        case DataType::QASYMM16:
            return "ushort";
        case DataType::S16:
        case DataType::QSYMM16:
            return "short";
        case DataType::U32:
            return "uint";
        case DataType::S32:
            return "int";
        case DataType::U64:
            return "ulong";
        case DataType::S64:
            return "long";
        case DataType::F16:
            return "half";
        case DataType::F32:
            return "float";
        default:
            ARM_COMPUTE_ERROR("Unsupported input data type.");
            return "";
    }
}

std::string get_cl_promoted_type_from_data_type(const DataType &dt)
{
    switch(dt)
    {
        case DataType::U8:
        case DataType::QASYMM8:
            return "ushort";
        case DataType::S8:
        case DataType::QASYMM8_SIGNED:
        case DataType::QSYMM8:
        case DataType::QSYMM8_PER_CHANNEL:
            return "short";
        case DataType::U16:
        case DataType::QASYMM16:
            return "uint";
        case DataType::S16:
        case DataType::QSYMM16:
            return "int";
        case DataType::U32:
            return "ulong";
        case DataType::S32:
            return "long";
        case DataType::F16:
            return "float";
        default:
            ARM_COMPUTE_ERROR("Cannot get promoted OpenCL type for the input data type.");
            return "";
    }
}

std::string get_cl_unsigned_type_from_element_size(size_t element_size)
{
    switch(element_size)
    {
        case 1:
            return "uchar";
        case 2:
            return "ushort";
        case 4:
            return "uint";
        case 8:
            return "ulong";
        default:
            ARM_COMPUTE_ERROR("Data type not supported");
            return "";
    }
}

std::string get_cl_signed_type_from_element_size(size_t element_size)
{
    switch(element_size)
    {
        case 1:
            return "char";
        case 2:
            return "short";
        case 4:
            return "int";
        case 8:
            return "long";
        default:
            ARM_COMPUTE_ERROR("Data type not supported");
            return "";
    }
}

std::string get_cl_select_type_from_data_type(const DataType &dt)
{
    switch(dt)
    {
        case DataType::U8:
        case DataType::QASYMM8:
            return "uchar";
        case DataType::S8:
        case DataType::QASYMM8_SIGNED:
        case DataType::QSYMM8:
        case DataType::QSYMM8_PER_CHANNEL:
            return "char";
        case DataType::U16:
        case DataType::QASYMM16:
            return "ushort";
        case DataType::F16:
        case DataType::S16:
        case DataType::QSYMM16:
            return "short";
        case DataType::U32:
            return "uint";
        case DataType::F32:
        case DataType::S32:
            return "int";
        case DataType::U64:
            return "ulong";
        case DataType::S64:
            return "long";
        default:
            ARM_COMPUTE_ERROR("Unsupported input data type.");
            return "";
    }
}

std::string get_cl_dot8_acc_type_from_data_type(const DataType &dt)
{
    switch(dt)
    {
        case DataType::U8:
        case DataType::QASYMM8:
            return "uint";
        case DataType::S8:
        case DataType::QASYMM8_SIGNED:
        case DataType::QSYMM8:
        case DataType::QSYMM8_PER_CHANNEL:
            return "int";
        default:
            ARM_COMPUTE_ERROR("Unsupported data type.");
            return "";
    }
}

std::string get_data_size_from_data_type(const DataType &dt)
{
    switch(dt)
    {
        case DataType::U8:
        case DataType::S8:
        case DataType::QSYMM8:
        case DataType::QASYMM8:
        case DataType::QASYMM8_SIGNED:
        case DataType::QSYMM8_PER_CHANNEL:
            return "8";
        case DataType::U16:
        case DataType::S16:
        case DataType::QSYMM16:
        case DataType::QASYMM16:
        case DataType::F16:
            return "16";
        case DataType::U32:
        case DataType::S32:
        case DataType::F32:
            return "32";
        case DataType::U64:
        case DataType::S64:
            return "64";
        default:
            ARM_COMPUTE_ERROR("Unsupported input data type.");
            return "0";
    }
}

GPUTarget get_target_from_device(const cl::Device &device)
{
    // Query device name size
    std::string device_name = device.getInfo<CL_DEVICE_NAME>();

    return get_target_from_name(device_name);
}

bool arm_non_uniform_workgroup_supported(const cl::Device &device)
{
    return device_supports_extension(device, "cl_arm_non_uniform_work_group_size");
}

bool fp16_supported(const cl::Device &device)
{
    return device_supports_extension(device, "cl_khr_fp16");
}

bool dot8_supported(const cl::Device &device)
{
    std::string     device_name = device.getInfo<CL_DEVICE_NAME>();
    const GPUTarget gpu_target  = get_target_from_name(device_name);

    // SW_WORKAROUND: Workaround for DDK revision r14p0.to enable cl_arm_integer_dot_product_int8
    std::set<GPUTarget> sw_workaround_issue = { GPUTarget::G76 };
    return (device_supports_extension(device, "cl_arm_integer_dot_product_int8") || sw_workaround_issue.count(gpu_target) != 0);
}

bool dot8_acc_supported(const cl::Device &device)
{
    return device_supports_extension(device, "cl_arm_integer_dot_product_accumulate_int8");
}

CLVersion get_cl_version(const cl::Device &device)
{
    std::string version_str = device.getInfo<CL_DEVICE_VERSION>();
    if(version_str.find("OpenCL 2") != std::string::npos)
    {
        return CLVersion::CL20;
    }
    else if(version_str.find("OpenCL 1.2") != std::string::npos)
    {
        return CLVersion::CL12;
    }
    else if(version_str.find("OpenCL 1.1") != std::string::npos)
    {
        return CLVersion::CL11;
    }
    else if(version_str.find("OpenCL 1.0") != std::string::npos)
    {
        return CLVersion::CL10;
    }

    return CLVersion::UNKNOWN;
}

bool device_supports_extension(const cl::Device &device, const char *extension_name)
{
    std::string extensions = device.getInfo<CL_DEVICE_EXTENSIONS>();
    auto        pos        = extensions.find(extension_name);
    return (pos != std::string::npos);
}

bool cl_winograd_convolution_layer_supported(const Size2D &output_tile, const Size2D &kernel_size, DataLayout data_layout)
{
    ARM_COMPUTE_ERROR_ON(data_layout == DataLayout::UNKNOWN);

    using WinogradConfiguration = std::pair<std::pair<int, int>, std::pair<int, int>>;

    std::vector<WinogradConfiguration> winograd_configs_nchw =
    {
        WinogradConfiguration(std::pair<int, int>(1, 2), std::pair<int, int>(1, 3)),
        WinogradConfiguration(std::pair<int, int>(1, 4), std::pair<int, int>(1, 3)),
        WinogradConfiguration(std::pair<int, int>(2, 1), std::pair<int, int>(3, 1)),
        WinogradConfiguration(std::pair<int, int>(4, 1), std::pair<int, int>(3, 1)),
        WinogradConfiguration(std::pair<int, int>(2, 2), std::pair<int, int>(3, 3)),
        WinogradConfiguration(std::pair<int, int>(4, 4), std::pair<int, int>(3, 3)),
        WinogradConfiguration(std::pair<int, int>(4, 4), std::pair<int, int>(5, 5)),
        WinogradConfiguration(std::pair<int, int>(4, 1), std::pair<int, int>(5, 1)),
        WinogradConfiguration(std::pair<int, int>(1, 4), std::pair<int, int>(1, 5))
    };

    std::vector<WinogradConfiguration> winograd_configs_nhwc =
    {
        WinogradConfiguration(std::pair<int, int>(2, 2), std::pair<int, int>(3, 3)),
        WinogradConfiguration(std::pair<int, int>(1, 4), std::pair<int, int>(1, 3)),
        WinogradConfiguration(std::pair<int, int>(4, 1), std::pair<int, int>(3, 1)),
        WinogradConfiguration(std::pair<int, int>(4, 4), std::pair<int, int>(3, 3)),
        WinogradConfiguration(std::pair<int, int>(4, 4), std::pair<int, int>(5, 5)),
        WinogradConfiguration(std::pair<int, int>(4, 1), std::pair<int, int>(5, 1)),
        WinogradConfiguration(std::pair<int, int>(1, 4), std::pair<int, int>(1, 5)),
        WinogradConfiguration(std::pair<int, int>(1, 2), std::pair<int, int>(1, 7)),
        WinogradConfiguration(std::pair<int, int>(2, 1), std::pair<int, int>(7, 1)),
        WinogradConfiguration(std::pair<int, int>(2, 2), std::pair<int, int>(7, 7)),
    };

    auto p = std::make_pair(std::pair<int, int>(output_tile.width, output_tile.height),
                            std::pair<int, int>(kernel_size.width, kernel_size.height));

    // Return true if supported
    if(data_layout == DataLayout::NCHW)
    {
        return (std::find(winograd_configs_nchw.begin(), winograd_configs_nchw.end(), p) != winograd_configs_nchw.end());
    }
    else
    {
        return (std::find(winograd_configs_nhwc.begin(), winograd_configs_nhwc.end(), p) != winograd_configs_nhwc.end());
    }
}

size_t preferred_vector_width(const cl::Device &device, const DataType dt)
{
    switch(dt)
    {
        case DataType::U8:
        case DataType::S8:
        case DataType::QASYMM8:
        case DataType::QASYMM8_SIGNED:
        case DataType::QSYMM8:
        case DataType::QSYMM8_PER_CHANNEL:
            return device.getInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR>();
        case DataType::U16:
        case DataType::S16:
        case DataType::QSYMM16:
        case DataType::QASYMM16:
            return device.getInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT>();
        case DataType::U32:
        case DataType::S32:
            return device.getInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT>();
        case DataType::F16:
        case DataType::F32:
            return device.getInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT>();
        case DataType::U64:
        case DataType::S64:
            return device.getInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG>();
        default:
            return 1;
    }
}

bool preferred_dummy_work_items_support(const cl::Device &device)
{
    ARM_COMPUTE_UNUSED(device);
    // TODO (COMPMID-2044)
    return true;
}

bool image2d_from_buffer_supported(const cl::Device &device)
{
    return device_supports_extension(device, "cl_khr_image2d_from_buffer");
}

size_t get_cl_image_pitch_alignment(const cl::Device &device)
{
    cl_uint pixel_aligment = 0;

    cl_int err = clGetDeviceInfo(device(), CL_DEVICE_IMAGE_PITCH_ALIGNMENT, sizeof(cl_uint), &pixel_aligment, nullptr);

    if(err == CL_SUCCESS)
    {
        return pixel_aligment;
    }
    else
    {
        return 0;
    }
}

cl::Kernel create_opencl_kernel(CLCoreRuntimeContext *ctx, const std::string &kernel_name, const CLBuildOptions &build_opts)
{
    if(ctx && ctx->kernel_library())
    {
        // New api going through the core context
        return static_cast<cl::Kernel>(ctx->kernel_library()->create_kernel(kernel_name, build_opts.options()));
    }
    else
    {
        // Legacy code through the singleton
        return static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel(kernel_name, build_opts.options()));
    }
}

cl::Kernel create_kernel(const CLCompileContext &ctx, const std::string &kernel_name, const std::set<std::string> &build_opts)
{
    const std::string program_name = CLKernelLibrary::get().get_program_name(kernel_name);
    std::pair<std::string, bool> kernel_src = CLKernelLibrary::get().get_program(program_name);
    const std::string kernel_path = CLKernelLibrary::get().get_kernel_path();
    return static_cast<cl::Kernel>(ctx.create_kernel(kernel_name, program_name, kernel_src.first, kernel_path, build_opts, kernel_src.second));
}

cl::NDRange create_lws_hint_parallel_implementations(unsigned int input_dimension, unsigned int vector_size)
{
    const unsigned int width_leftover = input_dimension % vector_size;
    const unsigned int border_width   = (width_leftover != 0) ? vector_size - width_leftover : 0;
    const unsigned int num_of_threads = ((input_dimension + border_width) / 16);
    return cl::NDRange(std::min(8U, num_of_threads));
}
} // namespace arm_compute
