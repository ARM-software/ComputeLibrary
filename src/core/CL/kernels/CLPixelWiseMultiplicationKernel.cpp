/*
 * Copyright (c) 2016-2019 ARM Limited.
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
#include "arm_compute/core/CL/kernels/CLPixelWiseMultiplicationKernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/CLValidate.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Window.h"

#include <cmath>
#include <cstdlib>
#include <set>
#include <string>

namespace arm_compute
{
namespace
{
constexpr unsigned int num_elems_processed_per_iteration = 16;

Status validate_arguments(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, float scale,
                          ConvertPolicy overflow_policy, RoundingPolicy rounding_policy)
{
    ARM_COMPUTE_UNUSED(overflow_policy);
    ARM_COMPUTE_UNUSED(rounding_policy);

    ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(input1);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input1, 1, DataType::U8, DataType::QASYMM8, DataType::S16, DataType::QSYMM16, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(input2);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input2, 1, DataType::U8, DataType::QASYMM8, DataType::S16, DataType::QSYMM16, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(scale < 0, "Scale cannot be negative.");

    const TensorShape &out_shape = TensorShape::broadcast_shape(input1->tensor_shape(), input2->tensor_shape());

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(out_shape.total_size() == 0, "Inputs are not broadcast compatible");

    // Validate in case of configured output
    if(output->total_size() > 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(output);
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::U8, DataType::QASYMM8, DataType::S16, DataType::QSYMM16, DataType::F16, DataType::F32);
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(output->data_type() == DataType::U8 && (input1->data_type() != DataType::U8 || input2->data_type() != DataType::U8),
                                        "Output can only be U8 if both inputs are U8");
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(output->data_type() == DataType::QASYMM8 && (input1->data_type() != DataType::QASYMM8 || input2->data_type() != DataType::QASYMM8),
                                        "Output can only be QASYMM8 if both inputs are QASYMM8");
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(output->data_type() == DataType::QSYMM16 && (input1->data_type() != DataType::QSYMM16 || input2->data_type() != DataType::QSYMM16),
                                        "Output can only be QSYMM16 if both inputs are QSYMM16");
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(detail::have_different_dimensions(out_shape, output->tensor_shape(), 0), "Wrong shape for output");
    }

    return Status{};
}

std::pair<Status, Window> validate_and_configure_window(ITensorInfo *input1, ITensorInfo *input2, ITensorInfo *output)
{
    const std::pair<TensorShape, ValidRegion> broadcast_pair = ITensorInfo::broadcast_shape_and_valid_region(*input1, *input2);
    const TensorShape &out_shape    = broadcast_pair.first;
    const ValidRegion &valid_region = broadcast_pair.second;

    // Auto initialize output if not initialized
    {
        set_shape_if_empty(*output, out_shape);

        if(input1->data_type() == DataType::S16 || input2->data_type() == DataType::S16)
        {
            set_format_if_unknown(*output, Format::S16);
        }
        else if(input1->data_type() == DataType::F32 || input2->data_type() == DataType::F32)
        {
            set_format_if_unknown(*output, Format::F32);
        }
        else if(input1->data_type() == DataType::QASYMM8)
        {
            set_data_type_if_unknown(*output, DataType::QASYMM8);
        }
        else if(input1->data_type() == DataType::QSYMM16)
        {
            set_data_type_if_unknown(*output, DataType::QSYMM16);
        }
    }

    Window win        = calculate_max_window(valid_region, Steps(num_elems_processed_per_iteration));
    Window win_input1 = win.broadcast_if_dimension_le_one(*input1);
    Window win_input2 = win.broadcast_if_dimension_le_one(*input2);

    AccessWindowHorizontal input1_access(input1, 0, num_elems_processed_per_iteration);
    AccessWindowHorizontal input2_access(input2, 0, num_elems_processed_per_iteration);
    AccessWindowHorizontal output_access(output, 0, num_elems_processed_per_iteration);

    bool window_changed = update_window_and_padding(win_input1, input1_access)
                          || update_window_and_padding(win_input2, input2_access)
                          || update_window_and_padding(win, output_access);

    output_access.set_valid_region(win, valid_region);

    Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
    return std::make_pair(err, win);
}
} // namespace

CLPixelWiseMultiplicationKernel::CLPixelWiseMultiplicationKernel()
    : _input1(nullptr), _input2(nullptr), _output(nullptr)
{
}

void CLPixelWiseMultiplicationKernel::configure(const ICLTensor *input1, const ICLTensor *input2, ICLTensor *output, float scale,
                                                ConvertPolicy overflow_policy, RoundingPolicy rounding_policy)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input1, input2, output);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input1->info(), input2->info(), output->info(),
                                                  scale, overflow_policy, rounding_policy));

    // Configure kernel window
    auto win_config = validate_and_configure_window(input1->info(), input2->info(), output->info());
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);

    _input1 = input1;
    _input2 = input2;
    _output = output;

    int scale_int = -1;
    // Extract sign, exponent and mantissa
    int   exponent            = 0;
    float normalized_mantissa = std::frexp(scale, &exponent);
    // Use int scaling if factor is equal to 1/2^n for 0 <= n <= 15
    // frexp returns 0.5 as mantissa which means that the exponent will be in the range of -1 <= e <= 14
    // Moreover, it will be negative as we deal with 1/2^n
    if((normalized_mantissa == 0.5f) && (-14 <= exponent) && (exponent <= 1))
    {
        // Store the positive exponent. We know that we compute 1/2^n
        // Additionally we need to subtract 1 to compensate that frexp used a mantissa of 0.5
        scale_int = std::abs(exponent - 1);
    }

    std::string compute_type;
    // Check if it has float inputs and output
    if(is_data_type_float(input1->info()->data_type()) || is_data_type_float(input2->info()->data_type()))
    {
        scale_int    = -1;
        compute_type = (input1->info()->data_type() == DataType::F32 || input2->info()->data_type() == DataType::F32) ? "float" : "half";
    }
    else
    {
        if(input1->info()->data_type() == DataType::S16 || input2->info()->data_type() == DataType::S16)
        {
            compute_type = "int";
        }
        else
        {
            compute_type = "ushort";
        }
    }

    const bool is_quantized = is_data_type_quantized(input1->info()->data_type());

    // Set kernel build options
    std::string    kernel_name = "pixelwise_mul";
    CLBuildOptions build_opts;
    build_opts.add_option("-DDATA_TYPE_IN1=" + get_cl_type_from_data_type(input1->info()->data_type()));
    build_opts.add_option("-DDATA_TYPE_IN2=" + get_cl_type_from_data_type(input2->info()->data_type()));
    build_opts.add_option("-DDATA_TYPE_OUT=" + get_cl_type_from_data_type(output->info()->data_type()));
    build_opts.add_option("-DVEC_SIZE=" + support::cpp11::to_string(num_elems_processed_per_iteration));
    if(is_quantized)
    {
        const UniformQuantizationInfo iq1_info = input1->info()->quantization_info().uniform();
        const UniformQuantizationInfo iq2_info = input2->info()->quantization_info().uniform();
        const UniformQuantizationInfo oq_info  = output->info()->quantization_info().uniform();

        build_opts.add_option_if(is_data_type_quantized_asymmetric(input1->info()->data_type()),
                                 "-DOFFSET_IN1=" + support::cpp11::to_string(iq1_info.offset));
        build_opts.add_option_if(is_data_type_quantized_asymmetric(input2->info()->data_type()),
                                 "-DOFFSET_IN2=" + support::cpp11::to_string(iq2_info.offset));
        build_opts.add_option_if(is_data_type_quantized_asymmetric(output->info()->data_type()),
                                 "-DOFFSET_OUT=" + support::cpp11::to_string(oq_info.offset));
        build_opts.add_option("-DSCALE_IN1=" + float_to_string_with_full_precision(iq1_info.scale));
        build_opts.add_option("-DSCALE_IN2=" + float_to_string_with_full_precision(iq2_info.scale));
        build_opts.add_option("-DSCALE_OUT=" + float_to_string_with_full_precision(oq_info.scale));
        kernel_name += "_quantized";
    }
    else
    {
        kernel_name += (scale_int >= 0) ? "_int" : "_float";
        build_opts.add_option_if_else(overflow_policy == ConvertPolicy::WRAP || is_data_type_float(output->info()->data_type()), "-DWRAP", "-DSATURATE");
        build_opts.add_option_if_else(rounding_policy == RoundingPolicy::TO_ZERO, "-DROUND=_rtz", "-DROUND=_rte");
        build_opts.add_option("-DDATA_TYPE_RES=" + compute_type);
    }

    // Create kernel
    _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel(kernel_name, build_opts.options()));

    // Set scale argument
    unsigned int idx = 3 * num_arguments_per_3D_tensor(); // Skip the inputs and output parameters

    if(scale_int >= 0 && !is_quantized)
    {
        _kernel.setArg(idx++, scale_int);
    }
    else
    {
        _kernel.setArg(idx++, scale);
    }

    ICLKernel::configure_internal(win_config.second);
}

Status CLPixelWiseMultiplicationKernel::validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, float scale,
                                                 ConvertPolicy overflow_policy, RoundingPolicy rounding_policy)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input1, input2, output);
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input1, input2, output, scale, overflow_policy, rounding_policy));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(input1->clone().get(), input2->clone().get(), output->clone().get()).first);

    return Status{};
}

void CLPixelWiseMultiplicationKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

    const TensorShape &in_shape1 = _input1->info()->tensor_shape();
    const TensorShape &in_shape2 = _input2->info()->tensor_shape();
    const TensorShape &out_shape = _output->info()->tensor_shape();

    bool can_collapse = true;
    if(std::min(in_shape1.total_size(), in_shape2.total_size()) > 1)
    {
        can_collapse = (std::min(in_shape1.num_dimensions(), in_shape2.num_dimensions()) > Window::DimZ);
        for(size_t d = Window::DimZ; can_collapse && (d < out_shape.num_dimensions()); ++d)
        {
            can_collapse = (in_shape1[d] == in_shape2[d]);
        }
    }

    bool   has_collapsed = false;
    Window collapsed     = can_collapse ? window.collapse_if_possible(ICLKernel::window(), Window::DimZ, &has_collapsed) : window;

    const TensorShape &in_shape1_collapsed = has_collapsed ? in_shape1.collapsed_from(Window::DimZ) : in_shape1;
    const TensorShape &in_shape2_collapsed = has_collapsed ? in_shape2.collapsed_from(Window::DimZ) : in_shape2;

    Window slice        = collapsed.first_slice_window_3D();
    Window slice_input1 = slice.broadcast_if_dimension_le_one(in_shape1_collapsed);
    Window slice_input2 = slice.broadcast_if_dimension_le_one(in_shape2_collapsed);

    do
    {
        unsigned int idx = 0;
        add_3D_tensor_argument(idx, _input1, slice_input1);
        add_3D_tensor_argument(idx, _input2, slice_input2);
        add_3D_tensor_argument(idx, _output, slice);
        enqueue(queue, *this, slice);

        ARM_COMPUTE_UNUSED(collapsed.slide_window_slice_3D(slice_input1));
        ARM_COMPUTE_UNUSED(collapsed.slide_window_slice_3D(slice_input2));
    }
    while(collapsed.slide_window_slice_3D(slice));
}

BorderSize CLPixelWiseMultiplicationKernel::border_size() const
{
    const unsigned int replicateSize = _output->info()->dimension(0) - std::min(_input1->info()->dimension(0), _input2->info()->dimension(0));
    const unsigned int border        = std::min<unsigned int>(num_elems_processed_per_iteration - 1U, replicateSize);
    return BorderSize{ 0, border, 0, 0 };
}

namespace
{
constexpr unsigned int num_elems_processed_per_iteration_complex = 1;

Status validate_arguments_complex(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output)
{
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input1, 2, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input2, 2, DataType::F32);

    const TensorShape &out_shape = TensorShape::broadcast_shape(input1->tensor_shape(), input2->tensor_shape());

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(out_shape.total_size() == 0, "Inputs are not broadcast compatible");

    // Validate in case of configured output
    if(output->total_size() > 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 2, DataType::F32);
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(detail::have_different_dimensions(out_shape, output->tensor_shape(), 0), "Wrong shape for output");
    }

    return Status{};
}

std::pair<Status, Window> validate_and_configure_window_complex(ITensorInfo *input1, ITensorInfo *input2, ITensorInfo *output)
{
    const std::pair<TensorShape, ValidRegion> broadcast_pair = ITensorInfo::broadcast_shape_and_valid_region(*input1, *input2);
    const TensorShape &out_shape    = broadcast_pair.first;
    const ValidRegion &valid_region = broadcast_pair.second;

    // Auto initialize output if not initialized
    const TensorInfo out_info(out_shape, input1->num_channels(), input1->data_type());
    auto_init_if_empty(*output, out_info);

    Window win        = calculate_max_window(valid_region, Steps(num_elems_processed_per_iteration_complex));
    Window win_input1 = win.broadcast_if_dimension_le_one(*input1);
    Window win_input2 = win.broadcast_if_dimension_le_one(*input2);

    AccessWindowHorizontal input1_access(input1, 0, num_elems_processed_per_iteration_complex);
    AccessWindowHorizontal input2_access(input2, 0, num_elems_processed_per_iteration_complex);
    AccessWindowHorizontal output_access(output, 0, num_elems_processed_per_iteration_complex);

    bool window_changed = update_window_and_padding(win_input1, input1_access)
                          || update_window_and_padding(win_input2, input2_access)
                          || update_window_and_padding(win, output_access);

    output_access.set_valid_region(win, valid_region);

    Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
    return std::make_pair(err, win);
}
} // namespace

CLComplexPixelWiseMultiplicationKernel::CLComplexPixelWiseMultiplicationKernel()
    : _input1(nullptr), _input2(nullptr), _output(nullptr)
{
}

void CLComplexPixelWiseMultiplicationKernel::configure(const ICLTensor *input1, const ICLTensor *input2, ICLTensor *output)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input1, input2, output);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments_complex(input1->info(), input2->info(), output->info()));

    // Configure kernel window
    auto win_config = validate_and_configure_window_complex(input1->info(), input2->info(), output->info());
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);

    _input1 = input1;
    _input2 = input2;
    _output = output;

    // Create kernel
    _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel("pixelwise_mul_complex"));

    ICLKernel::configure_internal(win_config.second);
}

Status CLComplexPixelWiseMultiplicationKernel::validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input1, input2, output);
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments_complex(input1, input2, output));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window_complex(input1->clone().get(), input2->clone().get(), output->clone().get()).first);

    return Status{};
}

void CLComplexPixelWiseMultiplicationKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

    const TensorShape &in_shape1 = _input1->info()->tensor_shape();
    const TensorShape &in_shape2 = _input2->info()->tensor_shape();
    const TensorShape &out_shape = _output->info()->tensor_shape();

    bool can_collapse = true;
    if(std::min(in_shape1.total_size(), in_shape2.total_size()) > 1)
    {
        can_collapse = (std::min(in_shape1.num_dimensions(), in_shape2.num_dimensions()) > Window::DimZ);
        for(size_t d = Window::DimZ; can_collapse && (d < out_shape.num_dimensions()); ++d)
        {
            can_collapse = (in_shape1[d] == in_shape2[d]);
        }
    }

    bool   has_collapsed = false;
    Window collapsed     = can_collapse ? window.collapse_if_possible(ICLKernel::window(), Window::DimZ, &has_collapsed) : window;

    const TensorShape &in_shape1_collapsed = has_collapsed ? in_shape1.collapsed_from(Window::DimZ) : in_shape1;
    const TensorShape &in_shape2_collapsed = has_collapsed ? in_shape2.collapsed_from(Window::DimZ) : in_shape2;

    Window slice        = collapsed.first_slice_window_3D();
    Window slice_input1 = slice.broadcast_if_dimension_le_one(in_shape1_collapsed);
    Window slice_input2 = slice.broadcast_if_dimension_le_one(in_shape2_collapsed);

    do
    {
        unsigned int idx = 0;
        add_3D_tensor_argument(idx, _input1, slice_input1);
        add_3D_tensor_argument(idx, _input2, slice_input2);
        add_3D_tensor_argument(idx, _output, slice);
        enqueue(queue, *this, slice);

        ARM_COMPUTE_UNUSED(collapsed.slide_window_slice_3D(slice_input1));
        ARM_COMPUTE_UNUSED(collapsed.slide_window_slice_3D(slice_input2));
    }
    while(collapsed.slide_window_slice_3D(slice));
}

BorderSize CLComplexPixelWiseMultiplicationKernel::border_size() const
{
    const unsigned int replicateSize = _output->info()->dimension(0) - std::min(_input1->info()->dimension(0), _input2->info()->dimension(0));
    const unsigned int border        = std::min<unsigned int>(num_elems_processed_per_iteration_complex - 1U, replicateSize);
    return BorderSize{ 0, border, 0, 0 };
}
} // namespace arm_compute
