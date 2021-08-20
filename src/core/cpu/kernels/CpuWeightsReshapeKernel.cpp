/*
 * Copyright (c) 2017-2021 Arm Limited.
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
#include "src/core/cpu/kernels/CpuWeightsReshapeKernel.h"

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Validate.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"

namespace arm_compute
{
namespace cpu
{
namespace kernels
{
namespace
{
TensorShape get_output_shape(const ITensorInfo *src, bool has_bias)
{
    TensorShape output_shape{ src->tensor_shape() };

    output_shape.collapse(3);
    const size_t tmp_dim = output_shape[0];
    output_shape.set(0, output_shape[1]);
    output_shape.set(1, tmp_dim + (has_bias ? 1 : 0));

    return output_shape;
}

Status validate_arguments(const ITensorInfo *src, const ITensorInfo *biases, const ITensorInfo *dst)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(src, dst);
    //Note: ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(src) is not needed here as this kernel doesn't use CPU FP16 instructions.
    ARM_COMPUTE_RETURN_ERROR_ON(src->data_type() == DataType::UNKNOWN);

    if(biases != nullptr)
    {
        ARM_COMPUTE_RETURN_ERROR_ON(is_data_type_quantized_asymmetric(src->data_type()));
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src, biases);
        ARM_COMPUTE_RETURN_ERROR_ON((src->num_dimensions() == 4) && (biases->num_dimensions() != 1));
        ARM_COMPUTE_RETURN_ERROR_ON((src->num_dimensions() == 5) && (biases->num_dimensions() != 2));
        ARM_COMPUTE_RETURN_ERROR_ON((src->num_dimensions() == 4) && (biases->dimension(0) != src->tensor_shape()[3]));
        ARM_COMPUTE_RETURN_ERROR_ON((src->num_dimensions() == 5) && (biases->dimension(0) != src->tensor_shape()[3] || biases->dimension(1) != src->tensor_shape()[4]));
    }

    // Checks performed when output is configured
    if(dst->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(dst->tensor_shape(), get_output_shape(src, biases != nullptr));
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src, dst);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_QUANTIZATION_INFO(src, dst);
    }

    return Status{};
}
} // namespace

void CpuWeightsReshapeKernel::configure(const ITensorInfo *src, const ITensorInfo *biases, ITensorInfo *dst)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src, dst);

    // Output tensor auto inizialitation if not yet initialized
    auto_init_if_empty(*dst, src->clone()->set_tensor_shape(get_output_shape(src, (biases != nullptr))));

    // Perform validation step
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(src,
                                                  biases,
                                                  dst));

    // Configure kernel
    Window window = calculate_max_window(*src, Steps());
    window.set(Window::DimX, Window::Dimension(0, src->dimension(0), src->dimension(0)));
    window.set(Window::DimY, Window::Dimension(0, src->dimension(1), src->dimension(1)));
    window.set(Window::DimZ, Window::Dimension(0, src->dimension(2), src->dimension(2)));
    ICpuKernel::configure(window);
}

Status CpuWeightsReshapeKernel::validate(const ITensorInfo *src, const ITensorInfo *biases, const ITensorInfo *dst)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(src, biases, dst));
    return Status{};
}

void CpuWeightsReshapeKernel::run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICpuKernel::window(), window);

    auto src    = tensors.get_const_tensor(TensorType::ACL_SRC);
    auto biases = tensors.get_const_tensor(TensorType::ACL_BIAS);
    auto dst    = tensors.get_tensor(TensorType::ACL_DST);

    const unsigned int kernel_size_x   = src->info()->dimension(0);
    const unsigned int kernel_size_y   = src->info()->dimension(1);
    const unsigned int kernel_depth    = src->info()->dimension(2);
    const unsigned int input_stride_x  = src->info()->strides_in_bytes().x();
    const unsigned int input_stride_y  = src->info()->strides_in_bytes().y();
    const unsigned int input_stride_z  = src->info()->strides_in_bytes().z();
    const unsigned int output_stride_y = dst->info()->strides_in_bytes().y();

    // Create iterators
    Iterator in(src, window);
    execute_window_loop(window, [&](const Coordinates & id)
    {
        // Get column index
        const int kernel_idx = id[3];
        const int kernel_idz = id[4];

        // Setup pointers
        const uint8_t *tmp_input_ptr        = in.ptr();
        uint8_t       *tmp_output_ptr       = dst->ptr_to_element(Coordinates(kernel_idx, 0, kernel_idz));
        const uint8_t *curr_input_row_ptr   = tmp_input_ptr;
        const uint8_t *curr_input_depth_ptr = tmp_input_ptr;

        // Linearize volume
        for(unsigned int d = 0; d < kernel_depth; ++d)
        {
            for(unsigned int j = 0; j < kernel_size_y; ++j)
            {
                for(unsigned int i = 0; i < kernel_size_x; ++i)
                {
                    std::memcpy(tmp_output_ptr, tmp_input_ptr, src->info()->element_size());
                    tmp_input_ptr += input_stride_x;
                    tmp_output_ptr += output_stride_y;
                }
                curr_input_row_ptr += input_stride_y;
                tmp_input_ptr = curr_input_row_ptr;
            }
            curr_input_depth_ptr += input_stride_z;
            curr_input_row_ptr = curr_input_depth_ptr;
            tmp_input_ptr      = curr_input_depth_ptr;
        }

        // Add bias
        if(biases != nullptr)
        {
            std::memcpy(tmp_output_ptr, biases->ptr_to_element(Coordinates(kernel_idx, kernel_idz)), src->info()->element_size());
        }
    },
    in);
}
const char *CpuWeightsReshapeKernel::name() const
{
    return "CpuWeightsReshapeKernel";
}
} // namespace kernels
} // namespace cpu
} // namespace arm_compute