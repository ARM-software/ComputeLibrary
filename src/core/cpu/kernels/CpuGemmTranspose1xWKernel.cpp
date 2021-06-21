/*
 * Copyright (c) 2016-2021 Arm Limited.
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
#include "src/core/cpu/kernels/CpuGemmTranspose1xWKernel.h"

#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"

#include <arm_neon.h>

namespace arm_compute
{
namespace cpu
{
namespace kernels
{
using namespace arm_compute::misc::shape_calculator;

void CpuGemmTranspose1xWKernel::configure(const ITensorInfo *src, ITensorInfo *dst)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src, dst);

    // Output tensor auto inizialitation if not yet initialized
    auto_init_if_empty(*dst, src->clone()->set_tensor_shape(compute_transpose1xW_with_element_size_shape(*src)));

    // Perform validate step
    ARM_COMPUTE_ERROR_THROW_ON(CpuGemmTranspose1xWKernel::validate(src, dst));

    const size_t vector_size = 16 / src->element_size();

    // Configure kernel window
    Window win = calculate_max_window(*src, Steps(vector_size));
    ICPPKernel::configure(win);
}

Status CpuGemmTranspose1xWKernel::validate(const ITensorInfo *src, const ITensorInfo *dst)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(src);
    ARM_COMPUTE_RETURN_ERROR_ON(src->data_type() == DataType::UNKNOWN);
    //Note: ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(src) is not needed here as this kernel doesn't use CPU FP16 instructions.

    if(dst->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(dst->tensor_shape(), compute_transpose1xW_with_element_size_shape(*src));
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src, dst);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_QUANTIZATION_INFO(src, dst);
    }

    return Status{};
}

void CpuGemmTranspose1xWKernel::run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);
    ARM_COMPUTE_ERROR_ON(tensors.empty());

    /*
     * Following an example of how the transposition1xW works when the src data type is F32
     *
     *         |a00 a01 a02 a03|
     *         |a10 a11 a12 a13|
     *         |a20 a21 a22 a23| = | a00 a01 a02 a03 || a10 a11 a12 a13 || a20 a21 a22 a23 || a30 a31 a32 a33 |
     *         |a30 a31 a32 a33|
     *
     * The dst matrix will have the following shape: [ height * W, ceil(width / W) ], where W = (16 / element size of the tensor)
     */

    // Set window for dst tensor. Set to 0 the X and Y dimensions in order to allow multi-threading implementation and future batched matrix multiplications
    Window win_out(window);
    win_out.set(Window::DimX, Window::Dimension(0, 0, 0));
    win_out.set(Window::DimY, Window::Dimension(0, 0, 0));

    const ITensor *src = tensors.get_const_tensor(TensorType::ACL_SRC);
    ITensor       *dst = tensors.get_tensor(TensorType::ACL_DST);

    Iterator in(src, window);
    Iterator out(dst, win_out);

    const size_t in_width     = src->info()->dimension(0);
    const size_t element_size = src->info()->element_size();
    const size_t out_stride   = dst->info()->strides_in_bytes()[1];
    const size_t vector_size  = 16 / element_size;

    execute_window_loop(window, [&](const Coordinates & id)
    {
        const uint8_t *in_ptr  = in.ptr();
        uint8_t *const out_ptr = out.ptr() + (id.y() * vector_size) * element_size + (id.x() / vector_size) * out_stride;

        for(size_t k = 0; k < vector_size; ++k)
        {
            // If the src width is not multiple of W, we fill the reference with 0s
            if((id.x() + k) >= in_width)
            {
                std::memset(out_ptr + k * element_size, 0, element_size);
            }
            else
            {
                std::memcpy(out_ptr + k * element_size, in_ptr + k * element_size, element_size);
            }
        }
    },
    in, out);
}

const char *CpuGemmTranspose1xWKernel::name() const
{
    return "CpuGemmTranspose1xWKernel";
}
} // namespace kernels
} // namespace cpu
} // namespace arm_compute
