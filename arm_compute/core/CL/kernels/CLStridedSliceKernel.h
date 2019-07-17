/*
 * Copyright (c) 2018-2019 ARM Limited.
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
#ifndef __ARM_COMPUTE_CL_STRIDED_SLICE_KERNEL_H__
#define __ARM_COMPUTE_CL_STRIDED_SLICE_KERNEL_H__

#include "arm_compute/core/CL/ICLKernel.h"
#include "arm_compute/core/Types.h"

#include <cstdint>

namespace arm_compute
{
// Forward declarations
class ICLTensor;

/** Interface for the kernel to perform tensor strided slicing */
class CLStridedSliceKernel : public ICLKernel
{
public:
    /** Default constructor */
    CLStridedSliceKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLStridedSliceKernel(const CLStridedSliceKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLStridedSliceKernel &operator=(const CLStridedSliceKernel &) = delete;
    /** Allow instances of this class to be moved */
    CLStridedSliceKernel(CLStridedSliceKernel &&) = default;
    /** Allow instances of this class to be moved */
    CLStridedSliceKernel &operator=(CLStridedSliceKernel &&) = default;
    /** Default destructor */
    ~CLStridedSliceKernel() = default;
    /** Configure kernel
     *
     * @note Supported tensor rank: up to 4
     *
     * @param[in]  input            Source tensor. Data type supported: U8/S8/QASYMM8/U16/S16/QSYMM16/U32/S32/F16/F32
     * @param[out] output           Destination tensor. Data type supported: Same as @p input
     * @param[in]  starts           The starts of the dimensions of the input tensor to be sliced. The length must be of rank(input).
     * @param[in]  ends             The ends of the dimensions of the input tensor to be sliced. The length must be of rank(input).
     * @param[in]  strides          The strides of the dimensions of the input tensor to be sliced. The length must be of rank(input).
     * @param[in]  begin_mask       If the ith bit of begin_mask is set, starts[i] is ignored and the fullest possible range in that dimension is used instead.
     * @param[in]  end_mask         If the ith bit of end_mask is set, ends[i] is ignored and the fullest possible range in that dimension is used instead.
     * @param[in]  shrink_axis_mask If the ith bit of shrink_axis_mask is set, it implies that the ith specification shrinks the dimensionality by 1.
     *                              A slice of size 1 starting from starts[i] in the dimension must be preserved.
     */
    void configure(const ICLTensor *input, ICLTensor *output,
                   const Coordinates &starts, const Coordinates &ends, const BiStrides &strides,
                   int32_t begin_mask, int32_t end_mask, int32_t shrink_axis_mask);

    /** Static function to check if given info will lead to a valid configuration of @ref CLStridedSliceKernel
     *
     * @note Supported tensor rank: up to 4
     *
     * @param[in] input            Source tensor. Data type supported: U8/S8/QASYMM8/U16/S16/QSYMM16/U32/S32/F16/F32
     * @param[in] output           Destination tensor. Data type supported: Same as @p input
     * @param[in] starts           The starts of the dimensions of the input tensor to be sliced. The length must be of rank(input).
     * @param[in] ends             The ends of the dimensions of the input tensor to be sliced. The length must be of rank(input).
     * @param[in] strides          The strides of the dimensions of the input tensor to be sliced. The length must be of rank(input).
     * @param[in] begin_mask       If the ith bit of begin_mask is set, starts[i] is ignored and the fullest possible range in that dimension is used instead.
     * @param[in] end_mask         If the ith bit of end_mask is set, ends[i] is ignored and the fullest possible range in that dimension is used instead.
     * @param[in] shrink_axis_mask If the ith bit of shrink_axis_mask is set, it implies that the ith specification shrinks the dimensionality by 1.
     *                             A slice of size 1 starting from starts[i] in the dimension must be preserved.
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output,
                           const Coordinates &starts, const Coordinates &ends, const BiStrides &strides,
                           int32_t begin_mask, int32_t end_mask, int32_t shrink_axis_mask);

    // Inherited methods overridden:
    void run(const Window &window, cl::CommandQueue &queue) override;

private:
    const ICLTensor *_input;  /**< Source tensor */
    ICLTensor       *_output; /**< Destination tensor */
};
} // namespace arm_compute
#endif /*__ARM_COMPUTE_CL_STRIDED_SLICE_KERNEL_H__ */
