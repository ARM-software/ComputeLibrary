/*
 * Copyright (c) 2018-2021 Arm Limited.
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
#ifndef ARM_COMPUTE_CL_STRIDED_SLICE_H
#define ARM_COMPUTE_CL_STRIDED_SLICE_H

#include "arm_compute/runtime/CL/CLRuntimeContext.h"
#include "arm_compute/runtime/CL/ICLOperator.h"
#include "arm_compute/runtime/IFunction.h"

namespace arm_compute
{
// Forward Declarations
class ICLTensor;

/** Basic function to run @ref CLStridedSliceKernel */
class CLStridedSlice : public IFunction
{
public:
    /** Constructor
     *
     * @param[in] ctx Runtime context to be used by the function
     */
    CLStridedSlice(CLRuntimeContext *ctx = nullptr);
    /** Destructor */
    ~CLStridedSlice();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLStridedSlice(const CLStridedSlice &) = delete;
    /** Default move constructor */
    CLStridedSlice(CLStridedSlice &&);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLStridedSlice &operator=(const CLStridedSlice &) = delete;
    /** Default move assignment operator */
    CLStridedSlice &operator=(CLStridedSlice &&);
    /** Configure kernel
     *
     * Valid data layouts:
     * - All
     *
     * Valid data type configurations:
     * |src            |dst            |
     * |:--------------|:--------------|
     * |All            |All            |
     *
     * @note Supported tensor rank: up to 4
     *
     * @param[in]  input            Source tensor. Data type supported: All.
     * @param[out] output           Destination tensor. Data type supported: Same as @p input
     * @param[in]  starts           The starts of the dimensions of the input tensor to be sliced. The length must be of rank(input).
     * @param[in]  ends             The ends of the dimensions of the input tensor to be sliced. The length must be of rank(input).
     * @param[in]  strides          The strides of the dimensions of the input tensor to be sliced. The length must be of rank(input).
     * @param[in]  begin_mask       (Optional) If the ith bit of begin_mask is set, starts[i] is ignored and the fullest possible range in that dimension is used instead.
     * @param[in]  end_mask         (Optional) If the ith bit of end_mask is set, ends[i] is ignored and the fullest possible range in that dimension is used instead.
     * @param[in]  shrink_axis_mask (Optional) If the ith bit of shrink_axis_mask is set, it implies that the ith specification shrinks the dimensionality by 1.
     *                              A slice of size 1 starting from starts[i] in the dimension must be preserved.
     */
    void configure(const ICLTensor *input, ICLTensor *output,
                   const Coordinates &starts, const Coordinates &ends, const BiStrides &strides,
                   int32_t begin_mask = 0, int32_t end_mask = 0, int32_t shrink_axis_mask = 0);
    /** Configure kernel
     *
     * @note Supported tensor rank: up to 4
     *
     * @param[in]  compile_context  The compile context to be used.
     * @param[in]  input            Source tensor. Data type supported: All.
     * @param[out] output           Destination tensor. Data type supported: Same as @p input
     * @param[in]  starts           The starts of the dimensions of the input tensor to be sliced. The length must be of rank(input).
     * @param[in]  ends             The ends of the dimensions of the input tensor to be sliced. The length must be of rank(input).
     * @param[in]  strides          The strides of the dimensions of the input tensor to be sliced. The length must be of rank(input).
     * @param[in]  begin_mask       (Optional) If the ith bit of begin_mask is set, starts[i] is ignored and the fullest possible range in that dimension is used instead.
     * @param[in]  end_mask         (Optional) If the ith bit of end_mask is set, ends[i] is ignored and the fullest possible range in that dimension is used instead.
     * @param[in]  shrink_axis_mask (Optional) If the ith bit of shrink_axis_mask is set, it implies that the ith specification shrinks the dimensionality by 1.
     *                              A slice of size 1 starting from starts[i] in the dimension must be preserved.
     */
    void configure(const CLCompileContext &compile_context, const ICLTensor *input, ICLTensor *output,
                   const Coordinates &starts, const Coordinates &ends, const BiStrides &strides,
                   int32_t begin_mask = 0, int32_t end_mask = 0, int32_t shrink_axis_mask = 0);

    /** Static function to check if given info will lead to a valid configuration of @ref CLStridedSlice
     *
     * @note Supported tensor rank: up to 4
     *
     * @param[in] input            Source tensor. Data type supported: All.
     * @param[in] output           Destination tensor. Data type supported: Same as @p input
     * @param[in] starts           The starts of the dimensions of the input tensor to be sliced. The length must be of rank(input).
     * @param[in] ends             The ends of the dimensions of the input tensor to be sliced. The length must be of rank(input).
     * @param[in] strides          The strides of the dimensions of the input tensor to be sliced. The length must be of rank(input).
     * @param[in] begin_mask       (Optional) If the ith bit of begin_mask is set, starts[i] is ignored and the fullest possible range in that dimension is used instead.
     * @param[in] end_mask         (Optional) If the ith bit of end_mask is set, ends[i] is ignored and the fullest possible range in that dimension is used instead.
     * @param[in] shrink_axis_mask (Optional) If the ith bit of shrink_axis_mask is set, it implies that the ith specification shrinks the dimensionality by 1.
     *                             A slice of size 1 starting from starts[i] in the dimension must be preserved.
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output,
                           const Coordinates &starts, const Coordinates &ends, const BiStrides &strides,
                           int32_t begin_mask = 0, int32_t end_mask = 0, int32_t shrink_axis_mask = 0);

    // Inherited methods overridden:
    void run() override;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

namespace experimental
{
/** Basic function to run @ref CLStridedSliceKernel */
class CLStridedSlice : public ICLOperator
{
public:
    /** Configure kernel
     *
     * @note Supported tensor rank: up to 4
     *
     * @param[in]  compile_context  The compile context to be used.
     * @param[in]  input            Source tensor info. Data type supported: All.
     * @param[out] output           Destination tensor info. Data type supported: Same as @p input
     * @param[in]  starts           The starts of the dimensions of the input tensor to be sliced. The length must be of rank(input).
     * @param[in]  ends             The ends of the dimensions of the input tensor to be sliced. The length must be of rank(input).
     * @param[in]  strides          The strides of the dimensions of the input tensor to be sliced. The length must be of rank(input).
     * @param[in]  begin_mask       (Optional) If the ith bit of begin_mask is set, starts[i] is ignored and the fullest possible range in that dimension is used instead.
     * @param[in]  end_mask         (Optional) If the ith bit of end_mask is set, ends[i] is ignored and the fullest possible range in that dimension is used instead.
     * @param[in]  shrink_axis_mask (Optional) If the ith bit of shrink_axis_mask is set, it implies that the ith specification shrinks the dimensionality by 1.
     *                              A slice of size 1 starting from starts[i] in the dimension must be preserved.
     */
    void configure(const CLCompileContext &compile_context, const ITensorInfo *input, ITensorInfo *output,
                   const Coordinates &starts, const Coordinates &ends, const BiStrides &strides,
                   int32_t begin_mask = 0, int32_t end_mask = 0, int32_t shrink_axis_mask = 0);

    /** Static function to check if given info will lead to a valid configuration of @ref CLStridedSlice
     *
     * @note Supported tensor rank: up to 4
     *
     * @param[in] input            Source tensor info. Data type supported: All.
     * @param[in] output           Destination tensor info. Data type supported: Same as @p input
     * @param[in] starts           The starts of the dimensions of the input tensor to be sliced. The length must be of rank(input).
     * @param[in] ends             The ends of the dimensions of the input tensor to be sliced. The length must be of rank(input).
     * @param[in] strides          The strides of the dimensions of the input tensor to be sliced. The length must be of rank(input).
     * @param[in] begin_mask       (Optional) If the ith bit of begin_mask is set, starts[i] is ignored and the fullest possible range in that dimension is used instead.
     * @param[in] end_mask         (Optional) If the ith bit of end_mask is set, ends[i] is ignored and the fullest possible range in that dimension is used instead.
     * @param[in] shrink_axis_mask (Optional) If the ith bit of shrink_axis_mask is set, it implies that the ith specification shrinks the dimensionality by 1.
     *                             A slice of size 1 starting from starts[i] in the dimension must be preserved.
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output,
                           const Coordinates &starts, const Coordinates &ends, const BiStrides &strides,
                           int32_t begin_mask = 0, int32_t end_mask = 0, int32_t shrink_axis_mask = 0);
};
} // namespace experimental
} // namespace arm_compute
#endif /* ARM_COMPUTE_CL_STRIDED_SLICE_H */
