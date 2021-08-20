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
#ifndef ARM_COMPUTE_CL_SLICE_H
#define ARM_COMPUTE_CL_SLICE_H

#include "arm_compute/runtime/CL/ICLOperator.h"
#include "arm_compute/runtime/IFunction.h"

namespace arm_compute
{
// Forward Declarations
class ICLTensor;
class CLCompileContext;
class ITensorInfo;

/** Basic function to perform tensor slicing */
class CLSlice : public IFunction
{
public:
    /** Default Constructor */
    CLSlice();
    /** Default Destructor */
    ~CLSlice();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLSlice(const CLSlice &) = delete;
    /** Default move constructor */
    CLSlice(CLSlice &&);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLSlice &operator=(const CLSlice &) = delete;
    /** Default move assignment operator */
    CLSlice &operator=(CLSlice &&);
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
     * @note Start indices must be non-negative. 0 <= starts[i]
     * @note End coordinates can be negative, which represents the number of elements before the end of that dimension.
     * @note End indices are not inclusive unless negative.
     *
     * @param[in]  input  Source tensor. Data type supported: All.
     * @param[out] output Destination tensor. Data type supported: Same as @p input
     * @param[in]  starts The starts of the dimensions of the input tensor to be sliced. The length must be of rank(input).
     * @param[in]  ends   The ends of the dimensions of the input tensor to be sliced. The length must be of rank(input).
     */
    void configure(const ICLTensor *input, ICLTensor *output, const Coordinates &starts, const Coordinates &ends);
    /** Configure kernel
     *
     * @note Supported tensor rank: up to 4
     * @note Start indices must be non-negative. 0 <= starts[i]
     * @note End coordinates can be negative, which represents the number of elements before the end of that dimension.
     * @note End indices are not inclusive unless negative.
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  input           Source tensor. Data type supported: All.
     * @param[out] output          Destination tensor. Data type supported: Same as @p input
     * @param[in]  starts          The starts of the dimensions of the input tensor to be sliced. The length must be of rank(input).
     * @param[in]  ends            The ends of the dimensions of the input tensor to be sliced. The length must be of rank(input).
     */
    void configure(const CLCompileContext &compile_context, const ICLTensor *input, ICLTensor *output, const Coordinates &starts, const Coordinates &ends);

    /** Static function to check if given info will lead to a valid configuration of @ref CLSlice
     *
     * @note Supported tensor rank: up to 4
     * @note Start indices must be non-negative. 0 <= starts[i]
     * @note End coordinates can be negative, which represents the number of elements before the end of that dimension.
     * @note End indices are not inclusive unless negative.
     *
     * @param[in] input  Source tensor info. Data type supported: All.
     * @param[in] output Destination tensor info. Data type supported: Same as @p input
     * @param[in] starts The starts of the dimensions of the input tensor to be sliced. The length must be of rank(input).
     * @param[in] ends   The ends of the dimensions of the input tensor to be sliced. The length must be of rank(input).
     *
     * @return A status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output, const Coordinates &starts, const Coordinates &ends);

    // Inherited methods overridden:
    void run() override;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

namespace experimental
{
/** Basic function to perform tensor slicing */
class CLSlice : public ICLOperator
{
public:
    /** Configure kernel
     *
     * @note Supported tensor rank: up to 4
     * @note Start indices must be non-negative. 0 <= starts[i]
     * @note End coordinates can be negative, which represents the number of elements before the end of that dimension.
     * @note End indices are not inclusive unless negative.
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  input           Source tensor info. Data type supported: All.
     * @param[out] output          Destination tensor info. Data type supported: Same as @p input
     * @param[in]  starts          The starts of the dimensions of the input tensor to be sliced. The length must be of rank(input).
     * @param[in]  ends            The ends of the dimensions of the input tensor to be sliced. The length must be of rank(input).
     */
    void configure(const CLCompileContext &compile_context, const ITensorInfo *input, ITensorInfo *output, const Coordinates &starts, const Coordinates &ends);

    /** Static function to check if given info will lead to a valid configuration of @ref CLSlice
     *
     * @note Supported tensor rank: up to 4
     * @note Start indices must be non-negative. 0 <= starts[i]
     * @note End coordinates can be negative, which represents the number of elements before the end of that dimension.
     * @note End indices are not inclusive unless negative.
     *
     * @param[in] input  Source tensor info. Data type supported: All
     * @param[in] output Destination tensor info. Data type supported: Same as @p input
     * @param[in] starts The starts of the dimensions of the input tensor to be sliced. The length must be of rank(input).
     * @param[in] ends   The ends of the dimensions of the input tensor to be sliced. The length must be of rank(input).
     *
     * @return A status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output, const Coordinates &starts, const Coordinates &ends);
};
} // namespace experimental
} // namespace arm_compute
#endif /* ARM_COMPUTE_CL_SLICE_H */
