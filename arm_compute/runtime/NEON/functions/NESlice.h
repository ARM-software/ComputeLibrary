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
#ifndef ARM_COMPUTE_NE_SLICE_H
#define ARM_COMPUTE_NE_SLICE_H

#include "arm_compute/runtime/IFunction.h"
#include "arm_compute/runtime/NEON/INEOperator.h"

namespace arm_compute
{
// Forward Declarations
class ITensor;

/** Basic function to perform tensor slicing */
class NESlice : public IFunction
{
public:
    /** Default Constructor */
    NESlice();
    /** Default Destructor */
    ~NESlice();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NESlice(const NESlice &) = delete;
    /** Default move constructor */
    NESlice(NESlice &&);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NESlice &operator=(const NESlice &) = delete;
    /** Default move assignment operator */
    NESlice &operator=(NESlice &&);

    /** Configure kernel
     *
     * Valid data layouts:
     * - All
     *
     * Valid data type configurations:
     * |src    |dst    |
     * |:------|:------|
     * |All    |All    |
     *
     * @note Supported tensor rank: up to 4
     * @note Start indices must be non-negative. 0 <= starts[i]
     * @note End coordinates can be negative, which represents the number of elements before the end of that dimension.
     * @note End indices are not inclusive unless negative.
     *
     * @param[in]  input  Source tensor. Data type supported: All
     * @param[out] output Destination tensor. Data type supported: Same as @p input
     * @param[in]  starts The starts of the dimensions of the input tensor to be sliced. The length must be of rank(input).
     * @param[in]  ends   The ends of the dimensions of the input tensor to be sliced. The length must be of rank(input).
     */
    void configure(const ITensor *input, ITensor *output, const Coordinates &starts, const Coordinates &ends);

    /** Static function to check if given info will lead to a valid configuration of @ref NESlice
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

    // Inherited methods overridden:
    void run() override;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

namespace experimental
{
/** Basic function to perform tensor slicing */
class NESlice : public INEOperator
{
public:
    /** Configure kernel
     *
     * @note Supported tensor rank: up to 4
     * @note Start indices must be non-negative. 0 <= starts[i]
     * @note End coordinates can be negative, which represents the number of elements before the end of that dimension.
     * @note End indices are not inclusive unless negative.
     *
     * @param[in]  input  Source tensor info. Data type supported: All
     * @param[out] output Destination tensor info. Data type supported: Same as @p input
     * @param[in]  starts The starts of the dimensions of the input tensor to be sliced. The length must be of rank(input).
     * @param[in]  ends   The ends of the dimensions of the input tensor to be sliced. The length must be of rank(input).
     */
    void configure(const ITensorInfo *input, ITensorInfo *output, const Coordinates &starts, const Coordinates &ends);

    /** Static function to check if given info will lead to a valid configuration of @ref NESlice
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
#endif /* ARM_COMPUTE_NE_SLICE_H */
