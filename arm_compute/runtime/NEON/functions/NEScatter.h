/*
 * Copyright (c) 2024 Arm Limited.
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
#ifndef ACL_ARM_COMPUTE_RUNTIME_NEON_FUNCTIONS_NESCATTER_H
#define ACL_ARM_COMPUTE_RUNTIME_NEON_FUNCTIONS_NESCATTER_H

#include "arm_compute/core/Error.h"
#include "arm_compute/runtime/IFunction.h"

#include <memory>

namespace arm_compute
{
class ITensor;
class ITensorInfo;
struct ScatterInfo;

/** Function to compute ScatterND Layer */
class NEScatter : public IFunction
{
public:
    /** Default Constructor */
    NEScatter();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEScatter(const NEScatter &) = delete;
    /** Default move constructor */
    NEScatter(NEScatter &&);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEScatter &operator=(const NEScatter &) = delete;
    /** Default move assignment operator */
    NEScatter &operator=(NEScatter &&);
    /** Default destructor */
    ~NEScatter();
    /** Initialise the kernel's inputs and outputs
     *
     * Valid data layouts:
     * - All
     *
     *
     * @param[in]  src     Source tensor. Values used to fill output. Can be nullptr when zero initialization is true.
     * @param[in]  updates Tensor containing values used to update output tensor. Data types supported: same as @p src
     * @param[in]  indices Tensor containing Indices to change in the output Tensor. Data types supported : U32
     * @param[out] output  Destination tensor. Data types supported: same as @p src.
     * @param[in]  info    Scatter info object.
     */
    void configure(
        const ITensor *src, const ITensor *updates, const ITensor *indices, ITensor *output, const ScatterInfo &info);
    /** Static function to check if given info will lead to a valid configuration of @ref CLScatter
     *
     * @param[in] src     Source tensor.
     * @param[in] updates Tensor containing values used for updating the output Tensor. Data types supported : same as @p src
     * @param[in] indices Tensor containing Indices to change in the output Tensor. Data types supported : U32
     * @param[in] output  Destination tensor. Data types supported: same as @p src.
     * @param[in] info    Scatter info containing type of scatter.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *src,
                           const ITensorInfo *updates,
                           const ITensorInfo *indices,
                           const ITensorInfo *output,
                           const ScatterInfo &info);

    // Inherited methods overridden:
    void run() override;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};
} // namespace arm_compute
#endif // ACL_ARM_COMPUTE_RUNTIME_NEON_FUNCTIONS_NESCATTER_H
