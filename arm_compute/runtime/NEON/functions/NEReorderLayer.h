/*
 * Copyright (c) 2023 Arm Limited.
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
#if defined(__aarch64__)

#ifndef ACL_ARM_COMPUTE_RUNTIME_NEON_FUNCTIONS_NEREORDERLAYER
#define ACL_ARM_COMPUTE_RUNTIME_NEON_FUNCTIONS_NEREORDERLAYER

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/IFunction.h"

namespace arm_compute
{
class ITensor;
class ITensorInfo;
class NEReorderKernel;
/** Function to compute blocked reorder. */
class NEReorderLayer : public IFunction
{
public:
    /** Default constructor */
    NEReorderLayer();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEReorderLayer(const NEReorderLayer &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEReorderLayer &operator=(const NEReorderLayer &) = delete;
    /** Prevent instances of this class from being moved (As this class contains non movable objects) */
    NEReorderLayer(NEReorderLayer &&) = delete;
    /** Prevent instances of this class from being moved (As this class contains non movable objects) */
    NEReorderLayer &operator=(NEReorderLayer &&) = delete;
    /** Default destructor */
    ~NEReorderLayer();
    /** Set the input and output tensors.
     *
     * Valid data layouts:
     * - NCHW
     *
     * Valid data type configurations:
     * |src      |dst       |
     * |:--------|:---------|
     * |F32      |F32       |
     *
     * @param[in]  input     Source tensor. Data type supported: F32. Data layouts supported: NCHW.
     * @param[out] output    Destination with the same dimensions, data type, data layout as  @p input
     *                       except last dimension of data layout which needs to be multiple of blocking parameter ksize
     * @param[in]  input_wf  WeightFormat of input.
     * @param[in]  output_wf WeightFormat of output.
     */
    void configure(const ITensor *input, ITensor *output, arm_compute::WeightFormat input_wf, arm_compute::WeightFormat output_wf);

    /** Static function to check if given info will lead to a valid configuration of @ref NEReorderLayer
     *
     * Similar to @ref NEReorderLayer::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output, arm_compute::WeightFormat input_wf, arm_compute::WeightFormat output_wf);

    // Inherited methods overridden:
    void run() override;

private:
    std::unique_ptr<NEReorderKernel> _reorder_kernel; /**< Reorder layer kernel */
};
} // namespace arm_compute
#endif /* ACL_ARM_COMPUTE_RUNTIME_NEON_FUNCTIONS_NEREORDERLAYER */

#endif  // defined(__aarch64__)