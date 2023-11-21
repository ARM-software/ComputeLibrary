/*
 * Copyright (c) 2018-2021, 2023 Arm Limited.
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
#ifndef ACL_ARM_COMPUTE_RUNTIME_NEON_FUNCTIONS_NESTACKLAYER_H
#define ACL_ARM_COMPUTE_RUNTIME_NEON_FUNCTIONS_NESTACKLAYER_H

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/IFunction.h"

#include <memory>
#include <vector>

namespace arm_compute
{
class ITensor;
class ITensorInfo;
class NEStackLayerKernel;

/** Basic function to stack tensors along an axis. This function calls the following kernel:
 *
 * -# @ref NEStackLayerKernel
 *
 */
class NEStackLayer : public IFunction
{
public:
    /** Default constructor */
    NEStackLayer();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEStackLayer(const NEStackLayer &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEStackLayer &operator=(const NEStackLayer &) = delete;
    /** Prevent instances of this class from being moved (As this class contains non movable objects) */
    NEStackLayer(NEStackLayer &&) = delete;
    /** Prevent instances of this class from being moved (As this class contains non movable objects) */
    NEStackLayer &operator=(NEStackLayer &&) = delete;
    /** Default destructor */
    ~NEStackLayer();
    /** Initialise the kernel's inputs vector and output.
     *
     * Valid data layouts:
     * - All
     *
     * Valid data type configurations:
     * |src            |dst            |
     * |:--------------|:--------------|
     * |All            |All            |
     *
     * @note Supported input tensor rank: up to 4
     *
     * @param[in]  input  The vectors containing all the tensors with the same shape to stack. Data types supported: All
     * @param[in]  axis   The dimension to stack the tensors along. It must be smaller than the number of input dimensions.
     *                    Negative values wrap around
     * @param[out] output Output tensor. Data types supported: Same as @p input.
     */
    void configure(const std::vector<ITensor *> &input, int axis, ITensor *output);
    /** Static function to check if given info will lead to a valid configuration of @ref NEStackLayerKernel
     *
     * @note Supported input tensor rank: up to 4
     *
     * @param[in] input  The vectors containing all the tensors info with the same shape to stack. Data types supported: All
     * @param[in] axis   The dimension to stack the tensors along. It must be smaller than the number of input dimensions.
     *                   Negative values wrap around
     * @param[in] output Output tensor info. Data types supported: Same as @p input.
     *
     * @return a status
     */
    static Status validate(const std::vector<ITensorInfo *> &input, int axis, const ITensorInfo *output);

    // Inherited methods overridden:
    void run() override;

private:
    std::unique_ptr<NEStackLayerKernel> _stack_kernel;
    bool                                _is_prepared;
};
} // namespace arm_compute
#endif // ACL_ARM_COMPUTE_RUNTIME_NEON_FUNCTIONS_NESTACKLAYER_H
