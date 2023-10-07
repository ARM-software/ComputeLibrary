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

#ifndef ACL_SRC_CORE_NEON_KERNELS_NESTACKLAYERKERNEL_H
#define ACL_SRC_CORE_NEON_KERNELS_NESTACKLAYERKERNEL_H

#include "arm_compute/core/Types.h"

#include "src/core/NEON/INEKernel.h"

#include <cstdint>
#include <functional>

namespace arm_compute
{
class ITensor;

/** Basic kernel to stack a rank-R tensor into one with rank-(R+1) along the axis dimension. */
class NEStackLayerKernel : public INEKernel
{
public:
    const char *name() const override
    {
        return "NEStackLayerKernel";
    }
    /** Default constructor */
    NEStackLayerKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEStackLayerKernel(const NEStackLayerKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEStackLayerKernel &operator=(const NEStackLayerKernel &) = delete;
    /** Allow instances of this class to be moved */
    NEStackLayerKernel(NEStackLayerKernel &&) = default;
    /** Allow instances of this class to be moved */
    NEStackLayerKernel &operator=(NEStackLayerKernel &&) = default;
    /** Default destructor */
    ~NEStackLayerKernel() = default;
    /** Initialise the kernel's inputs and output
     *
     * @note Supported input tensor rank: up to 4
     *
     * @param[in]  input  Input tensors. Data types supported: All
     * @param[in]  axis   The dimension to stack the tensors along. It must be smaller than the number of input dimensions.
     * @param[out] output Output tensor. Data types supported: Same as @p input.
     *
     */
    void configure(const std::vector<ITensor *> &input, uint32_t axis, ITensor *output);
    /**  Static function to check if given info will lead to a valid configuration of @ref NEStackLayerKernel
     *
     * @note Supported input tensor rank: up to 4
     *
     * @param[in] input  Input tensor infos. Data types supported: All
     * @param[in] axis   The dimension to stack the tensors along. It must be smaller than the number of input dimensions.
     * @param[in] output Output tensor info. Data types supported: Same as @p input.
     *
     * @return a status
     */
    static Status validate(const std::vector<ITensorInfo *> &input, uint32_t axis, const ITensorInfo *output);

    /** Prepare the reshape kernel for execution (Only executed once) for
     *  choosing the window and the algorithm.
     */
    void prepare();

    // Inherited methods overridden
    void run(const Window &window, const ThreadInfo &info) override;

    /** Get the dimension to split the kernel workload
     *
     * @return the split dimension
     */
    uint32_t get_split_dimension() const
    {
        return _split_dimension;
    }

private:
    std::vector<ITensor *> _input;
    ITensor               *_output;
    uint32_t               _axis;
    uint32_t               _split_dimension;

    std::function<void(const std::vector<ITensor *> &, ITensor *, uint32_t, const Window &)> _stack_fn{};
};
} // namespace arm_compute
#endif // ACL_SRC_CORE_NEON_KERNELS_NESTACKLAYERKERNEL_H
