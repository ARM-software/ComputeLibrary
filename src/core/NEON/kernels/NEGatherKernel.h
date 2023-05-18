/*
 * Copyright (c) 2019-2023 Arm Limited.
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

#ifndef ARM_COMPUTE_NEGATHERKERNEL_H
#define ARM_COMPUTE_NEGATHERKERNEL_H

#include "arm_compute/core/Types.h"
#include "src/core/NEON/INEKernel.h"

namespace arm_compute
{
// Forward declarations
class ITensor;

/** Kernel to perform gather operation. */
class NEGatherKernel : public INEKernel
{
public:
    /** Default constructor. */
    NEGatherKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers). */
    NEGatherKernel(const NEGatherKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers). */
    NEGatherKernel &operator=(const NEGatherKernel &) = delete;
    /** Allow instances of this class to be moved. */
    NEGatherKernel(NEGatherKernel &&) = default;
    /** Allow instances of this class to be moved. */
    NEGatherKernel &operator=(NEGatherKernel &&) = default;
    /** Default detructor */
    ~NEGatherKernel() = default;

    /** Name of the kernel
     *
     * @return Kernel name
     */
    const char *name() const override
    {
        return "NEGatherKernel";
    }
    /** Initialise the kernel's inputs and outputs
     *
     * @param[in]  input   Source tensor. Supported tensor rank: up to 4. Data type supported: All
     * @param[in]  indices Indices tensor. Supported tensor rank: up to 3. Must be one of the following type: U32/S32. Each value Must be in range [0, input.shape[@p axis])
     *                     @note 2D or 3D indices are only supported for the axis 1.
     * @param[out] output  Destination tensor. Data type supported: Same as @p input
     * @param[in]  axis    (Optional) The axis in @p input to gather @p indices from. Negative values wrap around. Defaults to 0.
     *
     */
    void configure(const ITensor *input, const ITensor *indices, ITensor *output, int axis = 0);

    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to @ref NEGatherKernel::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *indices, const ITensorInfo *output, int axis);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;

private:
    template <typename TIndex>
    void gather_common(const Window &window, const ThreadInfo &info);

    using kernel_ptr = void (NEGatherKernel::*)(const Window &window, const ThreadInfo &info);

    const ITensor *_input;
    const ITensor *_indices;
    int            _axis;
    ITensor       *_output;
    kernel_ptr     _func;

    Strides        _src_it_strides;
    Strides        _idx_it_strides;
};
} // namespace arm_compute
#endif /* ARM_COMPUTE_NEGATHERKERNEL_H */
