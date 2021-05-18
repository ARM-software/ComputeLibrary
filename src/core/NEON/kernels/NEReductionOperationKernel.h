/*
 * Copyright (c) 2017-2021 Arm Limited.
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
#ifndef ARM_COMPUTE_NEREDUCTIONOPERATIONKERNEL_H
#define ARM_COMPUTE_NEREDUCTIONOPERATIONKERNEL_H

#include "src/core/NEON/INEKernel.h"

namespace arm_compute
{
class ITensor;

/** Kernel to perform a reduction operation
 *
 * @note For ARG_MIN/ARG_MAX reduction, the default data type for an uninitialized
 *       output tensor is signed 32-bit integer (S32). It is the user's responsibility
 *       to check that the results do not overflow because the indices are computed
 *       in unsigned 32-bit (U32).
 */
class NEReductionOperationKernel : public INEKernel
{
public:
    const char *name() const override
    {
        return "NEReductionOperationKernel";
    }
    /** Default constructor */
    NEReductionOperationKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEReductionOperationKernel(const NEReductionOperationKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEReductionOperationKernel &operator=(const NEReductionOperationKernel &) = delete;
    /** Allow instances of this class to be moved */
    NEReductionOperationKernel(NEReductionOperationKernel &&) = default;
    /** Allow instances of this class to be moved */
    NEReductionOperationKernel &operator=(NEReductionOperationKernel &&) = default;
    /** Default destructor */
    ~NEReductionOperationKernel() = default;

    /** Set the source, destination of the kernel
     *
     * @param[in]  input  Source tensor. Data type supported: QASYMM8_SIGNED/QASYMM8/F16/F32/S32.
     * @param[out] output Destination tensor.Data types and data layouts supported: same as @p input, S32 for ARG_MIX/ARG_MAX.
     *                    Output will have the same number of dimensions as input.
     * @param[in]  axis   Axis along which to reduce. Supported reduction axis : 0
     * @param[in]  op     Reduction operation to perform.
     */
    void configure(const ITensor *input, ITensor *output, unsigned int axis, ReductionOperation op);

    /** Static function to check if given info will lead to a valid configuration of @ref NEReductionOperationKernel.
     *
     * @param[in] input  Source tensor info. Data type supported: QASYMM8_SIGNED/QASYMM8/F16/F32/S32.
     * @param[in] output Destination tensor info.Data types and data layouts supported: same as @p input, S32 for ARG_MIX/ARG_MAX.
     *                   Output will have the same number of dimensions as input.
     * @param[in] axis   Axis along which to reduce. Supported reduction axis : 0
     * @param[in] op     Reduction operation to perform.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output, unsigned int axis, ReductionOperation op);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;

private:
    const ITensor     *_input;
    ITensor           *_output;
    unsigned int       _reduction_axis;
    ReductionOperation _op;
};
} // namespace arm_compute
#endif /*ARM_COMPUTE_NEREDUCTIONOPERATIONKERNEL_H */
