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
#ifndef __ARM_COMPUTE_NEON_REDUCE_MEAN_H__
#define __ARM_COMPUTE_NEON_REDUCE_MEAN_H__

#include "arm_compute/runtime/IFunction.h"

#include "arm_compute/core/NEON/kernels/NEFillBorderKernel.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/MemoryGroup.h"
#include "arm_compute/runtime/NEON/functions/NEReductionOperation.h"
#include "arm_compute/runtime/NEON/functions/NEReshapeLayer.h"

namespace arm_compute
{
class ITensor;

/** Basic function to perform reduce operation */
class NEReduceMean : public IFunction
{
public:
    /** Constructor */
    NEReduceMean(std::shared_ptr<IMemoryManager> memory_manager = nullptr);
    /** Configure kernel
     *
     * @note Supported tensor rank: up to 4
     *
     * @param[in]  input          Source tensor. Data type supported: QASYMM8/F16/F32
     * @param[in]  reduction_axis Reduction axis vector.
     * @param[in]  keep_dims      If positive, retains reduced dimensions with length 1.
     * @param[out] output         Destination tensor. Data type supported: Same as @p input
     */
    void configure(ITensor *input, const Coordinates &reduction_axis, bool keep_dims, ITensor *output);

    /** Static function to check if given info will lead to a valid configuration of @ref NEReduceMean
     *
     * @param[in] input          Source tensor. Data type supported: QASYMM8/F16/F32
     * @param[in] reduction_axis Reduction axis vector.
     * @param[in] keep_dims      If positive, retains reduced dimensions with length 1.
     * @param[in] output         Destination tensor. Data type supported: Same as @p input
     *
     * @return A status
     */
    static Status validate(const ITensorInfo *input, const Coordinates &reduction_axis, bool keep_dims, const ITensorInfo *output);

    // Inherited methods overridden:
    void run() override;

private:
    MemoryGroup                       _memory_group;
    std::vector<NEReductionOperation> _reduction_kernels;
    std::vector<Tensor>               _reduced_outs;
    NEReshapeLayer                    _reshape;
    unsigned int                      _reduction_ops;
    bool                              _keep_dims;
};
} // namespace arm_compute
#endif /* __ARM_COMPUTE_NEON_REDUCE_MEAN_H__ */
