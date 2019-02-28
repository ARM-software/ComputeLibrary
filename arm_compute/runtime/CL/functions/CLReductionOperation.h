/*
 * Copyright (c) 2017-2019 ARM Limited.
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
#ifndef __ARM_COMPUTE_CLREDUCTIONOPERATION_H__
#define __ARM_COMPUTE_CLREDUCTIONOPERATION_H__

#include "arm_compute/core/CL/kernels/CLFillBorderKernel.h"
#include "arm_compute/core/CL/kernels/CLReductionOperationKernel.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/CLMemoryGroup.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/IFunction.h"
#include "arm_compute/runtime/IMemoryManager.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace arm_compute
{
class ICLTensor;

/** Perform reduction operation.
 */
class CLReductionOperation : public IFunction
{
public:
    /** Default Constructor.
     *
     * @param[in] memory_manager (Optional) Memory manager.
     */
    CLReductionOperation(std::shared_ptr<IMemoryManager> memory_manager = nullptr);

    /** Set the input and output tensors.
     *
     * @param[in]  input  Source tensor. Data types supported: QASYMM8/F16/F32.
     * @param[out] output Destination tensor. Data types and data layouts supported: Same as @p input.
     * @param[in]  axis   Axis along which to reduce. Supported reduction axis : 0, 1, 2, 3
     * @param[in]  op     Reduction operation to perform.
     */
    void configure(ICLTensor *input, ICLTensor *output, unsigned int axis, ReductionOperation op);

    /** Static function to check if given info will lead to a valid configuration of @ref CLReductionOperation.
     *
     * @param[in] input  Source tensor info. Data types supported: QASYMM8/F16/F32.
     * @param[in] output Destination tensor info. Data types and data layouts supported: Same as @p input.
     * @param[in] axis   Axis along which to reduce. Supported reduction axis : 0, 1, 2, 3
     * @param[in] op     Reduction operation to perform.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output, unsigned int axis, ReductionOperation op);

    // Inherited methods overridden:
    void run() override;

private:
    CLMemoryGroup                                 _memory_group;
    std::unique_ptr<CLTensor[]>                   _results_vector{ nullptr };
    std::unique_ptr<CLReductionOperationKernel[]> _reduction_kernels_vector{ nullptr };
    std::unique_ptr<CLFillBorderKernel[]>         _border_handlers_vector{ nullptr };
    unsigned int                                  _num_of_stages;
    unsigned int                                  _reduction_axis;
    bool                                          _is_serial;
};
} // namespace arm_compute
#endif /*__ARM_COMPUTE_CLREDUCTIONOPERATION_H__ */
