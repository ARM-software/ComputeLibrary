/*
 * Copyright (c) 2017 ARM Limited.
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
#ifndef __ARM_COMPUTE_CLL2NORMALIZE_H__
#define __ARM_COMPUTE_CLL2NORMALIZE_H__

#include "arm_compute/core/CL/kernels/CLL2NormalizeLayerKernel.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/CLMemoryGroup.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/ICLSimpleFunction.h"
#include "arm_compute/runtime/CL/functions/CLReductionOperation.h"
#include "arm_compute/runtime/IMemoryManager.h"

#include <cstdint>
#include <memory>

namespace arm_compute
{
class ICLTensor;

/** Perform reduction operation.
 */
class CLL2NormalizeLayer : public IFunction
{
public:
    /** Constructor */
    CLL2NormalizeLayer(std::shared_ptr<IMemoryManager> memory_manager = nullptr);

    /** Set the input and output tensors.
     *
     * @param[in]  input   Source tensor. Data types supported: QS8, QS16, F32.
     * @param[out] output  Destination tensor. Data types supported: Same as @p input.
     * @param[in]  axis    Axis along which to reduce. Supported reduction axis : 0
     * @param[in]  epsilon Lower bound value for the normalization.
     */
    void configure(ICLTensor *input, ICLTensor *output, unsigned int axis, float epsilon = 1e-12);

    // Inherited methods overridden:
    void run() override;

private:
    CLMemoryGroup            _memory_group;
    CLReductionOperation     _reduce_func;
    CLL2NormalizeLayerKernel _normalize_kernel;
    CLTensor                 _sumsq;
};
}
#endif /*__ARM_COMPUTE_CLL2NORMALIZE_H__ */
