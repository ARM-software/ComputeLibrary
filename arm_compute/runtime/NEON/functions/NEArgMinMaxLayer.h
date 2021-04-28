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
#ifndef ARM_COMPUTE_NEARGMINMAXLAYER_H
#define ARM_COMPUTE_NEARGMINMAXLAYER_H

#include "arm_compute/runtime/NEON/functions/NEReductionOperation.h"

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/MemoryGroup.h"
#include "arm_compute/runtime/NEON/INESimpleFunction.h"

namespace arm_compute
{
class ITensor;

/** Function to calculate the index of the minimum or maximum values in a
 *  tensor based on an axis.
 *
 *  This function calls the following kernels:
 *
 * -# @ref NEReductionOperationKernel
 * -# @ref NEFillBorderKernel
 *
 * @note The default data type for an uninitialized output tensor is
 *       signed 32-bit integer (S32). It is the user's responsibility to check
 *       that the results do not overflow because the indices are computed
 *       in unsigned 32-bit (U32).
 */
class NEArgMinMaxLayer : public IFunction
{
public:
    /** Constructor */
    NEArgMinMaxLayer(std::shared_ptr<IMemoryManager> memory_manager = nullptr);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEArgMinMaxLayer(const NEArgMinMaxLayer &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEArgMinMaxLayer &operator=(const NEArgMinMaxLayer &) = delete;
    /** Prevent instances of this class from being moved (As this class contains non movable objects) */
    NEArgMinMaxLayer(NEArgMinMaxLayer &&) = delete;
    /** Prevent instances of this class from being moved (As this class contains non movable objects) */
    NEArgMinMaxLayer &operator=(NEArgMinMaxLayer &&) = delete;
    /** Default destructor */
    ~NEArgMinMaxLayer();
    /** Set the input and output tensors.
     *
     * Valid data layouts:
     * - All
     *
     * Valid data type configurations:
     * |src            |dst        |
     * |:--------------|:----------|
     * |QASYMM8        |U32, S32   |
     * |QASYMM8_SIGNED |U32, S32   |
     * |S32            |U32, S32   |
     * |F16            |U32, S32   |
     * |F32            |U32, S32   |
     *
     * @param[in]  input  Input source tensor. Data types supported: QASYMM8_SIGNED/QASYMM8/S32/F16/F32.
     * @param[in]  axis   Axis to find max/min index.
     * @param[out] output Output source tensor. Data types supported: U32/S32.
     * @param[in]  op     Operation to perform: min or max
     */
    void configure(ITensor *input, int axis, ITensor *output, const ReductionOperation &op);
    /** Static function to check if given info will lead to a valid configuration of @ref NEArgMinMaxLayer
     *
     * @param[in] input  Input source tensor info. Data types supported: QASYMM8_SIGNED/QASYMM8/S32/F16/F32.
     * @param[in] axis   Axis to find max/min index.
     * @param[in] output Output source tensor info. Data types supported: U32/S32.
     * @param[in] op     Operation to perform: min or max
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, int axis, const ITensorInfo *output, const ReductionOperation &op);

    // Inherited methods overridden:
    void run() override;

private:
    std::unique_ptr<NEReductionOperation> _reduction_function;
};
} // namespace arm_compute
#endif /* ARM_COMPUTE_NEARGMINMAXLAYER_H */
