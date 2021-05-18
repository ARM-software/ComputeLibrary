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
#ifndef ARM_COMPUTE_CLARGMINMAXLAYER_H
#define ARM_COMPUTE_CLARGMINMAXLAYER_H

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/functions/CLReshapeLayer.h"
#include "arm_compute/runtime/IFunction.h"
#include "arm_compute/runtime/IMemoryManager.h"
#include "arm_compute/runtime/MemoryGroup.h"

namespace arm_compute
{
class ITensorInfo;
class ICLTensor;
class CLArgMinMaxLayerKernel;

/** Function to calculate the index of the minimum or maximum values in a
 *  tensor based on an axis.
 *
 * @note The default data type for an uninitialized output tensor is
 *       signed 32-bit integer (S32). It is the user's responsibility to check
 *       that the results do not overflow because the indices are computed
 *       in unsigned 32-bit (U32).
 */
class CLArgMinMaxLayer : public IFunction
{
public:
    /** Default Constructor.
     *
     * @param[in] memory_manager (Optional) Memory manager.
     */
    CLArgMinMaxLayer(std::shared_ptr<IMemoryManager> memory_manager = nullptr);
    /** Prevent instances of this class from being copied */
    CLArgMinMaxLayer(const CLArgMinMaxLayer &) = delete;
    /** Prevent instances of this class from being copied */
    CLArgMinMaxLayer &operator=(const CLArgMinMaxLayer &) = delete;
    /** Prevent instances of this class to be moved */
    CLArgMinMaxLayer(CLArgMinMaxLayer &&) = delete;
    /** Prevent instances of this class to be moved */
    CLArgMinMaxLayer &operator=(CLArgMinMaxLayer &&) = delete;
    /** Default destructor */
    ~CLArgMinMaxLayer();
    /** Set the input and output tensors.
     *
     * Valid data layouts:
     * - All
     *
     * Valid data type configurations:
     * |src            |dst       |
     * |:--------------|:---------|
     * |QASYMM8        |U32, S32  |
     * |QASYMM8_SIGNED |U32, S32  |
     * |S32            |U32, S32  |
     * |F16            |U32, S32  |
     * |F32            |U32, S32  |
     *
     * @param[in]  input  Input source tensor. Data types supported: QASYMM8/QASYMM8_SIGNED/S32/F16/F32.
     * @param[in]  axis   Axis to find max/min index.
     * @param[out] output Output source tensor. Data types supported: U32/S32.
     * @param[in]  op     Reduction operation to perform. Operations supported: ARG_IDX_MAX, ARG_IDX_MIN
     */
    void configure(const ICLTensor *input, int axis, ICLTensor *output, const ReductionOperation &op);
    /** Set the input and output tensors.
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  input           Input source tensor. Data types supported: QASYMM8/QASYMM8_SIGNED/S32/F16/F32.
     * @param[in]  axis            Axis to find max/min index.
     * @param[out] output          Output source tensor. Data types supported: U32/S32.
     * @param[in]  op              Reduction operation to perform. Operations supported: ARG_IDX_MAX, ARG_IDX_MIN
     */
    void configure(const CLCompileContext &compile_context, const ICLTensor *input, int axis, ICLTensor *output, const ReductionOperation &op);
    /** Static function to check if given info will lead to a valid configuration of @ref CLArgMinMaxLayer
     *
     * @param[in] input  Input source tensor info. Data types supported: QASYMM8/QASYMM8_SIGNED/S32/F16/F32.
     * @param[in] axis   Axis to find max/min index.
     * @param[in] output Output source tensor info. Data types supported: U32/S32.
     * @param[in] op     Reduction operation to perform. Operations supported: ARG_IDX_MAX, ARG_IDX_MIN
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, int axis, const ITensorInfo *output, const ReductionOperation &op);

    // Inherited methods overridden:
    void run() override;

private:
    MemoryGroup                                          _memory_group;
    std::vector<CLTensor>                                _results_vector;
    CLTensor                                             _not_reshaped_output;
    std::vector<std::unique_ptr<CLArgMinMaxLayerKernel>> _reduction_kernels_vector;
    CLReshapeLayer                                       _reshape;
    unsigned int                                         _num_of_stages;
    unsigned int                                         _reduction_axis;
};
} // namespace arm_compute
#endif /* ARM_COMPUTE_CLARGMINMAXLAYER_H */
