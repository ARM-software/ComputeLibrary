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
#ifndef ARM_COMPUTE_CL_REDUCE_MEAN_H
#define ARM_COMPUTE_CL_REDUCE_MEAN_H

#include "arm_compute/runtime/CL/ICLSimpleFunction.h"
#include "arm_compute/runtime/CL/functions/CLDequantizationLayer.h"
#include "arm_compute/runtime/CL/functions/CLElementwiseOperations.h"
#include "arm_compute/runtime/CL/functions/CLQuantizationLayer.h"
#include "arm_compute/runtime/CL/functions/CLReductionOperation.h"
#include "arm_compute/runtime/CL/functions/CLReshapeLayer.h"
#include "arm_compute/runtime/IMemoryManager.h"

namespace arm_compute
{
// Forward Declarations
class ICLTensor;

/** Basic function to perform reduce operation */
class CLReduceMean : public IFunction
{
public:
    /** Default constructor */
    CLReduceMean(std::shared_ptr<IMemoryManager> memory_manager = nullptr);
    /** Configure kernel
     *
     * Valid data layouts:
     * - All
     *
     * Valid data type configurations:
     * |src            |dst            |
     * |:--------------|:--------------|
     * |QASYMM8        |QASYMM8        |
     * |QASYMM8_SIGNED |QASYMM8_SIGNED |
     * |F16            |F16            |
     * |F32            |F32            |
     *
     * @note Supported tensor rank: up to 4
     *
     * @param[in]  input          Source tensor. Data type supported: QASYMM8/QASYMM8_SIGNED/F16/F32
     * @param[in]  reduction_axis Reduction axis vector.
     * @param[in]  keep_dims      If positive, retains reduced dimensions with length 1.
     * @param[out] output         Destination tensor. Data type supported: Same as @p input
     */
    void configure(ICLTensor *input, const Coordinates &reduction_axis, bool keep_dims, ICLTensor *output);
    /** Configure kernel
     *
     * @note Supported tensor rank: up to 4
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  input           Source tensor. Data type supported: QASYMM8/QASYMM8_SIGNED/F16/F32
     * @param[in]  reduction_axis  Reduction axis vector.
     * @param[in]  keep_dims       If positive, retains reduced dimensions with length 1.
     * @param[out] output          Destination tensor. Data type supported: Same as @p input
     */
    void configure(const CLCompileContext &compile_context, ICLTensor *input, const Coordinates &reduction_axis, bool keep_dims, ICLTensor *output);

    /** Static function to check if given info will lead to a valid configuration of @ref CLReduceMean
     *
     * @param[in] input          Source tensor. Data type supported: QASYMM8/QASYMM8_SIGNED/F16/F32
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
    std::vector<CLReductionOperation> _reduction_kernels;
    std::vector<CLTensor>             _reduced_outs;
    CLReshapeLayer                    _reshape;
    CLDequantizationLayer             _dequant;
    CLQuantizationLayer               _requant;
    int                               _reduction_ops;
    bool                              _keep_dims;
    bool                              _do_requant;
    CLTensor                          _input_no_quant;
    CLTensor                          _output_no_quant;
};
} // namespace arm_compute
#endif /* ARM_COMPUTE_CL_REDUCE_MEAN_H */
