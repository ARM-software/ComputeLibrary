/*
 * Copyright (c) 2020-2021 Arm Limited.
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
#ifndef ARM_COMPUTE_CLMAXUNPOOLINGLAYER_H
#define ARM_COMPUTE_CLMAXUNPOOLINGLAYER_H

#include "arm_compute/core/Error.h"
#include "arm_compute/runtime/CL/functions/CLFill.h"
#include "arm_compute/runtime/IFunction.h"

#include <memory>

namespace arm_compute
{
class CLCompileContext;
class ICLTensor;
class ITensorInfo;
class CLMaxUnpoolingLayerKernel;
struct PoolingLayerInfo;

/** Function to perform MaxUnpooling. This function calls the following OpenCL kernels:
 *
 * -# @ref CLFill
 * -# @ref CLMaxUnpoolingLayerKernel
 */
class CLMaxUnpoolingLayer : public IFunction
{
public:
    /** Constructor */
    CLMaxUnpoolingLayer();
    /** Prevent instances of this class from being copied */
    CLMaxUnpoolingLayer(const CLMaxUnpoolingLayer &) = delete;
    /** Prevent instances of this class from being copied */
    CLMaxUnpoolingLayer &operator=(const CLMaxUnpoolingLayer &) = delete;
    /** Default destructor */
    ~CLMaxUnpoolingLayer();
    /** Set the input and output tensors.
     *
     * Valid data layouts:
     * - NHWC
     * - NCHW
     *
     * Valid data type configurations:
     * |src            |dst            |
     * |:--------------|:--------------|
     * |QASYMM8        |QASYMM8        |
     * |QASYMM8_SIGNED |QASYMM8_SIGNED |
     * |F16            |F16            |
     * |F32            |F32            |
     *
     * @note Output shape must be equal to the shape of the original input to pool.
     *
     * @param[in]  input     Source tensor. Data types supported: QASYMM8/QASYMM8_SIGNED/F16/F32.
     * @param[in]  indices   Tensor containing the offset to store the input elements in the output tensor.
     *                       @ref CLPoolingLayer with indices should precede this function in order to
     *                       properly reconstruct the output tensor.
     *                       The tensor shape of this tensor has to be equal to the input tensor shape. Data type supported: U32.
     * @param[out] output    Destination tensor. Data types supported: Same as @p input.
     * @param[in]  pool_info Contains pooling operation information described in @ref PoolingLayerInfo.
     */
    void configure(ICLTensor *input, ICLTensor *indices, ICLTensor *output, const PoolingLayerInfo &pool_info);
    /** Set the input and output tensors.
     *
     * @note Output shape must be equal to the shape of the original input to pool.
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  input           Source tensor. Data types supported: QASYMM8/QASYMM8_SIGNED/F16/F32.
     * @param[in]  indices         Tensor containing the offset to store the input elements in the output tensor.
     *                             @ref CLPoolingLayer with indices should precede this function in order to
     *                             properly reconstruct the output tensor.
     *                             The tensor shape of this tensor has to be equal to the input tensor shape. Data type supported: U32.
     * @param[out] output          Destination tensor. Data types supported: Same as @p input.
     * @param[in]  pool_info       Contains pooling operation information described in @ref PoolingLayerInfo.
     */
    void configure(const CLCompileContext &compile_context, ICLTensor *input, ICLTensor *indices, ICLTensor *output, const PoolingLayerInfo &pool_info);
    /** Static function to check if given info will lead to a valid configuration of @ref CLMaxUnpoolingLayer
     *
     * @param[in] input     Source tensor info. Data types supported: QASYMM8/QASYMM8_SIGNED/F16/F32.
     * @param[in] output    Destination tensor info. Data types supported: Same as @p input.
     * @param[in] indices   TensorInfo associated to the tensor containing the offset to store the input elements in the output tensor.
     *                      @ref CLPoolingLayer with indices should precede this function in order to
     *                      properly reconstruct the output tensor.
     *                      The tensor shape of this tensor has to be equal to the input tensor shape. Data type supported: U32.
     * @param[in] pool_info Contains pooling operation information described in @ref PoolingLayerInfo.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *indices, const ITensorInfo *output, const PoolingLayerInfo &pool_info);

    // Inherited methods overridden:
    void run() override;

private:
    CLFill                                     _fill;
    std::unique_ptr<CLMaxUnpoolingLayerKernel> _unpooling_layer_kernel;
};
}
#endif /* ARM_COMPUTE_CLMAXUNPOOLINGLAYER_H */
