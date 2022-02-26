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
#ifndef ARM_COMPUTE_CLPOOLINGLAYER_H
#define ARM_COMPUTE_CLPOOLINGLAYER_H

#include "arm_compute/runtime/IFunction.h"

#include "arm_compute/core/Types.h"

#include <memory>

namespace arm_compute
{
class CLCompileContext;
class ICLTensor;
class ITensorInfo;

/** Basic function to run  @ref opencl::ClPool2d */
class CLPoolingLayer : public IFunction
{
public:
    /** Default Constructor */
    CLPoolingLayer();
    /** Default Destructor */
    ~CLPoolingLayer();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLPoolingLayer(const CLPoolingLayer &) = delete;
    /** Default move constructor */
    CLPoolingLayer(CLPoolingLayer &&) = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLPoolingLayer &operator=(const CLPoolingLayer &) = delete;
    /** Default move assignment operator */
    CLPoolingLayer &operator=(CLPoolingLayer &&) = default;
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
     * @note Source tensor is padded with -inf for MAX pooling and 0 otherwise
     *       Cases where pooling region is completely outside input tensor are not supported
     *
     * @param[in,out] input     Source tensor. (Written to only when padding != 0) Data types supported: QASYMM8/QASYMM8_SIGNED/F16/F32.
     * @param[out]    output    Destination tensor. Data types supported: Same as @p input.
     * @param[in]     pool_info Contains pooling operation information described in @ref PoolingLayerInfo.
     * @param[out]    indices   (optional) The indices of the maximal values. Data type supported: U32.
     */
    void configure(ICLTensor *input, ICLTensor *output, const PoolingLayerInfo &pool_info, ICLTensor *indices = nullptr);
    /** Set the input and output tensors.
     *
     * @param[in]     compile_context The compile context to be used.
     * @param[in,out] input           Source tensor. (Written to only when padding != 0) Data types supported: QASYMM8/QASYMM8_SIGNED/F16/F32.
     * @param[out]    output          Destination tensor. Data types supported: Same as @p input.
     * @param[in]     pool_info       Contains pooling operation information described in @ref PoolingLayerInfo.
     * @param[out]    indices         (optional) The indices of the maximal values. Data type supported: U32.
     */
    void configure(const CLCompileContext &compile_context, ICLTensor *input, ICLTensor *output, const PoolingLayerInfo &pool_info, ICLTensor *indices = nullptr);
    /** Static function to check if given info will lead to a valid configuration of @ref CLPoolingLayer
     *
     * @param[in] input     Source tensor info. Data types supported: QASYMM8/QASYMM8_SIGNED/F16/F32.
     * @param[in] output    Destination tensor info. Data types supported: Same as @p input.
     * @param[in] pool_info Contains pooling operation information described in @ref PoolingLayerInfo.
     * @param[in] indices   (optional) The indices of the maximal values. Data type supported: U32.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output, const PoolingLayerInfo &pool_info, const ITensorInfo *indices = nullptr);

    // Inherited methods overridden:
    void run() override;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};
} // namespace arm_compute
#endif /* ARM_COMPUTE_CLPOOLINGLAYER_H */
