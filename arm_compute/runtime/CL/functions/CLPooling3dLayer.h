/*
 * Copyright (c) 2022 Arm Limited.
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
#ifndef ARM_COMPUTE_CLPOOLING3DLAYER_H
#define ARM_COMPUTE_CLPOOLING3DLAYER_H

#include "arm_compute/runtime/IFunction.h"

#include "arm_compute/core/Types.h"

#include <memory>

namespace arm_compute
{
class CLCompileContext;
class ICLTensor;
class ITensorInfo;

/** Basic function to run  @ref opencl::ClPool3d */
class CLPooling3dLayer : public IFunction
{
public:
    /** Default Constructor */
    CLPooling3dLayer();
    /** Default Destructor */
    ~CLPooling3dLayer();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLPooling3dLayer(const CLPooling3dLayer &) = delete;
    /** Default move constructor */
    CLPooling3dLayer(CLPooling3dLayer &&) = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLPooling3dLayer &operator=(const CLPooling3dLayer &) = delete;
    /** Default move assignment operator */
    CLPooling3dLayer &operator=(CLPooling3dLayer &&) = default;
    /** Set the input and output tensors.
     *
     * Valid data layouts:
     * - NDHWC
     *
     * Valid data type configurations:
     * |src            |dst            |
     * |:--------------|:--------------|
     * |F16            |F16            |
     * |F32            |F32            |
     * |QASYMM8        |QASYMM8        |
     * |QASYMM8_SIGNED |QASYMM8_SIGNED |
     *
     * @note Source tensor is padded with -inf for MAX pooling and 0 otherwise
     *       Cases where pooling region is completely outside input tensor are not supported
     *
     * @note Asymmetric padding is not supported when dimension rounding type == CEIL.
     *
     * @param[in,out] input     Source tensor. Data types supported: F16/F32/QASYMM8/QASYMM8_SIGNED.
     * @param[out]    output    Destination tensor. Data types supported: Same as @p input.
     * @param[in]     pool_info Contains 3d pooling operation information described in @ref Pooling3dLayerInfo.
     */
    void configure(const ICLTensor *input, ICLTensor *output, const Pooling3dLayerInfo &pool_info);
    /** Set the input and output tensors.
     *
     * @param[in]     compile_context The compile context to be used.
     * @param[in,out] input           Source tensor. Data types supported: F16/F32/QASYMM8/QASYMM8_SIGNED.
     * @param[out]    output          Destination tensor. Data types supported: Same as @p input.
     * @param[in]     pool_info       Contains 3d pooling operation information described in @ref Pooling3dLayerInfo.
     */
    void configure(const CLCompileContext &compile_context, const ICLTensor *input, ICLTensor *output, const Pooling3dLayerInfo &pool_info);
    /** Static function to check if given info will lead to a valid configuration of @ref CLPooling3dLayer
     *
     * @param[in] input     Source tensor info. Data types supported: F16/F32/QASYMM8/QASYMM8_SIGNED.
     * @param[in] output    Destination tensor info. Data types supported: Same as @p input.
     * @param[in] pool_info Contains 3d pooling operation information described in @ref Pooling3dLayerInfo.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output, const Pooling3dLayerInfo &pool_info);

    // Inherited methods overridden:
    void run() override;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};
} // namespace arm_compute
#endif /* ARM_COMPUTE_CLPOOLING3DLAYER_H */
