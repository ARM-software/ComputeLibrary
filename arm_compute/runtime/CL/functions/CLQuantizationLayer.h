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
#ifndef ARM_COMPUTE_CLQUANTIZATIONLAYER_H
#define ARM_COMPUTE_CLQUANTIZATIONLAYER_H

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/IFunction.h"

#include <memory>

namespace arm_compute
{
class CLCompileContext;
class ICLTensor;
class ITensorInfo;

/** Basic function to simulate a quantization layer. This function calls the following CL kernels:
 *
 * -# @ref opencl::ClQuantize
 *
 * @note The implementation supports only 3D input tensors.
 *
 */
class CLQuantizationLayer : public IFunction
{
public:
    /** Default Constructor */
    CLQuantizationLayer();
    /** Default Destructor */
    ~CLQuantizationLayer();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLQuantizationLayer(const CLQuantizationLayer &) = delete;
    /** Default move constructor */
    CLQuantizationLayer(CLQuantizationLayer &&) = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLQuantizationLayer &operator=(const CLQuantizationLayer &) = delete;
    /** Default move assignment operator */
    CLQuantizationLayer &operator=(CLQuantizationLayer &&) = default;
    /** Set the input and output tensors.
     *
     * Valid data layouts:
     * - All
     *
     * Valid data type configurations:
     * |src                |dst                                |
     * |:------------------|:----------------------------------|
     * |QASYMM8            |QASYMM8, QASYMM8_SIGNED, QASYMM16  |
     * |QASYMM8_SIGNED     |QASYMM8, QASYMM8_SIGNED, QASYMM16  |
     * |F16                |QASYMM8, QASYMM8_SIGNED, QASYMM16  |
     * |F32                |QASYMM8, QASYMM8_SIGNED, QASYMM16  |
     *
     * @param[in]  input  Source tensor. The dimensions over the third will be interpreted as batches. Data types supported: QASYMM8/QASYMM8_SIGNED/F16/32.
     * @param[out] output Destination tensor with the same dimensions of input. Data types supported: QASYMM8/QASYMM8_SIGNED/QASYMM16.
     *
     * @note Output auto initialization is not supported by this function
     */
    void configure(const ICLTensor *input, ICLTensor *output);
    /** Set the input and output tensors.
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  input           Source tensor. The dimensions over the third will be interpreted as batches. Data types supported: QASYMM8/QASYMM8_SIGNED/F16/32.
     * @param[out] output          Destination tensor with the same dimensions of input. Data types supported: QASYMM8/QASYMM8_SIGNED/QASYMM16.
     *
     * @note Output auto initialization is not supported by this function
     */
    void configure(const CLCompileContext &compile_context, const ICLTensor *input, ICLTensor *output);
    /** Static function to check if given info will lead to a valid configuration of @ref CLQuantizationLayer
     *
     * @param[in] input  Input tensor info. The dimensions over the third will be interpreted as batches. Data types supported: QASYMM8/QASYMM8_SIGNED/F16/32.
     * @param[in] output Output tensor info. Data types supported: QASYMM8/QASYMM8_SIGNED/QASYMM16.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output);

    // Inherited methods overridden:
    void run() override;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};
} //namespace arm_compute
#endif /* ARM_COMPUTE_CLQUANTIZATIONLAYER_H */
