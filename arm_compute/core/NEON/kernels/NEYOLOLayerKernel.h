/*
 * Copyright (c) 2018 ARM Limited.
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
#ifndef __ARM_COMPUTE_NEYOLOLAYERKERNEL_H__
#define __ARM_COMPUTE_NEYOLOLAYERKERNEL_H__

#include "arm_compute/core/NEON/INEKernel.h"

namespace arm_compute
{
class ITensor;

/** Interface for the YOLO layer kernel. */
class NEYOLOLayerKernel : public INEKernel
{
public:
    const char *name() const override
    {
        return "NEYOLOLayerKernel";
    }
    /** Constructor */
    NEYOLOLayerKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEYOLOLayerKernel(const NEYOLOLayerKernel &) = delete;
    /** Default move constructor */
    NEYOLOLayerKernel(NEYOLOLayerKernel &&) = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEYOLOLayerKernel &operator=(const NEYOLOLayerKernel &) = delete;
    /** Default move assignment operator */
    NEYOLOLayerKernel &operator=(NEYOLOLayerKernel &&) = default;
    /** Default destructor */
    ~NEYOLOLayerKernel() = default;
    /** Set the input and output tensor.
     *
     * @note If the output tensor is a nullptr or is equal to the input, the activation function will be performed in-place
     *
     * @param[in, out] input       Source tensor. In case of @p output tensor = nullptr, this tensor will store the result
     *                             of the activation function. Data types supported: F16/F32.
     * @param[out]     output      Destination tensor. Data type supported: same as @p input
     * @param[in]      act_info    Activation layer parameters.
     * @param[in]      num_classes Number of classes to activate (must be submultiple of @p input channels)
     */
    void configure(ITensor *input, ITensor *output, const ActivationLayerInfo &act_info, int32_t num_classes);
    /** Static function to check if given info will lead to a valid configuration of @ref NEYOLOLayerKernel
     *
     * @param[in] input       Source tensor info. In case of @p output tensor info = nullptr, this tensor will store the result
     *                        of the activation function. Data types supported: F16/F32.
     * @param[in] output      Destination tensor info. Data type supported: same as @p input
     * @param[in] act_info    Activation layer information.
     * @param[in] num_classes Number of classes to activate (must be submultiple of @p input channels)
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output, const ActivationLayerInfo &act_info, int32_t num_classes);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;

private:
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    /** Function to run YOLO layer on fp16
     *
     * @param[in] window Region on which to execute the kernel.
     */
    void yolo_layer_fp16_nchw(const Window &window);
    /** Function to run batch normalization on fp16 on tensors with NHWC format
     *
     * @param[in] window Region on which to execute the kernel.
     */
    void yolo_layer_fp16_nhwc(const Window &window);
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
    /** Function to run YOLO layer on fp32
     *
     * @param[in] window Region on which to execute the kernel.
     */
    void yolo_layer_fp32_nchw(const Window &window);
    /** Function to run YOLO layer on fp32 on tensors with NHWC format
     *
     * @param[in] window Region on which to execute the kernel.
     */
    void yolo_layer_fp32_nhwc(const Window &window);
    /** Common signature for all the yolo layer functions
     *
     * @param[in] window Region on which to execute the kernel.
     */
    using YOLOFunctionPtr = void (NEYOLOLayerKernel::*)(const Window &window);

private:
    YOLOFunctionPtr     _func;
    ITensor            *_input;
    ITensor            *_output;
    ActivationLayerInfo _act_info;
    int32_t             _num_classes;
};
} // namespace arm_compute
#endif /*__ARM_COMPUTE_NEYOLOLAYERKERNEL_H__ */
