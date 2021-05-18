/*
 * Copyright (c) 2019-2021 Arm Limited.
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
#ifndef ARM_COMPUTE_CLINSTANCENORMALIZATIONLAYERKERNEL_H
#define ARM_COMPUTE_CLINSTANCENORMALIZATIONLAYERKERNEL_H

#include "src/core/CL/ICLKernel.h"

#include "arm_compute/core/KernelDescriptors.h"

namespace arm_compute
{
// Forward declarations
class ICLTensor;

/** Interface for performing an instance normalization */
class CLInstanceNormalizationLayerKernel : public ICLKernel
{
public:
    /** Constructor */
    CLInstanceNormalizationLayerKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLInstanceNormalizationLayerKernel(const CLInstanceNormalizationLayerKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLInstanceNormalizationLayerKernel &operator=(const CLInstanceNormalizationLayerKernel &) = delete;
    /** Default Move Constructor. */
    CLInstanceNormalizationLayerKernel(CLInstanceNormalizationLayerKernel &&) = default;
    /** Default move assignment operator */
    CLInstanceNormalizationLayerKernel &operator=(CLInstanceNormalizationLayerKernel &&) = default;
    /** Default destructor */
    ~CLInstanceNormalizationLayerKernel() = default;

    /** Set the input and output tensors.
     *
     * @param[in]      compile_context The compile context to be used.
     * @param[in, out] input           Source tensor. Data types supported: F16/F32. Data layout supported: NCHW, NHWC
     *                                 In case of @p output tensor = nullptr this tensor will store the result of the normalization.
     * @param[in]      mean_var        Tensor containing the precomputed mean and variance values. Data types supported: F32.
     * @param[out]     output          Destination tensor. Data types and data layouts supported: same as @p input.
     * @param[in]      info            Kernel meta-data descriptor
     */
    void configure(const CLCompileContext &compile_context, ICLTensor *input, ICLTensor *mean_var, ICLTensor *output, const InstanceNormalizationLayerKernelInfo &info);

    /** Static function to check if given info will lead to a valid configuration of @ref CLInstanceNormalizationLayer.
     *
     * @param[in] input  Source tensor info. Data types supported: F16/F32. Data layout supported: NHWC, NCHW
     * @param[in] output Destination tensor info. Data types and data layouts supported: same as @p input.
     * @param[in] info   Kernel meta-data descriptor
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output, const InstanceNormalizationLayerKernelInfo &info);

    // Inherited methods overridden:
    void run(const Window &window, cl::CommandQueue &queue) override;

private:
    ICLTensor *_input;
    ICLTensor *_output;
    ICLTensor *_mean;
    bool       _run_in_place;
};

/** Interface for compute Mean and Variance per channel */
class CLComputeMeanVariance : public ICLKernel
{
public:
    /** Constructor */
    CLComputeMeanVariance();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLComputeMeanVariance(const CLComputeMeanVariance &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLComputeMeanVariance &operator=(const CLComputeMeanVariance &) = delete;
    /** Default Move Constructor. */
    CLComputeMeanVariance(CLComputeMeanVariance &&) = default;
    /** Default move assignment operator */
    CLComputeMeanVariance &operator=(CLComputeMeanVariance &&) = default;
    /** Default destructor */
    ~CLComputeMeanVariance() = default;

    /** Set the input and output tensors.
     *
     * @param[in]      compile_context     The compile context to be used.
     * @param[in, out] input               Source tensor. Data types supported: F16/F32. Data layout supported: NCHW, NHWC
     *                                     In case of @p output tensor = nullptr this tensor will store the result of the normalization.
     * @param[out]     output              Destination tensor. Data types and data layouts supported: same as @p input.
     * @param[in]      use_mixed_precision Use mixed precision in case of FP16 execution
     */
    void configure(const CLCompileContext &compile_context, ICLTensor *input, ICLTensor *output, bool use_mixed_precision);

    /** Static function to check if given info will lead to a valid configuration of @ref CLInstanceNormalizationLayer.
     *
     * @param[in] input  Source tensor info. Data types supported: F16/F32. Data layout supported: NHWC, NCHW
     * @param[in] output Destination tensor info. Data types and data layouts supported: same as @p input.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output);

    // Inherited methods overridden:
    void run(const Window &window, cl::CommandQueue &queue) override;

private:
    ICLTensor *_input;
    ICLTensor *_output;
};
} // namespace arm_compute
#endif /*ARM_COMPUTE_CLINSTANCENORMALIZATIONLAYERKERNEL_H */
