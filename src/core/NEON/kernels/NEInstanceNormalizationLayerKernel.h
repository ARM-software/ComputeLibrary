/*
 * Copyright (c) 2019-2022 Arm Limited.
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
#ifndef ARM_COMPUTE_NEINSTANCENORMALIZATIONLAYERKERNEL_H
#define ARM_COMPUTE_NEINSTANCENORMALIZATIONLAYERKERNEL_H

#include "src/core/NEON/INEKernel.h"

namespace arm_compute
{
class ITensor;
struct InstanceNormalizationLayerKernelInfo;

/** Interface for performing an instance normalization */
class NEInstanceNormalizationLayerKernel : public INEKernel
{
public:
    const char *name() const override
    {
        return "NEInstanceNormalizationLayerKernel";
    }
    /** Default constructor */
    NEInstanceNormalizationLayerKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEInstanceNormalizationLayerKernel(const NEInstanceNormalizationLayerKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEInstanceNormalizationLayerKernel &operator=(const NEInstanceNormalizationLayerKernel &) = delete;
    /** Allow instances of this class to be moved */
    NEInstanceNormalizationLayerKernel(NEInstanceNormalizationLayerKernel &&) = default;
    /** Allow instances of this class to be moved */
    NEInstanceNormalizationLayerKernel &operator=(NEInstanceNormalizationLayerKernel &&) = default;
    /** Default destructor */
    ~NEInstanceNormalizationLayerKernel() = default;
    /** Set the input and output tensors.
     *
     * @param[in, out] input  Source tensor. Data types supported: F16/F32. Data layout supported: NCHW
     *                        In case of @p output tensor = nullptr this tensor will store the result of the normalization.
     * @param[out]     output Destination tensor. Data types and data layouts supported: same as @p input.
     * @param[in]      info   Kernel meta-data descriptor
     */
    void configure(ITensor *input, ITensor *output, const InstanceNormalizationLayerKernelInfo &info);

    /** Static function to check if given info will lead to a valid configuration of @ref NEInstanceNormalizationLayer.
     *
     * @param[in] input  Source tensor info. Data types supported: F16/F32. Data layout supported: NCHW
     * @param[in] output Destination tensor info. Data types and data layouts supported: same as @p input.
     * @param[in] info   Kernel meta-data descriptor
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output, const InstanceNormalizationLayerKernelInfo &info);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;

private:
    /** Common signature for all the specialized instance normalization functions
     *
     * @param[in, out] input   An input tensor. In case of @p output tensor = nullptr this tensor will store the result of the normalization.
     * @param[out]     output  The output tensor.
     * @param[in]      gamma   The scale scalar value applied to the normalized tensor. Defaults to 1.0
     * @param[in]      beta    The offset scalar value applied to the normalized tensor. Defaults to 0.0
     * @param[in]      epsilon Lower bound value for the normalization. Defaults to 1e-12
     */
    using NormalizationFunction = void(ITensor *input, ITensor *output, float gamma, float beta, float epsilon, const Window &window);

    ITensor *_input;
    ITensor *_output;
    float    _gamma;
    float    _beta;
    float    _epsilon;
    bool     _use_mixed_precision{ true };
};
} // namespace arm_compute
#endif /*ARM_COMPUTE_NEINSTANCENORMALIZATIONLAYERKERNEL_H */
