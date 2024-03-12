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
#ifndef ARM_COMPUTE_NEMEANSTDDEVNORMALIZATIONKERNEL_H
#define ARM_COMPUTE_NEMEANSTDDEVNORMALIZATIONKERNEL_H

#include "src/core/NEON/INEKernel.h"

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
#include <arm_fp16.h>
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */

namespace arm_compute
{
class ITensor;

/** Interface for the kernel to normalize the input 2D tensor across the first dimension with respect to mean and standard deviation of the same dimension. */
class NEMeanStdDevNormalizationKernel : public INEKernel
{
public:
    const char *name() const override
    {
        return "NEMeanStdDevNormalizationKernel";
    }
    /** Default constructor */
    NEMeanStdDevNormalizationKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEMeanStdDevNormalizationKernel(const NEMeanStdDevNormalizationKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEMeanStdDevNormalizationKernel &operator=(const NEMeanStdDevNormalizationKernel &) = delete;
    /** Allow instances of this class to be moved */
    NEMeanStdDevNormalizationKernel(NEMeanStdDevNormalizationKernel &&) = default;
    /** Allow instances of this class to be moved */
    NEMeanStdDevNormalizationKernel &operator=(NEMeanStdDevNormalizationKernel &&) = default;
    /** Default destructor */
    ~NEMeanStdDevNormalizationKernel() = default;
    /** Initialise the kernel's input and outputs.
     *
     * @note If the output tensor is a nullptr, the normalization will be performed in-place.
     *
     * @param[in, out] input   Source tensor with 2 dimensions. In case of @p output tensor = nullptr,
     *                         this tensor will store the result of the normalization. Data types supported: F16/F32.
     * @param[out]     output  (Optional) Destination tensor. It can be nullptr in case of in-place computation. Data type supported: same as @p input
     * @param[in]      epsilon (Optional) Small float to avoid division by zero in case of zero standard deviation. Defaults to 1e-8.
     */
    void configure(ITensor *input, ITensor *output = nullptr, float epsilon = 1e-8f);
    /** Static function to check if given info will lead to a valid configuration of @ref NEMeanStdDevNormalizationKernel
     *
     * @param[in] input   Source tensor info with 2 dimensions. In case of @p output tensor info = nullptr,
     *                    this tensor will store the result of the normalization. Data types supported: F16/F32.
     * @param[in] output  (Optional) Destination tensor info. It can be nullptr in case of in-place computation. Data type supported: same as @p input
     * @param[in] epsilon (Optional) Small float to avoid division by zero in case of zero standard deviation. Defaults to 1e-8.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output = nullptr, float epsilon = 1e-8f);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;

private:
    /** Normalizes the input with respect to mean and standard deviation.
     *
     * @param[in] window Region on which to execute the kernel.
     */
    template <typename ScalarType, int size>
    void mean_stddev_normalization(const Window &window);

    ITensor *_input;
    ITensor *_output;
    float    _epsilon;

    using MeanStdDevNormFunction = void (NEMeanStdDevNormalizationKernel::*)(const Window &window);
};
} // namespace arm_compute
#endif /* ARM_COMPUTE_NEMEANSTDDEVNORMALIZATIONKERNEL_H */
