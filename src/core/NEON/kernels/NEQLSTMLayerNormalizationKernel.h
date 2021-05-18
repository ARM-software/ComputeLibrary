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
#ifndef ARM_COMPUTE_NEQLSTMLAYERNORMALIZATIONKERNEL_H
#define ARM_COMPUTE_NEQLSTMLAYERNORMALIZATIONKERNEL_H

#include "src/core/NEON/INEKernel.h"
#include <functional>

namespace arm_compute
{
class ITensor;

/** Kernel to perform layer normalization for QLSTM. */
class NEQLSTMLayerNormalizationKernel : public INEKernel
{
public:
    const char *name() const override
    {
        return "NEQLSTMLayerNormalizationKernel";
    }
    /** Default constructor */
    NEQLSTMLayerNormalizationKernel() = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEQLSTMLayerNormalizationKernel(const NEQLSTMLayerNormalizationKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEQLSTMLayerNormalizationKernel &operator=(const NEQLSTMLayerNormalizationKernel &) = delete;
    /** Default Move Constructor. */
    NEQLSTMLayerNormalizationKernel(NEQLSTMLayerNormalizationKernel &&) = default;
    /** Default move assignment operator */
    NEQLSTMLayerNormalizationKernel &operator=(NEQLSTMLayerNormalizationKernel &&) = default;
    /** Default destructor */
    ~NEQLSTMLayerNormalizationKernel() = default;

    /** Set the input and output tensors.
     *
     * @param[in]  input  Source tensor. Data types supported: QSYMM16.
     * @param[out] output Destination tensor. Data types supported: Same as @p input.
     * @param[in]  weight Weight tensor. Data types supported: Same as @p input.
     * @param[in]  bias   Bias tensor. Data types supported: S32
     */
    void configure(const ITensor *input, ITensor *output, const ITensor *weight, const ITensor *bias);
    /** Static function to check if given info will lead to a valid configuration of @ref NEQLSTMLayerNormalizationKernel
     *
     * @param[in] input  Source tensor info. Data types supported: QSYMM16.
     * @param[in] output Destination tensor info. Data types supported: Same as @p input.
     * @param[in] weight Weight tensor info. Data types supported: Same as @p input.
     * @param[in] bias   Bias tensor info. Data types supported: S32
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output, const ITensorInfo *weight, const ITensorInfo *bias);
    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;

private:
    // constants
    static constexpr uint32_t max_input_dimension{ 2 };  /**< The maximum input dimension supported */
    static constexpr uint32_t max_weight_dimension{ 1 }; /**< The maximum weight dimension supported */
    static constexpr uint32_t max_bias_dimension{ 1 };   /**< The maximum bias dimension supported */
    static constexpr uint32_t vector_size_byte{ 16 };    /**< Computation vector size in byte */

    using ComputeFuncType = std::function<void(NEQLSTMLayerNormalizationKernel &)>;

    ComputeFuncType _fn{}; /**< Function pointer to computation function */

    const ITensor *_input
    {
        nullptr
    }; /**< Input tensor */
    const ITensor *_weight
    {
        nullptr
    }; /**< Weight tensor */
    const ITensor *_bias
    {
        nullptr
    };                           /**< Bias tensor */
    ITensor *_output{ nullptr }; /**< Output tensor */

    int32_t _output_multiplier{}; /**< Multiplier for output values */
    int32_t _output_shift{};      /**< Shift value for output values */

    int32_t _window_start_x{}; /**< The beginning of x-axis iteration */
    int32_t _window_end_x{};   /**< The end of x-axis iteration */
    int32_t _window_step_x{};  /**< The size of x-axis iteration's step */

    Window _inout_window{};  /**< Window for input and output tensor */
    Window _weight_window{}; /**< Window for weight and bias tensor */

    /** Function to configure initial windows for destination of computation
     *
     * @param[in] Target destination tensor to use for output window
     *
     * @return configured window
     */
    Window configure_window(ITensor *target);
    // Function to compute for data type QSYMM16
    void compute_qsymm16();
    /** Function to compute summation and summation of squared input of the given input pointer
     *
     * @param[in] Input_ptr pointer to input array
     *
     */
    std::pair<int64_t, int64_t> sum_qsymm16(const int16_t *input_ptr);
    /** Function to normalize values using computed mean and standard deviation
     *
     * @param[in] input_ptr     Pointer to input array
     * @param[in] output_ptr    Pointer to output array
     * @param[in] weight_ptr    Pointer to weight array
     * @param[in] bias_ptr      Pointer to bias array
     * @param[in] mean          Mean value
     * @param[in] inv_std_mul   Quantized multiplier for standard deviation
     * @param[in] inv_std_shift Shift for standard deviation
     *
     */
    void normalize_qasymm16(const int16_t *input_ptr,
                            int16_t       *output_ptr,
                            const int16_t *weight_ptr,
                            const int32_t *bias_ptr,
                            int32_t mean, int32_t inv_std_mul, int32_t inv_std_shift);
    /** Function to compute output quantization information */
    QuantizationInfo compute_output_qinfo();
};
} // namespace arm_compute
#endif /* ARM_COMPUTE_NEQLSTMLAYERNORMALIZATIONKERNEL_H */
