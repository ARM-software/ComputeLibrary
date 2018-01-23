/*
 * Copyright (c) 2017-2018 ARM Limited.
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
#ifndef __ARM_COMPUTE_NEGEMMWINOGRADLAYERKERNEL_H__
#define __ARM_COMPUTE_NEGEMMWINOGRADLAYERKERNEL_H__

#include "arm_compute/core/NEON/INEKernel.h"
#include "arm_compute/core/NEON/kernels/winograd/convolution.hpp"
#include "arm_compute/core/NEON/kernels/winograd/tensor.hpp"

namespace arm_compute
{
class ITensor;
class NEWinogradLayerKernel;
class NEWinogradLayerTransformInputKernel;
class NEWinogradLayerTransformWeightsKernel;

class Winograd3x3F32 final
{
public:
    /** Create a new Winograd convolution layer.
     *
     * @param[in]  n_batches         Number of batches in the input and output tensors.
     * @param[in]  n_input_channels  Number of feature maps in a batch of the input tensor.
     * @param[in]  n_input_rows      Number of rows in a feature map of the input tensor.
     * @param[in]  n_input_cols      Number of columns in a feature map of the input tensor.
     * @param[in]  n_output_channels Number of feature maps in the output tensor.
     * @param[in]  same_padding      Use "SAME" padding, otherwise use "VALID".
     * @param[in]  weights           Pointer to weight tensor in spatial domain. Must be ordered as "Height x Rows x Input Feature Maps x Output Feature Maps.
     * @param[out] weights_storage   Pointer to storage for weight tensor in the Winograd domain. Must be at least the size returned by `get_weight_storage_size
     * @param[in]  input             Pointer to NHWC ordered input tensor, in the spatial domain.
     * @param[out] winograd_input    Pointer to working space for the input tensor in the Winograd domain. Must be at least the size returned by `get_input_storage_size`.
     * @param[in]  biases            Pointer to the biases vector.
     * @param[out] output            Pointer to NHWC ordered output tensor, in the spatial domain.
     * @param[out] winograd_output   Pointer to working space for the output tensor in the Winograd domain. Must be at least the size returned by `get_output_storage_size`.
     */
    friend class NEWinogradLayerKernel;
    friend class NEWinogradLayerTransformInputKernel;
    friend class NEWinogradLayerTransformOutputKernel;
    friend class NEWinogradLayerTransformWeightsKernel;

    Winograd3x3F32(
        const int          n_batches,
        const int          n_input_channels,
        const int          n_input_rows,
        const int          n_input_cols,
        const int          n_output_channels,
        const bool         same_padding,
        const float *const weights,
        float *const       weights_storage,
        const float *const input,
        float *const       winograd_input,
        float *const       output,
        float *const       winograd_output);

    ~Winograd3x3F32();

private:
    class Private;
    std::unique_ptr<Private> _pimpl;
};

class INEWinogradLayerTransformKernel : public INEKernel
{
public:
    /** Constructor */
    INEWinogradLayerTransformKernel();

    /** Prevent instances of this class from being copied (As this class contains pointers) */
    INEWinogradLayerTransformKernel(const INEWinogradLayerTransformKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    INEWinogradLayerTransformKernel &operator=(const INEWinogradLayerTransformKernel &) = delete;
    /** Allow instances of this class to be moved */
    INEWinogradLayerTransformKernel(INEWinogradLayerTransformKernel &&) = default;
    /** Allow instances of this class to be moved */
    INEWinogradLayerTransformKernel &operator=(INEWinogradLayerTransformKernel &&) = default;

    virtual ~INEWinogradLayerTransformKernel() = default;

    /** Initialise the kernel
     *
     * @param[in] convolver A pointer to the winograd convolver, this object must have been configured and is ready to execute 16 GEMMS .
     */
    virtual void configure(Winograd3x3F32 *convolver);

protected:
    Winograd3x3F32 *_convolver;
};

class NEWinogradLayerTransformInputKernel final : public INEWinogradLayerTransformKernel
{
public:
    const char *name() const override
    {
        return "NEWinogradLayerTransformInputKernel";
    }
    // Inherited methods overridden:
    void configure(Winograd3x3F32 *convolver) override;
    void run(const Window &window, const ThreadInfo &info) override;
    bool is_parallelisable() const override;
};

class NEWinogradLayerTransformOutputKernel final : public INEKernel
{
public:
    const char *name() const override
    {
        return "NEWinogradLayerTransformOutputKernel";
    }
    /** Constructor */
    NEWinogradLayerTransformOutputKernel();

    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEWinogradLayerTransformOutputKernel(const NEWinogradLayerTransformOutputKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEWinogradLayerTransformOutputKernel &operator=(const NEWinogradLayerTransformOutputKernel &) = delete;
    /** Allow instances of this class to be moved */
    NEWinogradLayerTransformOutputKernel(NEWinogradLayerTransformOutputKernel &&) = default;
    /** Allow instances of this class to be moved */
    NEWinogradLayerTransformOutputKernel &operator=(NEWinogradLayerTransformOutputKernel &&) = default;

    ~NEWinogradLayerTransformOutputKernel() = default;

    /** Configure the output transform kernel.
     *
     * @param[in]  biases              Pointer to the biases tensor.
     * @param[in]  output_workingspace Pointer to working space for the output tensor in the Winograd domain.
     * @param[in]  matrix_stride       Output matrix stride, can be computed with winograd::WinogradGEMM<2, 2, 3, 3>::Convolution<float, float>::get_output_matrix_stride()
     * @param[out] output              Pointer to NHWC ordered output tensor, in the spatial domain.
     * @param[in]  n_batches           Number of batches in the input tensor.
     * @param[in]  n_rows              Number of rows in output tensor.
     * @param[in]  n_cols              Number of columns in output tensor.
     * @param[in]  n_channels          Number of feature maps in the output tensor.
     */
    void configure(
        const ITensor     *biases,
        const float *const output_workingspace,
        const int          matrix_stride,
        float *const       output,
        const int          n_batches,
        const int          n_rows,
        const int          n_cols,
        const int          n_channels);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;
    bool is_parallelisable() const override;

private:
    const ITensor *_biases;
    const float   *_output_workspace;
    int            _matrix_stride;
    int            _matrix_row_stride;
    float         *_output;
    int            _n_batches;
    int            _n_rows;
    int            _n_cols;
    int            _n_channels;
};

class NEWinogradLayerTransformWeightsKernel final : public INEWinogradLayerTransformKernel
{
public:
    const char *name() const override
    {
        return "NEWinogradLayerTransformWeightsKernel";
    }
    // Inherited methods overridden:
    void configure(Winograd3x3F32 *convolver) override;
    void run(const Window &window, const ThreadInfo &info) override;
    bool is_parallelisable() const override;
};

class NEWinogradLayerKernel final : public INEKernel
{
public:
    const char *name() const override
    {
        return "NEWinogradLayerKernel";
    }
    /** Constructor */
    NEWinogradLayerKernel();

    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEWinogradLayerKernel(const NEWinogradLayerKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEWinogradLayerKernel &operator=(const NEWinogradLayerKernel &) = delete;
    /** Allow instances of this class to be moved */
    NEWinogradLayerKernel(NEWinogradLayerKernel &&) = default;
    /** Allow instances of this class to be moved */
    NEWinogradLayerKernel &operator=(NEWinogradLayerKernel &&) = default;

    ~NEWinogradLayerKernel() = default;

    /** Initialise the kernel
     *
     * @param[in] convolver A pointer to the winograd convolver, this object must have been configured and is ready to execute 16 GEMMS .
     */
    void configure(Winograd3x3F32 *convolver);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;

    /** Determine how much memory (in units of TIn) to allocate for the
     * transformed weights.
     *
     * @param[in] n_output_channels Number of output feature maps.
     * @param[in] n_input_channels  Number of input feature maps.
     */
    static unsigned int get_weight_storage_size(
        const int n_output_channels,
        const int n_input_channels);

    /** Determine how much memory (in units of TIn) to allocate for the
     * transformed input.
     *
     * @param[in] n_batches    Number of batches in the input tensor.
     * @param[in] n_channels   Number of feature maps in the input tensor.
     * @param[in] n_rows       Number of rows in each feature map.
     * @param[in] n_cols       Number of columns in each feature map.
     * @param[in] same_padding Use "SAME" padding, otherwise use "VALID".
     */
    static unsigned int get_input_storage_size(
        const int  n_batches,
        const int  n_channels,
        const int  n_rows,
        const int  n_cols,
        const bool same_padding);

    /** Determine how much memory (in units of TOut) to allocate for the
     * (Winograd domain) output.
     *
     * @param[in] n_batches         Number of batches in the output tensor.
     * @param[in] n_rows            Number of rows in each feature map of the input tensor.
     * @param[in] n_cols            Number of columns in each feature map of the input tensor.
     * @param[in] n_output_channels Number of feature maps in the output tensor.
     * @param[in] same_padding      Use "SAME" padding, otherwise use "VALID".
     */
    static unsigned int get_output_storage_size(
        const int  n_batches,
        const int  n_rows,
        const int  n_cols,
        const int  n_output_channels,
        const bool same_padding);

protected:
    Winograd3x3F32 *_convolver;
};

} // namespace arm_compute
#endif /*__ARM_COMPUTE_NEGEMMWINOGRADLAYERKERNEL_H__*/
