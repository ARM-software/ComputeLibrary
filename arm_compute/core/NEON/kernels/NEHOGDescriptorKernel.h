/*
 * Copyright (c) 2016, 2017 ARM Limited.
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
#ifndef __ARM_COMPUTE_NEHOGDESCRIPTORKERNEL_H__
#define __ARM_COMPUTE_NEHOGDESCRIPTORKERNEL_H__

#include "arm_compute/core/IHOG.h"
#include "arm_compute/core/NEON/INEKernel.h"
#include "arm_compute/core/Size2D.h"

namespace arm_compute
{
class ITensor;

/** NEON kernel to perform HOG Orientation Binning */
class NEHOGOrientationBinningKernel : public INEKernel
{
public:
    /** Default constructor */
    NEHOGOrientationBinningKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEHOGOrientationBinningKernel(const NEHOGOrientationBinningKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEHOGOrientationBinningKernel &operator=(const NEHOGOrientationBinningKernel &) = delete;
    /** Allow instances of this class to be moved */
    NEHOGOrientationBinningKernel(NEHOGOrientationBinningKernel &&) = default;
    /** Allow instances of this class to be moved */
    NEHOGOrientationBinningKernel &operator=(NEHOGOrientationBinningKernel &&) = default;
    /** Default destructor */
    ~NEHOGOrientationBinningKernel() = default;

    /**  Initialise the kernel's inputs, output and HOG's metadata
     *
     * @param[in]  input_magnitude Input tensor which stores the magnitude of the gradient for each pixel. Data type supported: S16.
     * @param[in]  input_phase     Input tensor which stores the phase of the gradient for each pixel. Data type supported: U8
     * @param[out] output          Output tensor which stores the local HOG for each cell. Data type supported: F32. Number of channels supported: equal to the number of histogram bins per cell
     * @param[in]  hog_info        HOG's metadata
     */
    void configure(const ITensor *input_magnitude, const ITensor *input_phase, ITensor *output, const HOGInfo *hog_info);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;

private:
    /** Common signature for all the specialised block normalization functions
     *
     * @param[in]  mag_row_ptr   Pointer to the first row of the cell in the magnitude tensor
     * @param[in]  phase_row_ptr Pointer to the first row of the cell in the phase tensor
     * @param[out] output_ptr    Pointer to the output cell of hog space tensor
     * @param[in]  mag_stride    Stride of the magnitude tensor
     * @param[in]  phase_stride  Stride of the phase tensor
     * @param[in]  cell_width    Width of the cell
     * @param[in]  cell_height   Height of the cell
     * @param[in]  num_bins      Number of bins for each cell
     * @param[in]  phase_scale   Scale factor to apply to the phase in order to calculate the histogram index
     */
    using OrientBinFunc = void(const int16_t *__restrict mag_row_ptr, const uint8_t *__restrict phase_row_ptr, float *__restrict output_ptr, size_t mag_stride, size_t phase_stride, size_t cell_width,
                               size_t cell_height, size_t num_bins, float phase_scale);
    /** Orientation binning function to use for the particular cell width passed to configure() */
    OrientBinFunc *_func;
    const ITensor *_input_magnitude;
    const ITensor *_input_phase;
    ITensor       *_output;
    size_t         _cell_width;
    size_t         _cell_height;
    size_t         _num_bins;
    float          _phase_scale;
};

/** NEON kernel to perform HOG block normalization */
class NEHOGBlockNormalizationKernel : public INEKernel
{
public:
    /** Default constructor */
    NEHOGBlockNormalizationKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEHOGBlockNormalizationKernel(const NEHOGBlockNormalizationKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEHOGBlockNormalizationKernel &operator=(const NEHOGBlockNormalizationKernel &) = delete;
    /** Allow instances of this class to be moved */
    NEHOGBlockNormalizationKernel(NEHOGBlockNormalizationKernel &&) = default;
    /** Allow instances of this class to be moved */
    NEHOGBlockNormalizationKernel &operator=(NEHOGBlockNormalizationKernel &&) = default;
    /** Default destructor */
    ~NEHOGBlockNormalizationKernel() = default;

    /** Initialise the kernel's input, output and HOG's metadata
     *
     * @param[in]  input    Input tensor which stores the local HOG for each cell. Data type supported: F32. Number of channels supported: equal to the number of histogram bins per cell
     * @param[out] output   Output tensor which stores the normalised blocks. Data type supported: F32. Number of channels supported: equal to the number of histogram bins per block
     * @param[in]  hog_info HOG's metadata
     */
    void configure(const ITensor *input, ITensor *output, const HOGInfo *hog_info);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;

private:
    /** Common signature for all the specialised block normalization functions
     *
     * @param[in]  input_row_ptr              Pointer to the first row of the block in the input hog space tensor
     * @param[out] output_ptr                 Pointer to the output block of the hog normalized space
     * @param[in]  input_stride               Stride of the input hog space tensor
     * @param[in]  num_cells_per_block_height Number of cells per block along the Y direction
     * @param[in]  num_bins_block_x           Number of bins per block along the X direction
     * @param[in]  num_bins_block             Number of total bins per block
     * @param[in]  l2_hyst_threshold          Threshold to use for l2 hysteresis normalization
     */
    using BlockNormFunc = void(const float *input_row_ptr, float *output_ptr, size_t input_stride, size_t num_cells_per_block_height, size_t num_bins_block_x, size_t num_bins_block,
                               float l2_hyst_threshold);
    /** Block normalization function to use for the particular normalization type passed to configure() */
    BlockNormFunc *_func;
    const ITensor *_input;
    ITensor       *_output;
    Size2D         _num_cells_per_block;
    Size2D         _num_cells_per_block_stride;
    size_t         _num_bins;
    float          _l2_hyst_threshold;
};
} // namespace arm_compute
#endif /* __ARM_COMPUTE_NEHOGDESCRIPTORKERNEL_H__ */
