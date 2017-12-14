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
#ifndef __ARM_COMPUTE_NEHISTOGRAMKERNEL_H__
#define __ARM_COMPUTE_NEHISTOGRAMKERNEL_H__

#include "arm_compute/core/NEON/INEKernel.h"
#include "support/Mutex.h"

#include <cstddef>
#include <cstdint>

namespace arm_compute
{
class IDistribution1D;
class ITensor;
using IImage = ITensor;

/** Interface for the histogram kernel */
class NEHistogramKernel : public INEKernel
{
public:
    /** Default constructor */
    NEHistogramKernel();
    /** Default destructor */
    ~NEHistogramKernel() = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEHistogramKernel(const NEHistogramKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEHistogramKernel &operator=(const NEHistogramKernel &) = delete;
    /** Allow instances of this class to be moved */
    NEHistogramKernel(NEHistogramKernel &&) = default;
    /** Allow instances of this class to be moved */
    NEHistogramKernel &operator=(NEHistogramKernel &&) = default;

    /** Set the input image and the distribution output.
     *
     * @param[in]     input      Source image. Data type supported: U8.
     * @param[out]    output     Destination distribution.
     * @param[in,out] local_hist Array that the threads use to save their local histograms.
     *                           It's size should be equal to (number_of_threads * num_bins),
     *                           and the Window::thread_id() is used to determine the part of the array
     *                           used by each thread.
     * @param[out]    window_lut LUT with pre-calculated possible window values.
     *                           The size of the LUT should be equal to max_range_size and it will be filled
     *                           during the configure stage, while it re-used in every run, therefore can be
     *                           safely shared among threads.
     */
    void configure(const IImage *input, IDistribution1D *output, uint32_t *local_hist, uint32_t *window_lut);
    /** Set the input image and the distribution output.
     *
     * @note Used for histogram of fixed size equal to 256
     *
     * @param[in]  input  Source image. Data type supported: U8.
     * @param[out] output Destination distribution which must be of 256 bins..
     */
    void configure(const IImage *input, IDistribution1D *output);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;

private:
    /** Function to merge multiple partial histograms.
     *
     * @param[out] global_hist Pointer to the final histogram.
     * @param[in]  local_hist  Pointer to the partial histograms.
     * @param[in]  bins        Number of bins.
     */
    void merge_histogram(uint32_t *global_hist, const uint32_t *local_hist, size_t bins);
    /** Function to merge multiple minimum values of partial histograms.
     *
     * @param[out] global_min Pointer to the global min value.
     * @param[in]  local_min  Local min value.
     */
    void merge_min(uint8_t *global_min, const uint8_t &local_min);
    /** Function to perform histogram on the given window
     *
     * @param[in] win  Region on which to execute the kernel
     * @param[in] info Info about the executing thread
     */
    void histogram_U8(Window win, const ThreadInfo &info);
    /** Function to perform histogram on the given window where histogram is
     *         of fixed size 256 without ranges and offsets.
     *
     * @param[in] win  Region on which to execute the kernel
     * @param[in] info Info about the executing thread
     */
    void histogram_fixed_U8(Window win, const ThreadInfo &info);
    /** Pre-calculate the pixel windowing for every possible pixel
     *
     * Calculate (V - offset) * numBins / range where V is every possible pixel value.
     *
     * @note We currently support U8 image thus possible pixel values are between 0 and 255
     */
    void calculate_window_lut() const;
    /** Common signature for all the specialised Histogram functions
     *
     * @param[in] window Region on which to execute the kernel.
     */
    using HistogramFunctionPtr = void (NEHistogramKernel::*)(Window window, const ThreadInfo &info);

    HistogramFunctionPtr          _func; ///< Histogram function to use for the particular image types passed to configure()
    const IImage                 *_input;
    IDistribution1D              *_output;
    uint32_t                     *_local_hist;
    uint32_t                     *_window_lut;
    arm_compute::Mutex            _hist_mtx;
    static constexpr unsigned int _max_range_size{ 256 }; ///< 256 possible pixel values as we handle only U8 images
};
} // namespace arm_compute
#endif /*__ARM_COMPUTE_NEHISTOGRAMKERNEL_H__ */
