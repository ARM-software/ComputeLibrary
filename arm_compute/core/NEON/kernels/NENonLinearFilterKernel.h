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
#ifndef __ARM_COMPUTE_NENONLINEARFILTERKERNEL_H__
#define __ARM_COMPUTE_NENONLINEARFILTERKERNEL_H__

#include "arm_compute/core/NEON/INEKernel.h"
#include "arm_compute/core/Types.h"

#include <cstdint>

namespace arm_compute
{
class ITensor;

/** Interface for the kernel to apply a non-linear filter */
class NENonLinearFilterKernel : public INEKernel
{
public:
    /** Default constructor */
    NENonLinearFilterKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NENonLinearFilterKernel(NENonLinearFilterKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NENonLinearFilterKernel &operator=(NENonLinearFilterKernel &) = delete;
    /** Allow instances of this class to be moved */
    NENonLinearFilterKernel(NENonLinearFilterKernel &&) = default;
    /** Allow instances of this class to be moved */
    NENonLinearFilterKernel &operator=(NENonLinearFilterKernel &&) = default;
    /** Set the source, destination and border mode of the kernel
     *
     * @param[in]  input            Source tensor. Data type supported: U8
     * @param[out] output           Destination tensor. Data type supported: U8
     * @param[in]  function         Non linear function to perform
     * @param[in]  mask_size        Mask size. Supported sizes: 3, 5
     * @param[in]  pattern          Mask pattern
     * @param[in]  mask             The given mask. Will be used only if pattern is specified to PATTERN_OTHER
     * @param[in]  border_undefined True if the border mode is undefined. False if it's replicate or constant.
     */
    void configure(const ITensor *input, ITensor *output, NonLinearFilterFunction function, unsigned int mask_size, MatrixPattern pattern, const uint8_t *mask, bool border_undefined);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;
    BorderSize border_size() const override;

private:
    /** Fill mask with the corresponding given pattern.
     *
     * @param[in,out] mask    Mask to be filled according to pattern
     * @param[in]     cols    Columns (width) of mask
     * @param[in]     rows    Rows (height) of mask
     * @param[in]     pattern Pattern to fill the mask according to
     */
    void fill_mask(uint8_t *mask, int cols, int rows, MatrixPattern pattern);
    /** Apply a median filter when given mask pattern is defined as box.
     *
     * @param[in] win Window to apply the filter on.
     */
    template <int mask_w, int mask_h>
    void median_filter_box(const Window &win);
    /** Apply a min filter when given mask pattern is defined as box.
     *
     * @param[in] win Window to apply the filter on.
     */
    template <int mask_w, int mask_h>
    void min_filter_box(const Window &win);
    /** Apply a max filter when given mask pattern is defined as box.
     *
     * @param[in] win Window to apply the filter on.
     */
    template <int mask_w, int mask_h>
    void max_filter_box(const Window &win);
    /** Apply a median filter when given mask pattern is defined as cross.
     *
     * @param[in] win Window to apply the filter on.
     */
    template <int mask_w, int mask_h>
    void median_filter_cross(const Window &win);
    /** Apply a min filter when given mask pattern is defined as cross.
     *
     * @param[in] win Window to apply the filter on.
     */
    template <int mask_w, int mask_h>
    void min_filter_cross(const Window &win);
    /** Apply a max filter when given mask pattern is defined as cross.
     *
     * @param[in] win Window to apply the filter on.
     */
    template <int mask_w, int mask_h>
    void max_filter_cross(const Window &win);
    /** Apply a median filter when given mask pattern is defined as disk.
     *
     * @param[in] win Window to apply the filter on.
     */
    template <int mask_w, int mask_h>
    void median_filter_disk(const Window &win);
    /** Apply a min filter when given mask pattern is defined as disk.
     *
     * @param[in] win Window to apply the filter on.
     */
    template <int mask_w, int mask_h>
    void min_filter_disk(const Window &win);
    /** Apply a max filter when given mask pattern is defined as disk.
     *
     * @param[in] win Window to apply the filter on.
     */
    template <int mask_w, int mask_h>
    void max_filter_disk(const Window &win);
    /** Apply a non-linear filter when given mask has user-defined pattern.
     *
     * @param[in] win Window to apply the filter on.
     */
    template <int mask_w, int mask_h>
    void non_linear_filter_generic(const Window &win);

private:
    unsigned int            _border_width;
    const ITensor          *_input;
    ITensor                *_output;
    const uint8_t          *_mask;
    MatrixPattern           _pattern;
    NonLinearFilterFunction _function;
    unsigned int            _func_idx;
    BorderSize              _border_size;
};
} // namespace arm_compute
#endif /*__ARM_COMPUTE_NENONLINEARFILTERKERNEL_H__ */
