/*
 * Copyright (c) 2016-2021 Arm Limited.
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
#ifndef ARM_COMPUTE_NEHARRISCORNERSKERNEL_H
#define ARM_COMPUTE_NEHARRISCORNERSKERNEL_H

#include "arm_compute/core/CPP/kernels/CPPCornerCandidatesKernel.h"
#include "arm_compute/core/CPP/kernels/CPPSortEuclideanDistanceKernel.h"
#include "arm_compute/core/IArray.h"
#include "src/core/NEON/INEKernel.h"

#include <cstdint>

namespace arm_compute
{
class ITensor;
using IImage = ITensor;

/** Common interface for all Harris Score kernels */
class INEHarrisScoreKernel : public INEKernel
{
public:
    /** Default constructor */
    INEHarrisScoreKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    INEHarrisScoreKernel(const INEHarrisScoreKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    INEHarrisScoreKernel &operator=(const INEHarrisScoreKernel &) = delete;
    /** Allow instances of this class to be moved */
    INEHarrisScoreKernel(INEHarrisScoreKernel &&) = default;
    /** Allow instances of this class to be moved */
    INEHarrisScoreKernel &operator=(INEHarrisScoreKernel &&) = default;
    /** Default destructor */
    ~INEHarrisScoreKernel() = default;

public:
    /** Setup the kernel parameters
     *
     * @param[in]  input1           Source image (gradient X). Data types supported: S16/S32
     * @param[in]  input2           Source image (gradient Y). Data types supported: same as @ input1
     * @param[out] output           Destination image (harris score). Data types supported: F32
     * @param[in]  norm_factor      Normalization factor to use accordingly with the gradient size (Must be different from 0)
     * @param[in]  strength_thresh  Minimum threshold with which to eliminate Harris Corner scores (computed using the normalized Sobel kernel).
     * @param[in]  sensitivity      Sensitivity threshold k from the Harris-Stephens equation
     * @param[in]  border_undefined True if the border mode is undefined. False if it's replicate or constant.
     */
    virtual void configure(const IImage *input1, const IImage *input2, IImage *output, float norm_factor, float strength_thresh, float sensitivity, bool border_undefined) = 0;

protected:
    const IImage *_input1;          /**< Source image - Gx component */
    const IImage *_input2;          /**< Source image - Gy component */
    IImage       *_output;          /**< Source image - Harris score */
    float         _sensitivity;     /**< Sensitivity value */
    float         _strength_thresh; /**< Threshold value */
    float         _norm_factor;     /**< Normalization factor */
    BorderSize    _border_size;     /**< Border size */
};

/** Template Neon kernel to perform Harris Score.
 *  The implementation supports 3, 5, and 7 for the block_size
 */
template <int32_t block_size>
class NEHarrisScoreKernel : public INEHarrisScoreKernel
{
public:
    const char *name() const override
    {
        return "NEHarrisScoreKernel";
    }
    /** Default constructor */
    NEHarrisScoreKernel();
    // Inherited methods overridden:
    void configure(const IImage *input1, const IImage *input2, IImage *output, float norm_factor, float strength_thresh, float sensitivity, bool border_undefined) override;
    BorderSize border_size() const override;
    void run(const Window &window, const ThreadInfo &info) override;

private:
    /** Common signature for all the specialised harris score functions */
    using HarrisScoreFunction = void(const void *__restrict input1_ptr, const void *__restrict input2_ptr, void *__restrict output_ptr, int32_t input_stride,
                                     float norm_factor, float sensitivity, float strength_thresh);
    /** Harris Score function to use for the particular image types passed to configure() */
    HarrisScoreFunction *_func;
};
} // namespace arm_compute
#endif /* ARM_COMPUTE_NEHARRISCORNERSKERNEL_H */
