/*
 * Copyright (c) 2017 ARM Limited.
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
#ifndef __ARM_COMPUTE_CLHOGDESCRIPTORKERNEL_H__
#define __ARM_COMPUTE_CLHOGDESCRIPTORKERNEL_H__

#include "arm_compute/core/CL/ICLKernel.h"
#include "arm_compute/core/IHOG.h"
#include "arm_compute/core/Size2D.h"

namespace arm_compute
{
class ITensor;

/** OpenCL kernel to perform HOG Orientation Binning */
class CLHOGOrientationBinningKernel : public ICLKernel
{
public:
    /** Default constructor */
    CLHOGOrientationBinningKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLHOGOrientationBinningKernel(const CLHOGOrientationBinningKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLHOGOrientationBinningKernel &operator=(const CLHOGOrientationBinningKernel &) = delete;
    /** Allow instances of this class to be moved */
    CLHOGOrientationBinningKernel(CLHOGOrientationBinningKernel &&) = default;
    /** Allow instances of this class to be moved */
    CLHOGOrientationBinningKernel &operator=(CLHOGOrientationBinningKernel &&) = default;
    /** Default destructor */
    ~CLHOGOrientationBinningKernel() = default;

    /**  Initialise the kernel's inputs, output and HOG's metadata
     *
     * @param[in]  input_magnitude Input tensor which stores the magnitude of the gradient for each pixel. Data type supported: S16.
     * @param[in]  input_phase     Input tensor which stores the phase of the gradient for each pixel. Data type supported: U8
     * @param[out] output          Output tensor which stores the local HOG for each cell. DataType supported: F32. Number of channels supported: equal to the number of histogram bins per cell
     * @param[in]  hog_info        HOG's metadata
     */
    void configure(const ICLTensor *input_magnitude, const ICLTensor *input_phase, ICLTensor *output, const HOGInfo *hog_info);

    // Inherited methods overridden:
    void run(const Window &window, cl::CommandQueue &queue) override;

private:
    const ICLTensor *_input_magnitude;
    const ICLTensor *_input_phase;
    ICLTensor       *_output;
    Size2D           _cell_size;
};

/** OpenCL kernel to perform HOG block normalization */
class CLHOGBlockNormalizationKernel : public ICLKernel
{
public:
    /** Default constructor */
    CLHOGBlockNormalizationKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLHOGBlockNormalizationKernel(const CLHOGBlockNormalizationKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLHOGBlockNormalizationKernel &operator=(const CLHOGBlockNormalizationKernel &) = delete;
    /** Allow instances of this class to be moved */
    CLHOGBlockNormalizationKernel(CLHOGBlockNormalizationKernel &&) = default;
    /** Allow instances of this class to be moved */
    CLHOGBlockNormalizationKernel &operator=(CLHOGBlockNormalizationKernel &&) = default;
    /** Default destructor */
    ~CLHOGBlockNormalizationKernel() = default;

    /** Initialise the kernel's input, output and HOG's metadata
     *
     * @param[in]  input    Input tensor which stores the local HOG for each cell. Data type supported: F32. Number of channels supported: equal to the number of histogram bins per cell
     * @param[out] output   Output tensor which stores the normalised blocks. Data type supported: F32. Number of channels supported: equal to the number of histogram bins per block
     * @param[in]  hog_info HOG's metadata
     */
    void configure(const ICLTensor *input, ICLTensor *output, const HOGInfo *hog_info);

    // Inherited methods overridden:
    void run(const Window &window, cl::CommandQueue &queue) override;

private:
    const ICLTensor *_input;
    ICLTensor       *_output;
    Size2D           _num_cells_per_block_stride;
};
} // namespace arm_compute
#endif /* __ARM_COMPUTE_CLHOGDESCRIPTORKERNEL_H__ */
