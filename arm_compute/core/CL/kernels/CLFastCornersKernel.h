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
#ifndef __ARM_COMPUTE_CLFASTCORNERSKERNEL_H__
#define __ARM_COMPUTE_CLFASTCORNERSKERNEL_H__

#include "arm_compute/core/CL/ICLArray.h"
#include "arm_compute/core/CL/ICLKernel.h"
#include "arm_compute/core/Types.h"

#include <cstdint>

namespace cl
{
class Buffer;
}

namespace arm_compute
{
class ICLTensor;
using ICLImage = ICLTensor;

/** CL kernel to perform fast corners */
class CLFastCornersKernel : public ICLKernel
{
public:
    /** Default constructor */
    CLFastCornersKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLFastCornersKernel(const CLFastCornersKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLFastCornersKernel &operator=(const CLFastCornersKernel &) = delete;
    /** Allow instances of this class to be moved */
    CLFastCornersKernel(CLFastCornersKernel &&) = default;
    /** Allow instances of this class to be moved */
    CLFastCornersKernel &operator=(CLFastCornersKernel &&) = default;
    /** Default destructor */
    ~CLFastCornersKernel() = default;

    /** Initialise the kernel.
     *
     * @param[in]  input               Source image. Data types supported: U8.
     * @param[out] output              Output image. Data types supported: U8.
     * @param[in]  threshold           Threshold on difference between intensity of the central pixel and pixels on Bresenham's circle of radius 3.
     * @param[in]  non_max_suppression True if non-maxima suppresion is applied, false otherwise.
     * @param[in]  border_mode         Strategy to use for borders.
     */
    void configure(const ICLImage *input, ICLImage *output, float threshold, bool non_max_suppression, BorderMode border_mode);

    // Inherited methods overridden
    void run(const Window &window, cl::CommandQueue &queue) override;
    BorderSize border_size() const override;

private:
    const ICLImage *_input;
    ICLImage       *_output;
};

/** CL kernel to copy keypoints information to ICLKeyPointArray and counts the number of key points */
class CLCopyToArrayKernel : public ICLKernel
{
public:
    /** Default constructor */
    CLCopyToArrayKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLCopyToArrayKernel(const CLCopyToArrayKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLCopyToArrayKernel &operator=(const CLCopyToArrayKernel &) = delete;
    /** Allow instances of this class to be moved */
    CLCopyToArrayKernel(CLCopyToArrayKernel &&) = default;
    /** Allow instances of this class to be moved */
    CLCopyToArrayKernel &operator=(CLCopyToArrayKernel &&) = default;
    /** Default destructor */
    ~CLCopyToArrayKernel() = default;

    /** Initialise the kernel.
     *
     * @param[in]  input         Source image. Data types supported: U8.
     * @param[in]  update_number Flag to indicate whether we need to update the number of corners
     * @param[out] corners       Array of keypoints to store the results.
     * @param[out] num_buffers   Number of keypoints to store the results.
     */
    void configure(const ICLImage *input, bool update_number, ICLKeyPointArray *corners, cl::Buffer *num_buffers);

    // Inherited methods overridden:
    void run(const Window &window, cl::CommandQueue &queue) override;

private:
    const ICLImage   *_input;      /**< source image */
    ICLKeyPointArray *_corners;    /**< destination array */
    cl::Buffer       *_num_buffer; /**< CL memory to record number of key points in the array */
};
} // namespace arm_compute
#endif /* __ARM_COMPUTE_CLFASTCORNERSKERNEL_H__ */
