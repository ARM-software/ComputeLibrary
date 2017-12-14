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
#ifndef __ARM_COMPUTE_CLCHANNELEXTRACTKERNEL_H__
#define __ARM_COMPUTE_CLCHANNELEXTRACTKERNEL_H__

#include "arm_compute/core/CL/ICLKernel.h"
#include "arm_compute/core/Types.h"

#include <cstdint>

namespace arm_compute
{
class ICLMultiImage;
class ICLTensor;
using ICLImage = ICLTensor;

/** Interface for the channel extract kernel */
class CLChannelExtractKernel : public ICLKernel
{
public:
    /** Default constructor */
    CLChannelExtractKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLChannelExtractKernel(const CLChannelExtractKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLChannelExtractKernel &operator=(const CLChannelExtractKernel &) = delete;
    /** Allow instances of this class to be moved */
    CLChannelExtractKernel(CLChannelExtractKernel &&) = default;
    /** Allow instances of this class to be moved */
    CLChannelExtractKernel &operator=(CLChannelExtractKernel &&) = default;
    /** Default destructor */
    ~CLChannelExtractKernel() = default;
    /** Set the input and output of the kernel
     *
     * @param[in]  input   Source tensor. Formats supported: RGB888/RGBA8888/YUYV422/UYVY422
     * @param[in]  channel Channel to extract.
     * @param[out] output  Destination tensor. Must be of U8 format.
     */
    void configure(const ICLTensor *input, Channel channel, ICLTensor *output);
    /** Set the input and output of the kernel
     *
     * @param[in]  input   Multi-planar source image. Formats supported: NV12/NV21/IYUV/YUV444
     * @param[in]  channel Channel to extract.
     * @param[out] output  Single-planar 2D destination image. Must be of U8 format.
     */
    void configure(const ICLMultiImage *input, Channel channel, ICLImage *output);

    // Inherited methods overridden:
    void run(const Window &window, cl::CommandQueue &queue) override;

private:
    const ICLTensor *_input;
    ICLTensor       *_output;
    uint32_t         _num_elems_processed_per_iteration;
    uint32_t         _subsampling;
};
} // namespace arm_compute
#endif /* __ARM_COMPUTE_CLCHANNELEXTRACTKERNEL_H__ */
