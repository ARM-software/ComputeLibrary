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
#ifndef __ARM_COMPUTE_NECHANNELEXTRACTKERNEL_H__
#define __ARM_COMPUTE_NECHANNELEXTRACTKERNEL_H__

#include "arm_compute/core/NEON/INESimpleKernel.h"
#include "arm_compute/core/Types.h"

#include <cstdint>

namespace arm_compute
{
class IMultiImage;
class ITensor;
using IImage = ITensor;

/** Interface for the channel extract kernel */
class NEChannelExtractKernel : public INESimpleKernel
{
public:
    /** Default constructor */
    NEChannelExtractKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEChannelExtractKernel(const NEChannelExtractKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEChannelExtractKernel &operator=(const NEChannelExtractKernel &) = delete;
    /** Allow instances of this class to be moved */
    NEChannelExtractKernel(NEChannelExtractKernel &&) = default;
    /** Allow instances of this class to be moved */
    NEChannelExtractKernel &operator=(NEChannelExtractKernel &&) = default;
    /** Default destructor */
    ~NEChannelExtractKernel() = default;

    /** Set the input and output of the kernel
     *
     * @param[in]  input   Source tensor. Formats supported: RGB888/RGBA8888/YUYV422/UYVY422
     * @param[in]  channel Channel to extract.
     * @param[out] output  Destination tensor. Format supported: u8
     */
    void configure(const ITensor *input, Channel channel, ITensor *output);
    /** Set the input and output of the kernel
     *
     * @param[in]  input   Multi-planar source image. Formats supported: NV12/NV21/IYUV/YUV444
     * @param[in]  channel Channel to extract.
     * @param[out] output  Single-planar destination image. Format supported: U8
     */
    void configure(const IMultiImage *input, Channel channel, IImage *output);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;

private:
    /** Extract one channel from a two channel planar tensor.
     *
     * @param[in] win Region on which to execute the kernel.
     */
    void extract_1C_from_2C_img(const Window &win);
    /** Extract one channel from a three channel planar tensor.
     *
     * @param[in] win Region on which to execute the kernel.
     */
    void extract_1C_from_3C_img(const Window &win);
    /** Extract one channel from a four channel planar tensor.
     *
     * @param[in] win Region on which to execute the kernel.
     */
    void extract_1C_from_4C_img(const Window &win);
    /** Extract U/V channel from a single planar YUVY/UYVY tensor.
     *
     * @param[in] win Region on which to execute the kernel.
     */
    void extract_YUYV_uv(const Window &win);
    /** Copies a full plane to the output tensor.
     *
     * @param[in] win Region on which to execute the kernel.
     */
    void copy_plane(const Window &win);
    /** Common signature for all the specialised ChannelExtract functions
     *
     * @param[in] window Region on which to execute the kernel.
     */
    using ChannelExtractFunction = void (NEChannelExtractKernel::*)(const Window &window);
    /** ChannelExtract function to use for the particular tensor types passed to configure() */
    ChannelExtractFunction _func;
    unsigned int           _lut_index;
};
} // namespace arm_compute
#endif /* __ARM_COMPUTE_NECHANNELEXTRACTKERNEL_H__ */
