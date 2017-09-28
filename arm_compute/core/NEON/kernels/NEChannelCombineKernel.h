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
#ifndef __ARM_COMPUTE_NECHANNELCOMBINEKERNEL_H__
#define __ARM_COMPUTE_NECHANNELCOMBINEKERNEL_H__

#include "arm_compute/core/NEON/INEKernel.h"

#include <array>
#include <cstdint>

namespace arm_compute
{
class IMultiImage;
class ITensor;
using IImage = ITensor;

/** Interface for the channel combine kernel */
class NEChannelCombineKernel : public INEKernel
{
public:
    /** Default constructor */
    NEChannelCombineKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEChannelCombineKernel(const NEChannelCombineKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEChannelCombineKernel &operator=(const NEChannelCombineKernel &) = delete;
    /** Allow instances of this class to be moved */
    NEChannelCombineKernel(NEChannelCombineKernel &&) = default;
    /** Allow instances of this class to be moved */
    NEChannelCombineKernel &operator=(NEChannelCombineKernel &&) = default;
    /** Default destructor */
    ~NEChannelCombineKernel() = default;

    /** Configure function's inputs and outputs.
     *
     * @param[in]  plane0 The 2D plane that forms channel 0. Data type supported: U8
     * @param[in]  plane1 The 2D plane that forms channel 1. Data type supported: U8
     * @param[in]  plane2 The 2D plane that forms channel 2. Data type supported: U8
     * @param[in]  plane3 The 2D plane that forms channel 3. Data type supported: U8
     * @param[out] output The single planar output tensor. Formats supported: RGB888/RGBA8888/UYVY422/YUYV422
     */
    void configure(const ITensor *plane0, const ITensor *plane1, const ITensor *plane2, const ITensor *plane3, ITensor *output);
    /** Configure function's inputs and outputs.
     *
     * @param[in]  plane0 The 2D plane that forms channel 0. Data type supported: U8
     * @param[in]  plane1 The 2D plane that forms channel 1. Data type supported: U8
     * @param[in]  plane2 The 2D plane that forms channel 2. Data type supported: U8
     * @param[out] output The multi planar output tensor. Formats supported: NV12/NV21/IYUV/YUV444
     */
    void configure(const IImage *plane0, const IImage *plane1, const IImage *plane2, IMultiImage *output);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;
    bool is_parallelisable() const override;

private:
    /** Combine 3 planes to form a three channel single plane tensor.
     *
     * @param[in] win Region on which to execute the kernel.
     */
    void combine_3C(const Window &win);
    /** Combine 4 planes to form a four channel single plane tensor.
     *
     * @param[in] win Region on which to execute the kernel.
     */
    void combine_4C(const Window &win);
    /** Combine 3 planes to form a single plane YUV tensor.
     *
     * @param[in] win Region on which to execute the kernel.
     */
    template <bool is_yuyv>
    void combine_YUV_1p(const Window &win);
    /** Combine 3 planes to form a two plane YUV tensor.
     *
     * @param[in] win Region on which to execute the kernel.
     */
    void combine_YUV_2p(const Window &win);
    /** Combine 3 planes to form a three plane YUV tensor.
     *
     * @param[in] win Region on which to execute the kernel.
     */
    void combine_YUV_3p(const Window &win);
    /** Copies a full plane to the output tensor.
     *
     * @param[in] win Region on which to execute the kernel.
     */
    void copy_plane(const Window &win, uint32_t plane_id);
    /** Common signature for all the specialised ChannelCombine functions
     *
     * @param[in] window Region on which to execute the kernel.
     */
    using ChannelCombineFunction = void (NEChannelCombineKernel::*)(const Window &window);
    /** ChannelCombine function to use for the particular tensor types passed to configure() */
    ChannelCombineFunction _func;
    std::array<const ITensor *, 4> _planes;
    ITensor     *_output;
    IMultiImage *_output_multi;
    std::array<uint32_t, 3> _x_subsampling;
    std::array<uint32_t, 3> _y_subsampling;
    unsigned int _num_elems_processed_per_iteration;
    bool         _is_parallelizable;
};
} // namespace arm_compute
#endif /* __ARM_COMPUTE_NECHANNELCOMBINEKERNEL_H__ */
