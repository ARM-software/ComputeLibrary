/*
 * Copyright (c) 2016-2023 Arm Limited.
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
#ifndef ARM_COMPUTE_CORE_UTILS_FORMATUTILS_H
#define ARM_COMPUTE_CORE_UTILS_FORMATUTILS_H

#include "arm_compute/core/Types.h"

namespace arm_compute
{
/** The size in bytes of the pixel format
 *
 * @param[in] format Input format
 *
 * @return The size in bytes of the pixel format
 */
inline size_t pixel_size_from_format(Format format)
{
    switch(format)
    {
        case Format::U8:
            return 1;
        case Format::U16:
        case Format::S16:
        case Format::BFLOAT16:
        case Format::F16:
        case Format::UV88:
        case Format::YUYV422:
        case Format::UYVY422:
            return 2;
        case Format::RGB888:
            return 3;
        case Format::RGBA8888:
            return 4;
        case Format::U32:
        case Format::S32:
        case Format::F32:
            return 4;
        //Doesn't make sense for planar formats:
        case Format::NV12:
        case Format::NV21:
        case Format::IYUV:
        case Format::YUV444:
        default:
            ARM_COMPUTE_ERROR("Undefined pixel size for given format");
            return 0;
    }
}

/** Return the plane index of a given channel given an input format.
 *
 * @param[in] format  Input format
 * @param[in] channel Input channel
 *
 * @return The plane index of the specific channel of the specific format
 */
inline int plane_idx_from_channel(Format format, Channel channel)
{
    switch(format)
    {
        // Single planar formats have a single plane
        case Format::U8:
        case Format::U16:
        case Format::S16:
        case Format::U32:
        case Format::S32:
        case Format::BFLOAT16:
        case Format::F16:
        case Format::F32:
        case Format::UV88:
        case Format::RGB888:
        case Format::RGBA8888:
        case Format::YUYV422:
        case Format::UYVY422:
            return 0;
        // Multi planar formats
        case Format::NV12:
        case Format::NV21:
        {
            // Channel U and V share the same plane of format UV88
            switch(channel)
            {
                case Channel::Y:
                    return 0;
                case Channel::U:
                case Channel::V:
                    return 1;
                default:
                    ARM_COMPUTE_ERROR("Not supported channel");
                    return 0;
            }
        }
        case Format::IYUV:
        case Format::YUV444:
        {
            switch(channel)
            {
                case Channel::Y:
                    return 0;
                case Channel::U:
                    return 1;
                case Channel::V:
                    return 2;
                default:
                    ARM_COMPUTE_ERROR("Not supported channel");
                    return 0;
            }
        }
        default:
            ARM_COMPUTE_ERROR("Not supported format");
            return 0;
    }
}

/** Return the channel index of a given channel given an input format.
 *
 * @param[in] format  Input format
 * @param[in] channel Input channel
 *
 * @return The channel index of the specific channel of the specific format
 */
inline int channel_idx_from_format(Format format, Channel channel)
{
    switch(format)
    {
        case Format::RGB888:
        {
            switch(channel)
            {
                case Channel::R:
                    return 0;
                case Channel::G:
                    return 1;
                case Channel::B:
                    return 2;
                default:
                    ARM_COMPUTE_ERROR("Not supported channel");
                    return 0;
            }
        }
        case Format::RGBA8888:
        {
            switch(channel)
            {
                case Channel::R:
                    return 0;
                case Channel::G:
                    return 1;
                case Channel::B:
                    return 2;
                case Channel::A:
                    return 3;
                default:
                    ARM_COMPUTE_ERROR("Not supported channel");
                    return 0;
            }
        }
        case Format::YUYV422:
        {
            switch(channel)
            {
                case Channel::Y:
                    return 0;
                case Channel::U:
                    return 1;
                case Channel::V:
                    return 3;
                default:
                    ARM_COMPUTE_ERROR("Not supported channel");
                    return 0;
            }
        }
        case Format::UYVY422:
        {
            switch(channel)
            {
                case Channel::Y:
                    return 1;
                case Channel::U:
                    return 0;
                case Channel::V:
                    return 2;
                default:
                    ARM_COMPUTE_ERROR("Not supported channel");
                    return 0;
            }
        }
        case Format::NV12:
        {
            switch(channel)
            {
                case Channel::Y:
                    return 0;
                case Channel::U:
                    return 0;
                case Channel::V:
                    return 1;
                default:
                    ARM_COMPUTE_ERROR("Not supported channel");
                    return 0;
            }
        }
        case Format::NV21:
        {
            switch(channel)
            {
                case Channel::Y:
                    return 0;
                case Channel::U:
                    return 1;
                case Channel::V:
                    return 0;
                default:
                    ARM_COMPUTE_ERROR("Not supported channel");
                    return 0;
            }
        }
        case Format::YUV444:
        case Format::IYUV:
        {
            switch(channel)
            {
                case Channel::Y:
                    return 0;
                case Channel::U:
                    return 0;
                case Channel::V:
                    return 0;
                default:
                    ARM_COMPUTE_ERROR("Not supported channel");
                    return 0;
            }
        }
        default:
            ARM_COMPUTE_ERROR("Not supported format");
            return 0;
    }
}

/** Return the number of planes for a given format
 *
 * @param[in] format Input format
 *
 * @return The number of planes for a given image format.
 */
inline size_t num_planes_from_format(Format format)
{
    switch(format)
    {
        case Format::U8:
        case Format::S16:
        case Format::U16:
        case Format::S32:
        case Format::U32:
        case Format::BFLOAT16:
        case Format::F16:
        case Format::F32:
        case Format::RGB888:
        case Format::RGBA8888:
        case Format::YUYV422:
        case Format::UYVY422:
            return 1;
        case Format::NV12:
        case Format::NV21:
            return 2;
        case Format::IYUV:
        case Format::YUV444:
            return 3;
        default:
            ARM_COMPUTE_ERROR("Not supported format");
            return 0;
    }
}

/** Return the number of channels for a given single-planar pixel format
 *
 * @param[in] format Input format
 *
 * @return The number of channels for a given image format.
 */
inline size_t num_channels_from_format(Format format)
{
    switch(format)
    {
        case Format::U8:
        case Format::U16:
        case Format::S16:
        case Format::U32:
        case Format::S32:
        case Format::BFLOAT16:
        case Format::F16:
        case Format::F32:
            return 1;
        // Because the U and V channels are subsampled
        // these formats appear like having only 2 channels:
        case Format::YUYV422:
        case Format::UYVY422:
            return 2;
        case Format::UV88:
            return 2;
        case Format::RGB888:
            return 3;
        case Format::RGBA8888:
            return 4;
        //Doesn't make sense for planar formats:
        case Format::NV12:
        case Format::NV21:
        case Format::IYUV:
        case Format::YUV444:
        default:
            return 0;
    }
}

/** Convert a tensor format into a string.
 *
 * @param[in] format @ref Format to be translated to string.
 *
 * @return The string describing the format.
 */
const std::string &string_from_format(Format format);

}
#endif /*ARM_COMPUTE_CORE_UTILS_FORMATUTILS_H */
