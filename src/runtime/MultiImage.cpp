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
#include "arm_compute/runtime/MultiImage.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/runtime/TensorAllocator.h"

using namespace arm_compute;

MultiImage::MultiImage()
    : _info(), _plane()
{
}

const MultiImageInfo *MultiImage::info() const
{
    return &_info;
}

void MultiImage::init(unsigned int width, unsigned int height, Format format)
{
    internal_init(width, height, format, false);
}

void MultiImage::init_auto_padding(unsigned int width, unsigned int height, Format format)
{
    internal_init(width, height, format, true);
}

void MultiImage::internal_init(unsigned int width, unsigned int height, Format format, bool auto_padding)
{
    TensorInfo info(width, height, Format::U8);

    if(auto_padding)
    {
        info.auto_padding();
    }

    switch(format)
    {
        case Format::U8:
        case Format::S16:
        case Format::U16:
        case Format::S32:
        case Format::F16:
        case Format::F32:
        case Format::U32:
        case Format::RGB888:
        case Format::RGBA8888:
        case Format::YUYV422:
        case Format::UYVY422:
        {
            TensorInfo info_full(width, height, format);

            if(auto_padding)
            {
                info_full.auto_padding();
            }

            std::get<0>(_plane).allocator()->init(info_full);
            break;
        }
        case Format::NV12:
        case Format::NV21:
        {
            TensorInfo info_uv88(width / 2, height / 2, Format::UV88);

            if(auto_padding)
            {
                info_uv88.auto_padding();
            }

            std::get<0>(_plane).allocator()->init(info);
            std::get<1>(_plane).allocator()->init(info_uv88);
            break;
        }
        case Format::IYUV:
        {
            TensorInfo info_sub2(width / 2, height / 2, Format::U8);

            if(auto_padding)
            {
                info_sub2.auto_padding();
            }

            std::get<0>(_plane).allocator()->init(info);
            std::get<1>(_plane).allocator()->init(info_sub2);
            std::get<2>(_plane).allocator()->init(info_sub2);
            break;
        }
        case Format::YUV444:
            std::get<0>(_plane).allocator()->init(info);
            std::get<1>(_plane).allocator()->init(info);
            std::get<2>(_plane).allocator()->init(info);
            break;
        default:
            ARM_COMPUTE_ERROR("Not supported");
            break;
    }

    _info.init(width, height, format);
}

void MultiImage::allocate()
{
    switch(_info.format())
    {
        case Format::U8:
        case Format::S16:
        case Format::U16:
        case Format::S32:
        case Format::F16:
        case Format::F32:
        case Format::U32:
        case Format::RGB888:
        case Format::RGBA8888:
        case Format::YUYV422:
        case Format::UYVY422:
            std::get<0>(_plane).allocator()->allocate();
            break;
        case Format::NV12:
        case Format::NV21:
            std::get<0>(_plane).allocator()->allocate();
            std::get<1>(_plane).allocator()->allocate();
            break;
        case Format::IYUV:
        case Format::YUV444:
            std::get<0>(_plane).allocator()->allocate();
            std::get<1>(_plane).allocator()->allocate();
            std::get<2>(_plane).allocator()->allocate();
            break;
        default:
            ARM_COMPUTE_ERROR("Not supported");
            break;
    }
}

void MultiImage::create_subimage(MultiImage *image, const Coordinates &coords, unsigned int width, unsigned int height)
{
    arm_compute::Format format = image->info()->format();
    const TensorInfo    info(width, height, Format::U8);

    switch(format)
    {
        case Format::U8:
        case Format::S16:
        case Format::U16:
        case Format::S32:
        case Format::F32:
        case Format::F16:
        case Format::U32:
        case Format::RGB888:
        case Format::RGBA8888:
        case Format::YUYV422:
        case Format::UYVY422:
        {
            const TensorInfo info_full(width, height, format);
            std::get<0>(_plane).allocator()->init(*dynamic_cast<Image *>(image->plane(0))->allocator(), coords, info_full);
            break;
        }
        case Format::NV12:
        case Format::NV21:
        {
            const TensorInfo info_uv88(width / 2, height / 2, Format::UV88);
            std::get<0>(_plane).allocator()->init(*dynamic_cast<Image *>(image->plane(0))->allocator(), coords, info);
            std::get<1>(_plane).allocator()->init(*dynamic_cast<Image *>(image->plane(1))->allocator(), coords, info_uv88);
            break;
        }
        case Format::IYUV:
        {
            const TensorInfo info_sub2(width / 2, height / 2, Format::U8);
            std::get<0>(_plane).allocator()->init(*dynamic_cast<Image *>(image->plane(0))->allocator(), coords, info);
            std::get<1>(_plane).allocator()->init(*dynamic_cast<Image *>(image->plane(1))->allocator(), coords, info_sub2);
            std::get<2>(_plane).allocator()->init(*dynamic_cast<Image *>(image->plane(2))->allocator(), coords, info_sub2);
            break;
        }
        case Format::YUV444:
            std::get<0>(_plane).allocator()->init(*dynamic_cast<Image *>(image->plane(0))->allocator(), coords, info);
            std::get<1>(_plane).allocator()->init(*dynamic_cast<Image *>(image->plane(0))->allocator(), coords, info);
            std::get<2>(_plane).allocator()->init(*dynamic_cast<Image *>(image->plane(0))->allocator(), coords, info);
            break;
        default:
            ARM_COMPUTE_ERROR("Not supported");
            break;
    }

    _info.init(width, height, format);
}

Image *MultiImage::plane(unsigned int index)
{
    return &_plane[index];
}

const Image *MultiImage::plane(unsigned int index) const
{
    return &_plane[index];
}
