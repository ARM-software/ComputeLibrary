/*
 * Copyright (c) 2017-2018 ARM Limited.
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
#ifndef ARM_COMPUTE_TEST_CHANNEL_COMBINE_FIXTURE
#define ARM_COMPUTE_TEST_CHANNEL_COMBINE_FIXTURE

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/IAccessor.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/Helpers.h"
#include "tests/validation/reference/ChannelCombine.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
template <typename TensorType>
inline std::vector<TensorType> create_tensor_planes(const TensorShape &shape, Format format)
{
    TensorShape image_shape = adjust_odd_shape(shape, format);
    TensorInfo  info(image_shape, Format::U8);

    std::vector<TensorType> tensor_planes;

    switch(format)
    {
        case Format::RGB888:
        case Format::RGBA8888:
        case Format::YUV444:
        {
            tensor_planes.resize(3);

            if(format == Format::RGBA8888)
            {
                tensor_planes.resize(4);
            }

            for(unsigned int plane_idx = 0; plane_idx < tensor_planes.size(); ++plane_idx)
            {
                tensor_planes[plane_idx].allocator()->init(info);
            }

            break;
        }
        case Format::YUYV422:
        case Format::UYVY422:
        {
            const TensorShape uv_shape = calculate_subsampled_shape(image_shape, format);
            const TensorInfo  info_hor2(uv_shape, Format::U8);

            tensor_planes.resize(3);

            tensor_planes[0].allocator()->init(info);
            tensor_planes[1].allocator()->init(info_hor2);
            tensor_planes[2].allocator()->init(info_hor2);
            break;
        }
        case Format::NV12:
        case Format::NV21:
        case Format::IYUV:
        {
            const TensorShape sub2_shape = calculate_subsampled_shape(image_shape, format);
            const TensorInfo  info_sub2(sub2_shape, Format::U8);

            tensor_planes.resize(3);

            tensor_planes[0].allocator()->init(info);
            tensor_planes[1].allocator()->init(info_sub2);
            tensor_planes[2].allocator()->init(info_sub2);
            break;
        }
        default:
            ARM_COMPUTE_ERROR("Not supported");
            break;
    }

    return tensor_planes;
}
} // namespace

template <typename MultiImageType, typename TensorType, typename AccessorType, typename FunctionType, typename T>
class ChannelCombineValidationFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(TensorShape shape, Format format)
    {
        _num_planes = num_planes_from_format(format);
        _target     = compute_target(shape, format);
        _reference  = compute_reference(shape, format);
    }

protected:
    template <typename U>
    void fill(U &&tensor, int i)
    {
        library->fill_tensor_uniform(tensor, i);
    }

    template <typename U>
    std::vector<SimpleTensor<U>> create_tensor_planes_reference(const TensorShape &shape, Format format)
    {
        std::vector<SimpleTensor<U>> tensor_planes;

        TensorShape image_shape = adjust_odd_shape(shape, format);

        switch(format)
        {
            case Format::RGB888:
            case Format::RGBA8888:
            case Format::YUV444:
            {
                if(format == Format::RGBA8888)
                {
                    tensor_planes.emplace_back(image_shape, Format::U8);
                }

                tensor_planes.emplace_back(image_shape, Format::U8);
                tensor_planes.emplace_back(image_shape, Format::U8);
                tensor_planes.emplace_back(image_shape, Format::U8);
                break;
            }
            case Format::YUYV422:
            case Format::UYVY422:
            {
                const TensorShape hor2_shape = calculate_subsampled_shape(image_shape, format);

                tensor_planes.emplace_back(image_shape, Format::U8);
                tensor_planes.emplace_back(hor2_shape, Format::U8);
                tensor_planes.emplace_back(hor2_shape, Format::U8);
                break;
            }
            case Format::NV12:
            case Format::NV21:
            case Format::IYUV:
            {
                const TensorShape shape_sub2 = calculate_subsampled_shape(image_shape, format);

                tensor_planes.emplace_back(image_shape, Format::U8);
                tensor_planes.emplace_back(shape_sub2, Format::U8);
                tensor_planes.emplace_back(shape_sub2, Format::U8);
                break;
            }
            default:
                ARM_COMPUTE_ERROR("Not supported");
                break;
        }

        return tensor_planes;
    }

    MultiImageType compute_target(const TensorShape &shape, Format format)
    {
        // Create tensors
        std::vector<TensorType> ref_src = create_tensor_planes<TensorType>(shape, format);
        MultiImageType          dst     = create_multi_image<MultiImageType>(shape, format);

        // Create and configure function
        FunctionType channel_combine;

        if(1 == _num_planes)
        {
            const TensorType *tensor_extra = ((Format::RGBA8888 == format) ? &ref_src[3] : nullptr);
            TensorType       *tensor_dst   = dynamic_cast<TensorType *>(dst.plane(0));

            channel_combine.configure(&ref_src[0], &ref_src[1], &ref_src[2], tensor_extra, tensor_dst);
        }
        else
        {
            channel_combine.configure(&ref_src[0], &ref_src[1], &ref_src[2], &dst);
        }

        for(unsigned int plane_idx = 0; plane_idx < _num_planes; ++plane_idx)
        {
            const TensorType *dst_plane = static_cast<const TensorType *>(dst.plane(plane_idx));

            ARM_COMPUTE_EXPECT(dst_plane->info()->is_resizable(), framework::LogLevel::ERRORS);
        }

        for(unsigned int plane_idx = 0; plane_idx < ref_src.size(); ++plane_idx)
        {
            ARM_COMPUTE_EXPECT(ref_src[plane_idx].info()->is_resizable(), framework::LogLevel::ERRORS);
        }

        // Allocate tensors
        dst.allocate();

        for(unsigned int plane_idx = 0; plane_idx < ref_src.size(); ++plane_idx)
        {
            ref_src[plane_idx].allocator()->allocate();
        }

        for(unsigned int plane_idx = 0; plane_idx < _num_planes; ++plane_idx)
        {
            const TensorType *dst_plane = static_cast<const TensorType *>(dst.plane(plane_idx));

            ARM_COMPUTE_EXPECT(!dst_plane->info()->is_resizable(), framework::LogLevel::ERRORS);
        }

        for(unsigned int plane_idx = 0; plane_idx < ref_src.size(); ++plane_idx)
        {
            ARM_COMPUTE_EXPECT(!ref_src[plane_idx].info()->is_resizable(), framework::LogLevel::ERRORS);
        }

        // Fill tensor planes
        for(unsigned int plane_idx = 0; plane_idx < ref_src.size(); ++plane_idx)
        {
            fill(AccessorType(ref_src[plane_idx]), plane_idx);
        }

        // Compute function
        channel_combine.run();

        return dst;
    }

    std::vector<SimpleTensor<T>> compute_reference(const TensorShape &shape, Format format)
    {
        // Create reference
        std::vector<SimpleTensor<T>> ref_src = create_tensor_planes_reference<T>(shape, format);

        // Fill references
        for(unsigned int plane_idx = 0; plane_idx < ref_src.size(); ++plane_idx)
        {
            fill(ref_src[plane_idx], plane_idx);
        }

        return reference::channel_combine<T>(shape, ref_src, format);
    }

    unsigned int                 _num_planes{};
    MultiImageType               _target{};
    std::vector<SimpleTensor<T>> _reference{};
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_CHANNEL_COMBINE_FIXTURE */
