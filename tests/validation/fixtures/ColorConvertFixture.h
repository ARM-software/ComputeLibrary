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
#ifndef ARM_COMPUTE_TEST_COLOR_CONVERT_FIXTURE
#define ARM_COMPUTE_TEST_COLOR_CONVERT_FIXTURE

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/IAccessor.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/Helpers.h"
#include "tests/validation/reference/ColorConvert.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
}
template <typename MultiImageType, typename TensorType, typename AccessorType, typename FunctionType, typename T>
class ColorConvertValidationFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(TensorShape shape, Format src_format, Format dst_format)
    {
        shape = adjust_odd_shape(shape, src_format);
        shape = adjust_odd_shape(shape, dst_format);

        _target    = compute_target(shape, src_format, dst_format);
        _reference = compute_reference(shape, src_format, dst_format);
    }

protected:
    template <typename U>
    void fill(U &&tensor, int i)
    {
        library->fill_tensor_uniform(tensor, i);
    }

    std::vector<SimpleTensor<T>> create_tensor_planes_reference(const TensorShape &shape, Format format)
    {
        std::vector<SimpleTensor<T>> tensor_planes;

        switch(format)
        {
            case Format::RGB888:
            case Format::RGBA8888:
            case Format::YUYV422:
            case Format::UYVY422:
            {
                tensor_planes.emplace_back(shape, format);
                break;
            }
            case Format::NV12:
            case Format::NV21:
            {
                const TensorShape shape_uv88 = calculate_subsampled_shape(shape, Format::UV88);

                tensor_planes.emplace_back(shape, Format::U8);
                tensor_planes.emplace_back(shape_uv88, Format::UV88);
                break;
            }
            case Format::IYUV:
            {
                const TensorShape shape_sub2 = calculate_subsampled_shape(shape, Format::IYUV);

                tensor_planes.emplace_back(shape, Format::U8);
                tensor_planes.emplace_back(shape_sub2, Format::U8);
                tensor_planes.emplace_back(shape_sub2, Format::U8);
                break;
            }
            case Format::YUV444:
            {
                tensor_planes.emplace_back(shape, Format::U8);
                tensor_planes.emplace_back(shape, Format::U8);
                tensor_planes.emplace_back(shape, Format::U8);
                break;
            }
            default:
                ARM_COMPUTE_ERROR("Not supported");
                break;
        }

        return tensor_planes;
    }

    MultiImageType compute_target(const TensorShape &shape, Format src_format, Format dst_format)
    {
        _src_num_planes = num_planes_from_format(src_format);
        _dst_num_planes = num_planes_from_format(dst_format);

        // Create tensors
        MultiImageType ref_src = create_multi_image<MultiImageType>(shape, src_format);
        MultiImageType ref_dst = create_multi_image<MultiImageType>(shape, dst_format);

        // Create and configure function
        FunctionType color_convert;

        if(1U == _src_num_planes)
        {
            const TensorType *plane_src = static_cast<TensorType *>(ref_src.plane(0));

            if(1U == _dst_num_planes)
            {
                TensorType *dst_plane = static_cast<TensorType *>(ref_dst.plane(0));
                color_convert.configure(plane_src, dst_plane);
            }
            else
            {
                color_convert.configure(plane_src, &ref_dst);
            }
        }
        else
        {
            if(1U == _dst_num_planes)
            {
                TensorType *dst_plane = static_cast<TensorType *>(ref_dst.plane(0));
                color_convert.configure(&ref_src, dst_plane);
            }
            else
            {
                color_convert.configure(&ref_src, &ref_dst);
            }
        }

        for(unsigned int plane_idx = 0; plane_idx < _src_num_planes; ++plane_idx)
        {
            const TensorType *src_plane = static_cast<const TensorType *>(ref_src.plane(plane_idx));

            ARM_COMPUTE_EXPECT(src_plane->info()->is_resizable(), framework::LogLevel::ERRORS);
        }
        for(unsigned int plane_idx = 0; plane_idx < _dst_num_planes; ++plane_idx)
        {
            const TensorType *dst_plane = static_cast<const TensorType *>(ref_dst.plane(plane_idx));

            ARM_COMPUTE_EXPECT(dst_plane->info()->is_resizable(), framework::LogLevel::ERRORS);
        }

        // Allocate tensors
        ref_src.allocate();
        ref_dst.allocate();

        for(unsigned int plane_idx = 0; plane_idx < _src_num_planes; ++plane_idx)
        {
            const TensorType *src_plane = static_cast<const TensorType *>(ref_src.plane(plane_idx));
            ARM_COMPUTE_EXPECT(!src_plane->info()->is_resizable(), framework::LogLevel::ERRORS);
        }

        for(unsigned int plane_idx = 0; plane_idx < _dst_num_planes; ++plane_idx)
        {
            const TensorType *dst_plane = static_cast<const TensorType *>(ref_dst.plane(plane_idx));
            ARM_COMPUTE_EXPECT(!dst_plane->info()->is_resizable(), framework::LogLevel::ERRORS);
        }

        // Fill tensor planes
        for(unsigned int plane_idx = 0; plane_idx < _src_num_planes; ++plane_idx)
        {
            TensorType *src_plane = static_cast<TensorType *>(ref_src.plane(plane_idx));

            fill(AccessorType(*src_plane), plane_idx);
        }

        // Compute function
        color_convert.run();

        return ref_dst;
    }

    std::vector<SimpleTensor<T>> compute_reference(const TensorShape &shape, Format src_format, Format dst_format)
    {
        // Create reference
        std::vector<SimpleTensor<T>> ref_src = create_tensor_planes_reference(shape, src_format);

        // Fill references
        for(unsigned int plane_idx = 0; plane_idx < ref_src.size(); ++plane_idx)
        {
            fill(ref_src[plane_idx], plane_idx);
        }

        return reference::color_convert<T>(shape, ref_src, src_format, dst_format);
    }

    unsigned int                 _src_num_planes{};
    unsigned int                 _dst_num_planes{};
    MultiImageType               _target{};
    std::vector<SimpleTensor<T>> _reference{};
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_COLOR_CONVERT_FIXTURE */
