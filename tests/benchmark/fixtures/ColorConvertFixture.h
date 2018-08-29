/*
 * Copyright (c) 2018 ARM Limited.
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
#include "tests/Globals.h"
#include "tests/Utils.h"
#include "tests/framework/Fixture.h"

namespace arm_compute
{
namespace test
{
namespace benchmark
{
template <typename MultiImageType, typename TensorType, typename AccessorType, typename FunctionType>
class ColorConvertFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(TensorShape input_shape, Format src_format, Format dst_format)
    {
        _src_num_planes = num_planes_from_format(src_format);
        _dst_num_planes = num_planes_from_format(dst_format);

        TensorShape dst_shape = adjust_odd_shape(input_shape, src_format);
        dst_shape             = adjust_odd_shape(dst_shape, dst_format);

        // Create tensors
        ref_src = create_multi_image<MultiImageType>(dst_shape, src_format);
        ref_dst = create_multi_image<MultiImageType>(dst_shape, dst_format);

        if(1U == _src_num_planes)
        {
            const TensorType *plane_src = static_cast<TensorType *>(ref_src.plane(0));

            if(1U == _dst_num_planes)
            {
                TensorType *plane_dst = static_cast<TensorType *>(ref_dst.plane(0));
                colorconvert_func.configure(plane_src, plane_dst);
            }
            else
            {
                colorconvert_func.configure(plane_src, &ref_dst);
            }
        }
        else
        {
            if(1U == _dst_num_planes)
            {
                TensorType *plane_dst = static_cast<TensorType *>(ref_dst.plane(0));
                colorconvert_func.configure(&ref_src, plane_dst);
            }
            else
            {
                colorconvert_func.configure(&ref_src, &ref_dst);
            }
        }

        // Allocate tensors
        ref_src.allocate();
        ref_dst.allocate();

        // Fill tensor planes
        for(unsigned int plane_idx = 0; plane_idx < _src_num_planes; ++plane_idx)
        {
            TensorType *src_plane = static_cast<TensorType *>(ref_src.plane(plane_idx));

            fill(AccessorType(*src_plane), plane_idx);
        }
    }

    void run()
    {
        colorconvert_func.run();
    }

    void sync()
    {
        sync_if_necessary<TensorType>();
        for(unsigned int plane_idx = 0; plane_idx < _dst_num_planes; ++plane_idx)
        {
            TensorType *dst_plane = static_cast<TensorType *>(ref_dst.plane(plane_idx));
            sync_tensor_if_necessary<TensorType>(*dst_plane);
        }
    }

    void teardown()
    {
        for(unsigned int plane_idx = 0; plane_idx < _src_num_planes; ++plane_idx)
        {
            TensorType *src_plane = static_cast<TensorType *>(ref_src.plane(plane_idx));
            src_plane->allocator()->free();
        }
        for(unsigned int plane_idx = 0; plane_idx < _dst_num_planes; ++plane_idx)
        {
            TensorType *dst_plane = static_cast<TensorType *>(ref_dst.plane(plane_idx));
            dst_plane->allocator()->free();
        }
    }

protected:
    template <typename U>
    void fill(U &&tensor, int i)
    {
        library->fill_tensor_uniform(tensor, i);
    }

private:
    MultiImageType ref_src{};
    MultiImageType ref_dst{};
    FunctionType   colorconvert_func{};

    unsigned int _src_num_planes{};
    unsigned int _dst_num_planes{};
};
} // namespace benchmark
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_COLOR_CONVERT_FIXTURE */
