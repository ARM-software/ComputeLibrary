/*
 * Copyright (c) 2019 ARM Limited.
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
#ifndef ARM_COMPUTE_TEST_SLICE_OPERATIONS_FIXTURE
#define ARM_COMPUTE_TEST_SLICE_OPERATIONS_FIXTURE

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"

#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/IAccessor.h"
#include "tests/RawLutAccessor.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/Helpers.h"
#include "tests/validation/reference/CropResize.h"
#include "tests/validation/reference/Permute.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class CropResizeFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(TensorShape src_shape, TensorShape boxes_shape, Coordinates2D crop_size, InterpolationPolicy method,
               float extrapolation_value, bool is_outside_bounds, DataType data_type)
    {
        _target    = compute_target(src_shape, boxes_shape, crop_size, method, extrapolation_value, is_outside_bounds, data_type);
        _reference = compute_reference(src_shape, boxes_shape, crop_size, method, extrapolation_value, is_outside_bounds, data_type);
    }

protected:
    template <typename U>
    void fill(U &&tensor, int i)
    {
        library->fill_tensor_uniform(tensor, i);
    }

    template <typename U, typename V>
    void fill(U &&tensor, int i, V min, V max)
    {
        library->fill_tensor_uniform(tensor, i, min, max);
    }

    TensorType compute_target(const TensorShape &src_shape, const TensorShape &boxes_shape, const Coordinates2D &crop_size, InterpolationPolicy method,
                              float extrapolation_value, bool is_outside_bounds, DataType data_type)
    {
        TensorShape dst_shape(src_shape[0], crop_size.x, crop_size.y, boxes_shape[1]);

        // Create tensors
        TensorType src       = create_tensor<TensorType>(src_shape, data_type, 1, QuantizationInfo(), DataLayout::NHWC);
        TensorType boxes     = create_tensor<TensorType>(boxes_shape, DataType::F32);
        TensorType boxes_ind = create_tensor<TensorType>(TensorShape(boxes_shape[1]), DataType::S32);
        TensorType dst       = create_tensor<TensorType>(dst_shape, DataType::F32, 1, QuantizationInfo(), DataLayout::NHWC);

        // Create and configure function
        FunctionType crop;
        crop.configure(&src, &boxes, &boxes_ind, &dst, crop_size, method, extrapolation_value);

        ARM_COMPUTE_EXPECT(src.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(boxes.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(boxes_ind.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(dst.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Allocate tensors
        src.allocator()->allocate();
        boxes.allocator()->allocate();
        boxes_ind.allocator()->allocate();
        dst.allocator()->allocate();

        ARM_COMPUTE_EXPECT(!src.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!boxes.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!boxes_ind.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!dst.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Fill tensors
        fill(AccessorType(src), 0);
        fill(AccessorType(boxes), 1, is_outside_bounds ? 0.0f - out_of_bounds_reach : 0.0f, is_outside_bounds ? 1.0f + out_of_bounds_reach : 1.0f);
        fill(AccessorType(boxes_ind), 2, 0, static_cast<int32_t>(src_shape[3] - 1));

        // Compute function
        crop.run();
        return dst;
    }

    SimpleTensor<float> compute_reference(const TensorShape &src_shape, const TensorShape &boxes_shape, const Coordinates2D &crop_size, InterpolationPolicy method,
                                          float extrapolation_value, bool is_outside_bounds, DataType data_type)
    {
        // Create reference
        SimpleTensor<T>       src{ src_shape, data_type, 1, QuantizationInfo(), DataLayout::NHWC };
        SimpleTensor<float>   boxes{ boxes_shape, DataType::F32 };
        SimpleTensor<int32_t> boxes_ind{ TensorShape(boxes_shape[1]), DataType::S32 };

        // Fill reference
        fill(src, 0);
        fill(boxes, 1, is_outside_bounds ? 0.0f - out_of_bounds_reach : 0.0f, is_outside_bounds ? 1.0f + out_of_bounds_reach : 1.0f);
        fill(boxes_ind, 2, 0, static_cast<int32_t>(src.shape()[3] - 1));

        SimpleTensor<float> output = reference::crop_and_resize(src, boxes, boxes_ind, crop_size, method, extrapolation_value);

        SimpleTensor<float> permuted = reference::permute(output, PermutationVector(1, 2U, 0U));
        return permuted;
    }

    constexpr static float out_of_bounds_reach = 2.0f;

    TensorType          _target{};
    SimpleTensor<float> _reference{};
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_SLICE_OPERATIONS_FIXTURE */
