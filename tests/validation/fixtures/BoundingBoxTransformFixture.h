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
#ifndef ARM_COMPUTE_TEST_BOUNDINGBOXTRANSFORM_FIXTURE
#define ARM_COMPUTE_TEST_BOUNDINGBOXTRANSFORM_FIXTURE

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/IAccessor.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/Helpers.h"
#include "tests/validation/reference/BoundingBoxTransform.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class BoundingBoxTransformFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(TensorShape deltas_shape, const BoundingBoxTransformInfo &info, DataType data_type)
    {
        std::mt19937 gen_target(library->seed());
        _target = compute_target(deltas_shape, data_type, info, gen_target);

        std::mt19937 gen_reference(library->seed());
        _reference = compute_reference(deltas_shape, data_type, info, gen_reference);
    }

protected:
    TensorType compute_target(const TensorShape &deltas_shape, DataType data_type,
                              const BoundingBoxTransformInfo &bbox_info, std::mt19937 &gen)
    {
        // Create tensors
        TensorShape boxes_shape(4, deltas_shape[1]);
        TensorType  deltas = create_tensor<TensorType>(deltas_shape, data_type);
        TensorType  boxes  = create_tensor<TensorType>(boxes_shape, data_type);
        TensorType  pred_boxes;

        // Create and configure function
        FunctionType bbox_transform;
        bbox_transform.configure(&boxes, &pred_boxes, &deltas, bbox_info);

        ARM_COMPUTE_EXPECT(deltas.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(boxes.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(pred_boxes.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Allocate tensors
        deltas.allocator()->allocate();
        boxes.allocator()->allocate();
        pred_boxes.allocator()->allocate();

        ARM_COMPUTE_EXPECT(!deltas.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!boxes.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Fill tensors
        TensorShape img_shape(bbox_info.scale() * bbox_info.img_width(), bbox_info.scale() * bbox_info.img_height());
        generate_boxes(AccessorType(boxes), img_shape, boxes_shape[1], gen);
        generate_deltas(AccessorType(deltas), AccessorType(boxes), img_shape, deltas_shape[1], deltas_shape[0] / 4, gen);

        // Compute function
        bbox_transform.run();

        return pred_boxes;
    }

    SimpleTensor<T> compute_reference(const TensorShape              &deltas_shape,
                                      DataType                        data_type,
                                      const BoundingBoxTransformInfo &bbox_info, std::mt19937 &gen)
    {
        // Create reference tensor
        TensorShape     boxes_shape(4, deltas_shape[1]);
        SimpleTensor<T> boxes{ boxes_shape, data_type };
        SimpleTensor<T> deltas{ deltas_shape, data_type };

        // Fill reference tensor
        TensorShape img_shape(bbox_info.scale() * bbox_info.img_width(), bbox_info.scale() * bbox_info.img_height());
        generate_boxes(boxes, img_shape, boxes_shape[1], gen);
        generate_deltas(deltas, boxes, img_shape, deltas_shape[1], deltas_shape[0] / 4, gen);

        return reference::bounding_box_transform(boxes, deltas, bbox_info);
    }

    TensorType      _target{};
    SimpleTensor<T> _reference{};

private:
    template <typename U>
    void generate_deltas(U &&deltas, U &&boxes, const TensorShape &image_shape, size_t num_boxes, size_t num_classes, std::mt19937 &gen)
    {
        T *deltas_ptr = static_cast<T *>(deltas.data());
        T *boxes_ptr  = static_cast<T *>(boxes.data());

        std::uniform_int_distribution<> dist_x1(0, image_shape[0] - 1);
        std::uniform_int_distribution<> dist_y1(0, image_shape[1] - 1);
        std::uniform_int_distribution<> dist_w(1, image_shape[0]);
        std::uniform_int_distribution<> dist_h(1, image_shape[1]);

        for(size_t i = 0; i < num_boxes; ++i)
        {
            const T ex_width  = boxes_ptr[4 * i + 2] - boxes_ptr[4 * i] + T(1);
            const T ex_height = boxes_ptr[4 * i + 3] - boxes_ptr[4 * i + 1] + T(1);
            const T ex_ctr_x  = boxes_ptr[4 * i] + T(0.5) * ex_width;
            const T ex_ctr_y  = boxes_ptr[4 * i + 1] + T(0.5) * ex_height;

            for(size_t j = 0; j < num_classes; ++j)
            {
                const T x1     = T(dist_x1(gen));
                const T y1     = T(dist_y1(gen));
                const T width  = T(dist_w(gen));
                const T height = T(dist_h(gen));
                const T ctr_x  = x1 + T(0.5) * width;
                const T ctr_y  = y1 + T(0.5) * height;

                deltas_ptr[4 * num_classes * i + 4 * j]     = (ctr_x - ex_ctr_x) / ex_width;
                deltas_ptr[4 * num_classes * i + 4 * j + 1] = (ctr_y - ex_ctr_y) / ex_height;
                deltas_ptr[4 * num_classes * i + 4 * j + 2] = log(width / ex_width);
                deltas_ptr[4 * num_classes * i + 4 * j + 3] = log(height / ex_height);
            }
        }
    }

    template <typename U>
    void generate_boxes(U &&boxes, const TensorShape &image_shape, size_t num_boxes, std::mt19937 &gen)
    {
        T *boxes_ptr = (T *)boxes.data();

        std::uniform_int_distribution<> dist_x1(0, image_shape[0] - 1);
        std::uniform_int_distribution<> dist_y1(0, image_shape[1] - 1);
        std::uniform_int_distribution<> dist_w(1, image_shape[0]);
        std::uniform_int_distribution<> dist_h(1, image_shape[1]);

        for(size_t i = 0; i < num_boxes; ++i)
        {
            boxes_ptr[4 * i]     = dist_x1(gen);
            boxes_ptr[4 * i + 1] = dist_y1(gen);
            boxes_ptr[4 * i + 2] = boxes_ptr[4 * i] + dist_w(gen) - 1;
            boxes_ptr[4 * i + 3] = boxes_ptr[4 * i + 1] + dist_h(gen) - 1;
        }
    }
};

} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_BOUNDINGBOXTRANSFORM_FIXTURE */
