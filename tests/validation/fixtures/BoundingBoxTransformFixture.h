/*
 * Copyright (c) 2018-2021, 2023 Arm Limited.
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
namespace
{
std::vector<float> generate_deltas(std::vector<float> &boxes, const TensorShape &image_shape, size_t num_boxes, size_t num_classes, std::mt19937 &gen)
{
    std::vector<float> deltas(num_boxes * 4 * num_classes);

    std::uniform_int_distribution<> dist_x1(0, image_shape[0] - 1);
    std::uniform_int_distribution<> dist_y1(0, image_shape[1] - 1);
    std::uniform_int_distribution<> dist_w(1, image_shape[0]);
    std::uniform_int_distribution<> dist_h(1, image_shape[1]);

    for(size_t i = 0; i < num_boxes; ++i)
    {
        const float ex_width  = boxes[4 * i + 2] - boxes[4 * i] + 1.f;
        const float ex_height = boxes[4 * i + 3] - boxes[4 * i + 1] + 1.f;
        const float ex_ctr_x  = boxes[4 * i] + 0.5f * ex_width;
        const float ex_ctr_y  = boxes[4 * i + 1] + 0.5f * ex_height;

        for(size_t j = 0; j < num_classes; ++j)
        {
            const float x1     = dist_x1(gen);
            const float y1     = dist_y1(gen);
            const float width  = dist_w(gen);
            const float height = dist_h(gen);
            const float ctr_x  = x1 + 0.5f * width;
            const float ctr_y  = y1 + 0.5f * height;

            deltas[4 * num_classes * i + 4 * j]     = (ctr_x - ex_ctr_x) / ex_width;
            deltas[4 * num_classes * i + 4 * j + 1] = (ctr_y - ex_ctr_y) / ex_height;
            deltas[4 * num_classes * i + 4 * j + 2] = log(width / ex_width);
            deltas[4 * num_classes * i + 4 * j + 3] = log(height / ex_height);
        }
    }
    return deltas;
}

std::vector<float> generate_boxes(const TensorShape &image_shape, size_t num_boxes, std::mt19937 &gen)
{
    std::vector<float> boxes(num_boxes * 4);

    std::uniform_int_distribution<> dist_x1(0, image_shape[0] - 1);
    std::uniform_int_distribution<> dist_y1(0, image_shape[1] - 1);
    std::uniform_int_distribution<> dist_w(1, image_shape[0]);
    std::uniform_int_distribution<> dist_h(1, image_shape[1]);

    for(size_t i = 0; i < num_boxes; ++i)
    {
        boxes[4 * i]     = dist_x1(gen);
        boxes[4 * i + 1] = dist_y1(gen);
        boxes[4 * i + 2] = boxes[4 * i] + dist_w(gen) - 1;
        boxes[4 * i + 3] = boxes[4 * i + 1] + dist_h(gen) - 1;
    }
    return boxes;
}
} // namespace

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class BoundingBoxTransformGenericFixture : public framework::Fixture
{
public:
    using TDeltas = typename std::conditional<std::is_same<typename std::decay<T>::type, uint16_t>::value, uint8_t, T>::type;

    void setup(TensorShape deltas_shape, const BoundingBoxTransformInfo &info, DataType data_type, QuantizationInfo deltas_qinfo)
    {
        const bool is_qasymm16 = data_type == DataType::QASYMM16;
        _data_type_deltas      = (is_qasymm16) ? DataType::QASYMM8 : data_type;
        _boxes_qinfo           = (is_qasymm16) ? QuantizationInfo(.125f, 0) : QuantizationInfo();

        std::mt19937 gen_target(library->seed());
        _target = compute_target(deltas_shape, data_type, info, gen_target, deltas_qinfo);

        std::mt19937 gen_reference(library->seed());
        _reference = compute_reference(deltas_shape, data_type, info, gen_reference, deltas_qinfo);
    }

protected:
    template <typename data_type, typename U>
    void fill(U &&tensor, std::vector<float> values)
    {
        data_type *data_ptr = reinterpret_cast<data_type *>(tensor.data());
        switch(tensor.data_type())
        {
            case DataType::QASYMM8:
                for(size_t i = 0; i < values.size(); ++i)
                {
                    data_ptr[i] = quantize_qasymm8(values[i], tensor.quantization_info());
                }
                break;
            case DataType::QASYMM16:
                for(size_t i = 0; i < values.size(); ++i)
                {
                    data_ptr[i] = quantize_qasymm16(values[i], tensor.quantization_info());
                }
                break;
            default:
                for(size_t i = 0; i < values.size(); ++i)
                {
                    data_ptr[i] = static_cast<data_type>(values[i]);
                }
        }
    }

    TensorType compute_target(const TensorShape &deltas_shape, DataType data_type,
                              const BoundingBoxTransformInfo &bbox_info, std::mt19937 &gen,
                              QuantizationInfo deltas_qinfo)
    {
        // Create tensors
        TensorShape boxes_shape(4, deltas_shape[1]);
        TensorType  deltas = create_tensor<TensorType>(deltas_shape, _data_type_deltas, 1, deltas_qinfo);
        TensorType  boxes  = create_tensor<TensorType>(boxes_shape, data_type, 1, _boxes_qinfo);
        TensorType  pred_boxes;

        // Create and configure function
        FunctionType bbox_transform;
        bbox_transform.configure(&boxes, &pred_boxes, &deltas, bbox_info);

        ARM_COMPUTE_ASSERT(deltas.info()->is_resizable());
        ARM_COMPUTE_ASSERT(boxes.info()->is_resizable());
        ARM_COMPUTE_ASSERT(pred_boxes.info()->is_resizable());

        // Allocate tensors
        deltas.allocator()->allocate();
        boxes.allocator()->allocate();
        pred_boxes.allocator()->allocate();

        ARM_COMPUTE_ASSERT(!deltas.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!boxes.info()->is_resizable());

        // Fill tensors
        TensorShape        img_shape(bbox_info.scale() * bbox_info.img_width(), bbox_info.scale() * bbox_info.img_height());
        std::vector<float> boxes_vec  = generate_boxes(img_shape, boxes_shape[1], gen);
        std::vector<float> deltas_vec = generate_deltas(boxes_vec, img_shape, deltas_shape[1], deltas_shape[0] / 4, gen);
        fill<T>(AccessorType(boxes), boxes_vec);
        fill<TDeltas>(AccessorType(deltas), deltas_vec);

        // Compute function
        bbox_transform.run();

        return pred_boxes;
    }

    SimpleTensor<T> compute_reference(const TensorShape              &deltas_shape,
                                      DataType                        data_type,
                                      const BoundingBoxTransformInfo &bbox_info,
                                      std::mt19937                   &gen,
                                      QuantizationInfo                deltas_qinfo)
    {
        // Create reference tensor
        TensorShape           boxes_shape(4, deltas_shape[1]);
        SimpleTensor<T>       boxes{ boxes_shape, data_type, 1, _boxes_qinfo };
        SimpleTensor<TDeltas> deltas{ deltas_shape, _data_type_deltas, 1, deltas_qinfo };

        // Fill reference tensor
        TensorShape        img_shape(bbox_info.scale() * bbox_info.img_width(), bbox_info.scale() * bbox_info.img_height());
        std::vector<float> boxes_vec  = generate_boxes(img_shape, boxes_shape[1], gen);
        std::vector<float> deltas_vec = generate_deltas(boxes_vec, img_shape, deltas_shape[1], deltas_shape[0] / 4, gen);
        fill<T>(boxes, boxes_vec);
        fill<TDeltas>(deltas, deltas_vec);

        return reference::bounding_box_transform(boxes, deltas, bbox_info);
    }

    TensorType       _target{};
    SimpleTensor<T>  _reference{};
    DataType         _data_type_deltas{};
    QuantizationInfo _boxes_qinfo{};

private:
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class BoundingBoxTransformFixture : public BoundingBoxTransformGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    void setup(TensorShape deltas_shape, const BoundingBoxTransformInfo &info, DataType data_type)
    {
        BoundingBoxTransformGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(deltas_shape, info, data_type, QuantizationInfo());
    }

private:
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class BoundingBoxTransformQuantizedFixture : public BoundingBoxTransformGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    void setup(TensorShape deltas_shape, const BoundingBoxTransformInfo &info, DataType data_type, QuantizationInfo deltas_qinfo)
    {
        BoundingBoxTransformGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(deltas_shape, info, data_type, deltas_qinfo);
    }
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_BOUNDINGBOXTRANSFORM_FIXTURE */
