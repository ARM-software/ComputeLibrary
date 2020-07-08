/*
 * Copyright (c) 2018-2020 Arm Limited.
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
#ifndef ARM_COMPUTE_TEST_ROIALIGNLAYER_FIXTURE
#define ARM_COMPUTE_TEST_ROIALIGNLAYER_FIXTURE

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/IAccessor.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/Helpers.h"
#include "tests/validation/reference/ROIAlignLayer.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
template <typename TensorType, typename AccessorType, typename FunctionType, typename T, typename TRois>
class ROIAlignLayerGenericFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(TensorShape input_shape, const ROIPoolingLayerInfo pool_info, TensorShape rois_shape, DataType data_type, DataLayout data_layout, QuantizationInfo qinfo, QuantizationInfo output_qinfo)
    {
        _rois_data_type = is_data_type_quantized_asymmetric(data_type) ? DataType::QASYMM16 : data_type;
        _target         = compute_target(input_shape, data_type, data_layout, pool_info, rois_shape, qinfo, output_qinfo);
        _reference      = compute_reference(input_shape, data_type, pool_info, rois_shape, qinfo, output_qinfo);
    }

protected:
    template <typename U>
    void fill(U &&tensor)
    {
        library->fill_tensor_uniform(tensor, 0);
    }

    template <typename U>
    void generate_rois(U &&rois, const TensorShape &shape, const ROIPoolingLayerInfo &pool_info, TensorShape rois_shape, DataLayout data_layout = DataLayout::NCHW)
    {
        const size_t values_per_roi = rois_shape.x();
        const size_t num_rois       = rois_shape.y();

        std::mt19937 gen(library->seed());
        TRois       *rois_ptr = static_cast<TRois *>(rois.data());

        const float pool_width  = pool_info.pooled_width();
        const float pool_height = pool_info.pooled_height();
        const float roi_scale   = pool_info.spatial_scale();

        // Calculate distribution bounds
        const auto scaled_width  = static_cast<float>((shape[get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH)] / roi_scale) / pool_width);
        const auto scaled_height = static_cast<float>((shape[get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT)] / roi_scale) / pool_height);
        const auto min_width     = static_cast<float>(pool_width / roi_scale);
        const auto min_height    = static_cast<float>(pool_height / roi_scale);

        // Create distributions
        std::uniform_int_distribution<int> dist_batch(0, shape[3] - 1);
        std::uniform_int_distribution<>    dist_x1(0, scaled_width);
        std::uniform_int_distribution<>    dist_y1(0, scaled_height);
        std::uniform_int_distribution<>    dist_w(min_width, std::max(float(min_width), (pool_width - 2) * scaled_width));
        std::uniform_int_distribution<>    dist_h(min_height, std::max(float(min_height), (pool_height - 2) * scaled_height));

        for(unsigned int pw = 0; pw < num_rois; ++pw)
        {
            const auto batch_idx = dist_batch(gen);
            const auto x1        = dist_x1(gen);
            const auto y1        = dist_y1(gen);
            const auto x2        = x1 + dist_w(gen);
            const auto y2        = y1 + dist_h(gen);

            rois_ptr[values_per_roi * pw] = batch_idx;
            if(rois.data_type() == DataType::QASYMM16)
            {
                rois_ptr[values_per_roi * pw + 1] = quantize_qasymm16(static_cast<float>(x1), rois.quantization_info());
                rois_ptr[values_per_roi * pw + 2] = quantize_qasymm16(static_cast<float>(y1), rois.quantization_info());
                rois_ptr[values_per_roi * pw + 3] = quantize_qasymm16(static_cast<float>(x2), rois.quantization_info());
                rois_ptr[values_per_roi * pw + 4] = quantize_qasymm16(static_cast<float>(y2), rois.quantization_info());
            }
            else
            {
                rois_ptr[values_per_roi * pw + 1] = static_cast<TRois>(x1);
                rois_ptr[values_per_roi * pw + 2] = static_cast<TRois>(y1);
                rois_ptr[values_per_roi * pw + 3] = static_cast<TRois>(x2);
                rois_ptr[values_per_roi * pw + 4] = static_cast<TRois>(y2);
            }
        }
    }

    TensorType compute_target(TensorShape                input_shape,
                              DataType                   data_type,
                              DataLayout                 data_layout,
                              const ROIPoolingLayerInfo &pool_info,
                              const TensorShape          rois_shape,
                              const QuantizationInfo    &qinfo,
                              const QuantizationInfo    &output_qinfo)
    {
        if(data_layout == DataLayout::NHWC)
        {
            permute(input_shape, PermutationVector(2U, 0U, 1U));
        }

        const QuantizationInfo rois_qinfo = is_data_type_quantized(data_type) ? QuantizationInfo(0.125f, 0) : QuantizationInfo();

        // Create tensors
        TensorType src         = create_tensor<TensorType>(input_shape, data_type, 1, qinfo, data_layout);
        TensorType rois_tensor = create_tensor<TensorType>(rois_shape, _rois_data_type, 1, rois_qinfo);

        const TensorShape dst_shape = misc::shape_calculator::compute_roi_align_shape(*(src.info()), *(rois_tensor.info()), pool_info);
        TensorType        dst       = create_tensor<TensorType>(dst_shape, data_type, 1, output_qinfo, data_layout);

        // Create and configure function
        FunctionType roi_align_layer;
        roi_align_layer.configure(&src, &rois_tensor, &dst, pool_info);

        ARM_COMPUTE_EXPECT(src.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(rois_tensor.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(dst.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Allocate tensors
        src.allocator()->allocate();
        rois_tensor.allocator()->allocate();
        dst.allocator()->allocate();

        ARM_COMPUTE_EXPECT(!src.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!rois_tensor.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!dst.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Fill tensors
        fill(AccessorType(src));
        generate_rois(AccessorType(rois_tensor), input_shape, pool_info, rois_shape, data_layout);

        // Compute function
        roi_align_layer.run();

        return dst;
    }

    SimpleTensor<T> compute_reference(const TensorShape         &input_shape,
                                      DataType                   data_type,
                                      const ROIPoolingLayerInfo &pool_info,
                                      const TensorShape          rois_shape,
                                      const QuantizationInfo    &qinfo,
                                      const QuantizationInfo    &output_qinfo)
    {
        // Create reference tensor
        SimpleTensor<T>        src{ input_shape, data_type, 1, qinfo };
        const QuantizationInfo rois_qinfo = is_data_type_quantized(data_type) ? QuantizationInfo(0.125f, 0) : QuantizationInfo();
        SimpleTensor<TRois>    rois_tensor{ rois_shape, _rois_data_type, 1, rois_qinfo };

        // Fill reference tensor
        fill(src);
        generate_rois(rois_tensor, input_shape, pool_info, rois_shape);

        return reference::roi_align_layer(src, rois_tensor, pool_info, output_qinfo);
    }

    TensorType      _target{};
    SimpleTensor<T> _reference{};
    DataType        _rois_data_type{};
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T, typename TRois>
class ROIAlignLayerFixture : public ROIAlignLayerGenericFixture<TensorType, AccessorType, FunctionType, T, TRois>
{
public:
    template <typename...>
    void setup(TensorShape input_shape, const ROIPoolingLayerInfo pool_info, TensorShape rois_shape, DataType data_type, DataLayout data_layout)
    {
        ROIAlignLayerGenericFixture<TensorType, AccessorType, FunctionType, T, TRois>::setup(input_shape, pool_info, rois_shape, data_type, data_layout,
                                                                                             QuantizationInfo(), QuantizationInfo());
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T, typename TRois>
class ROIAlignLayerQuantizedFixture : public ROIAlignLayerGenericFixture<TensorType, AccessorType, FunctionType, T, TRois>
{
public:
    template <typename...>
    void setup(TensorShape input_shape, const ROIPoolingLayerInfo pool_info, TensorShape rois_shape, DataType data_type,
               DataLayout data_layout, QuantizationInfo qinfo, QuantizationInfo output_qinfo)
    {
        ROIAlignLayerGenericFixture<TensorType, AccessorType, FunctionType, T, TRois>::setup(input_shape, pool_info, rois_shape,
                                                                                             data_type, data_layout, qinfo, output_qinfo);
    }
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_ROIALIGNLAYER_FIXTURE */
