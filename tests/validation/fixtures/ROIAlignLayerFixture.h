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
#ifndef ARM_COMPUTE_TEST_ROIALIGNLAYER_FIXTURE
#define ARM_COMPUTE_TEST_ROIALIGNLAYER_FIXTURE

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/functions/CLROIAlignLayer.h"
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
template <typename TensorType, typename AccessorType, typename FunctionType, typename Array_T, typename ArrayAccessor, typename T>
class ROIAlignLayerFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(TensorShape input_shape, const ROIPoolingLayerInfo pool_info, unsigned int num_rois, DataType data_type, int batches)
    {
        input_shape.set(2, batches);
        std::vector<ROI> rois = generate_random_rois(input_shape, pool_info, num_rois, 0U);

        _target    = compute_target(input_shape, data_type, rois, pool_info);
        _reference = compute_reference(input_shape, data_type, rois, pool_info);
    }

protected:
    template <typename U>
    void fill(U &&tensor)
    {
        library->fill_tensor_uniform(tensor, 0);
    }

    TensorType compute_target(const TensorShape         &input_shape,
                              DataType                   data_type,
                              std::vector<ROI> const    &rois,
                              const ROIPoolingLayerInfo &pool_info)
    {
        // Create tensors
        TensorType src = create_tensor<TensorType>(input_shape, data_type);
        TensorType dst;

        size_t num_rois = rois.size();

        // Create roi arrays
        std::unique_ptr<Array_T> rois_array = arm_compute::support::cpp14::make_unique<Array_T>(num_rois);
        fill_array(ArrayAccessor(*rois_array), rois);

        // Create and configure function
        FunctionType roi_align_layer;
        roi_align_layer.configure(&src, rois_array.get(), &dst, pool_info);

        ARM_COMPUTE_EXPECT(src.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(dst.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Allocate tensors
        src.allocator()->allocate();
        dst.allocator()->allocate();

        ARM_COMPUTE_EXPECT(!src.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!dst.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Fill tensors
        fill(AccessorType(src));

        // Compute function
        roi_align_layer.run();

        return dst;
    }

    SimpleTensor<T> compute_reference(const TensorShape         &input_shape,
                                      DataType                   data_type,
                                      std::vector<ROI> const    &rois,
                                      const ROIPoolingLayerInfo &pool_info)
    {
        // Create reference tensor
        SimpleTensor<T> src{ input_shape, data_type };

        // Fill reference tensor
        fill(src);

        return reference::roi_align_layer(src, rois, pool_info);
    }

    TensorType      _target{};
    SimpleTensor<T> _reference{};
};

} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_ROIALIGNLAYER_FIXTURE */
