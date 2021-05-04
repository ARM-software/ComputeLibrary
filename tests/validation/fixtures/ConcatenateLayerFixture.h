/*
 * Copyright (c) 2018-2021 Arm Limited.
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
#ifndef ARM_COMPUTE_TEST_WIDTHCONCATENATE_LAYER_FIXTURE
#define ARM_COMPUTE_TEST_WIDTHCONCATENATE_LAYER_FIXTURE

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/IAccessor.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/Helpers.h"
#include "tests/validation/reference/ConcatenateLayer.h"

#include <random>

namespace arm_compute
{
namespace test
{
namespace validation
{
template <typename TensorType, typename ITensorType, typename AccessorType, typename FunctionType, typename T, bool CI = true>
class ConcatenateLayerValidationFixture : public framework::Fixture
{
private:
    using SrcITensorType = typename std::conditional<CI, const ITensorType, ITensorType>::type;

public:
    template <typename...>
    void setup(TensorShape shape, DataType data_type, unsigned int axis)
    {
        // Create input shapes
        std::mt19937                    gen(library->seed());
        std::uniform_int_distribution<> num_dis(2, 8);
        std::uniform_int_distribution<> offset_dis(0, 20);

        const int num_tensors = num_dis(gen);

        std::vector<TensorShape> shapes(num_tensors, shape);

        // vector holding the quantization info:
        //      the last element is the output quantization info
        //      all other elements are the quantization info for the input tensors
        std::vector<QuantizationInfo> qinfo(num_tensors + 1, QuantizationInfo());
        for(auto &qi : qinfo)
        {
            qi = QuantizationInfo(1.f / 255.f, offset_dis(gen));
        }
        std::bernoulli_distribution           mutate_dis(0.5f);
        std::uniform_real_distribution<float> change_dis(-0.25f, 0.f);

        // Generate more shapes based on the input
        for(auto &s : shapes)
        {
            // Randomly change the dimension
            if(mutate_dis(gen))
            {
                // Decrease the dimension by a small percentage. Don't increase
                // as that could make tensor too large.
                s.set(axis, s[axis] + 2 * static_cast<int>(s[axis] * change_dis(gen)));
            }
        }

        _target    = compute_target(shapes, qinfo, data_type, axis);
        _reference = compute_reference(shapes, qinfo, data_type, axis);
    }

protected:
    template <typename U>
    void fill(U &&tensor, int i)
    {
        library->fill_tensor_uniform(tensor, i);
    }

    TensorType compute_target(const std::vector<TensorShape> &shapes, const std::vector<QuantizationInfo> &qinfo, DataType data_type, unsigned int axis)
    {
        std::vector<TensorType>       srcs;
        std::vector<SrcITensorType *> src_ptrs;

        // Create tensors
        srcs.reserve(shapes.size());

        for(size_t j = 0; j < shapes.size(); ++j)
        {
            srcs.emplace_back(create_tensor<TensorType>(shapes[j], data_type, 1, qinfo[j]));
            src_ptrs.emplace_back(&srcs.back());
        }

        const TensorShape dst_shape = misc::shape_calculator::calculate_concatenate_shape(src_ptrs, axis);
        TensorType        dst       = create_tensor<TensorType>(dst_shape, data_type, 1, qinfo[shapes.size()]);

        // Create and configure function
        FunctionType concat;
        concat.configure(src_ptrs, &dst, axis);

        for(auto &src : srcs)
        {
            ARM_COMPUTE_ASSERT(src.info()->is_resizable());
        }

        ARM_COMPUTE_ASSERT(dst.info()->is_resizable());

        // Allocate tensors
        for(auto &src : srcs)
        {
            src.allocator()->allocate();
            ARM_COMPUTE_ASSERT(!src.info()->is_resizable());
        }

        dst.allocator()->allocate();
        ARM_COMPUTE_ASSERT(!dst.info()->is_resizable());

        // Fill tensors
        int i = 0;
        for(auto &src : srcs)
        {
            fill(AccessorType(src), i++);
        }

        // Compute function
        concat.run();

        return dst;
    }

    SimpleTensor<T> compute_reference(std::vector<TensorShape> &shapes, const std::vector<QuantizationInfo> &qinfo, DataType data_type, unsigned int axis)
    {
        std::vector<SimpleTensor<T>> srcs;
        std::vector<TensorShape *>   src_ptrs;

        // Create and fill tensors
        for(size_t j = 0; j < shapes.size(); ++j)
        {
            srcs.emplace_back(shapes[j], data_type, 1, qinfo[j]);
            fill(srcs.back(), j);
            src_ptrs.emplace_back(&shapes[j]);
        }

        const TensorShape dst_shape = misc::shape_calculator::calculate_concatenate_shape(src_ptrs, axis);
        SimpleTensor<T>   dst{ dst_shape, data_type, 1, qinfo[shapes.size()] };
        return reference::concatenate_layer<T>(srcs, dst, axis);
    }

    TensorType      _target{};
    SimpleTensor<T> _reference{};
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_WIDTHCONCATENATE_LAYER_FIXTURE */
