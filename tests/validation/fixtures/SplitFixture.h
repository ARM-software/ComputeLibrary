/*
 * Copyright (c) 2018-2020 ARM Limited.
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
#ifndef ARM_COMPUTE_TEST_SPLIT_FIXTURE
#define ARM_COMPUTE_TEST_SPLIT_FIXTURE

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"

#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/IAccessor.h"
#include "tests/RawLutAccessor.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/Helpers.h"
#include "tests/validation/reference/SliceOperations.h"

#include <algorithm>

namespace arm_compute
{
namespace test
{
namespace validation
{
template <typename TensorType, typename ITensorType, typename AccessorType, typename FunctionType, typename T>
class SplitFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(TensorShape shape, unsigned int axis, unsigned int splits, DataType data_type)
    {
        _target    = compute_target(shape, axis, splits, data_type);
        _reference = compute_reference(shape, axis, splits, data_type);
    }

protected:
    template <typename U>
    void fill(U &&tensor, int i)
    {
        library->fill_tensor_uniform(tensor, i);
    }

    std::vector<TensorType> compute_target(const TensorShape &shape, unsigned int axis, unsigned int splits, DataType data_type)
    {
        // Create tensors
        TensorType                 src = create_tensor<TensorType>(shape, data_type);
        std::vector<TensorType>    dsts(splits);
        std::vector<ITensorType *> dsts_ptr;
        for(auto &dst : dsts)
        {
            dsts_ptr.emplace_back(&dst);
        }

        // Create and configure function
        FunctionType split;
        split.configure(&src, dsts_ptr, axis);

        ARM_COMPUTE_EXPECT(src.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(std::all_of(dsts.cbegin(), dsts.cend(), [](const TensorType & t)
        {
            return t.info()->is_resizable();
        }),
        framework::LogLevel::ERRORS);

        // Allocate tensors
        src.allocator()->allocate();
        for(unsigned int i = 0; i < splits; ++i)
        {
            dsts[i].allocator()->allocate();
        }

        ARM_COMPUTE_EXPECT(!src.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(std::all_of(dsts.cbegin(), dsts.cend(), [](const TensorType & t)
        {
            return !t.info()->is_resizable();
        }),
        framework::LogLevel::ERRORS);

        // Fill tensors
        fill(AccessorType(src), 0);

        // Compute function
        split.run();

        return dsts;
    }

    std::vector<SimpleTensor<T>> compute_reference(const TensorShape &shape, unsigned int axis, unsigned int splits, DataType data_type)
    {
        // Create reference
        SimpleTensor<T>              src{ shape, data_type };
        std::vector<SimpleTensor<T>> dsts;

        // Fill reference
        fill(src, 0);

        // Calculate splice for each split
        const size_t axis_split_step = shape[axis] / splits;
        unsigned int axis_offset     = 0;

        // Start/End coordinates
        Coordinates start_coords;
        Coordinates end_coords;
        for(unsigned int d = 0; d < shape.num_dimensions(); ++d)
        {
            end_coords.set(d, -1);
        }

        for(unsigned int i = 0; i < splits; ++i)
        {
            // Update coordinate on axis
            start_coords.set(axis, axis_offset);
            end_coords.set(axis, axis_offset + axis_split_step);

            dsts.emplace_back(std::move(reference::slice(src, start_coords, end_coords)));

            axis_offset += axis_split_step;
        }

        return dsts;
    }

    std::vector<TensorType>      _target{};
    std::vector<SimpleTensor<T>> _reference{};
};

template <typename TensorType, typename ITensorType, typename AccessorType, typename FunctionType, typename T>
class SplitShapesFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(TensorShape shape, unsigned int axis, std::vector<TensorShape> split_shapes, DataType data_type)
    {
        _target    = compute_target(shape, axis, split_shapes, data_type);
        _reference = compute_reference(shape, axis, split_shapes, data_type);
    }

protected:
    template <typename U>
    void fill(U &&tensor, int i)
    {
        library->fill_tensor_uniform(tensor, i);
    }

    std::vector<TensorType> compute_target(TensorShape shape, unsigned int axis, std::vector<TensorShape> split_shapes, DataType data_type)
    {
        // Create tensors
        TensorType                 src = create_tensor<TensorType>(shape, data_type);
        std::vector<TensorType>    dsts{};
        std::vector<ITensorType *> dsts_ptr;

        for(const auto &split_shape : split_shapes)
        {
            TensorType dst = create_tensor<TensorType>(split_shape, data_type);
            dsts.push_back(std::move(dst));
        }

        for(auto &dst : dsts)
        {
            dsts_ptr.emplace_back(&dst);
        }

        // Create and configure function
        FunctionType split;
        split.configure(&src, dsts_ptr, axis);

        ARM_COMPUTE_EXPECT(src.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(std::all_of(dsts.cbegin(), dsts.cend(), [](const TensorType & t)
        {
            return t.info()->is_resizable();
        }),
        framework::LogLevel::ERRORS);

        // Allocate tensors
        src.allocator()->allocate();
        for(unsigned int i = 0; i < dsts.size(); ++i)
        {
            dsts[i].allocator()->allocate();
        }

        ARM_COMPUTE_EXPECT(!src.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(std::all_of(dsts.cbegin(), dsts.cend(), [](const TensorType & t)
        {
            return !t.info()->is_resizable();
        }),
        framework::LogLevel::ERRORS);

        // Fill tensors
        fill(AccessorType(src), 0);

        // Compute function
        split.run();

        return dsts;
    }

    std::vector<SimpleTensor<T>> compute_reference(TensorShape shape, unsigned int axis, std::vector<TensorShape> split_shapes, DataType data_type)
    {
        // Create reference
        SimpleTensor<T>              src{ shape, data_type };
        std::vector<SimpleTensor<T>> dsts;

        // Fill reference
        fill(src, 0);

        unsigned int axis_offset{ 0 };
        for(const auto &split_shape : split_shapes)
        {
            // Calculate splice for each split
            const size_t axis_split_step = split_shape[axis];

            // Start/End coordinates
            Coordinates start_coords;
            Coordinates end_coords;
            for(unsigned int d = 0; d < shape.num_dimensions(); ++d)
            {
                end_coords.set(d, -1);
            }

            // Update coordinate on axis
            start_coords.set(axis, axis_offset);
            end_coords.set(axis, axis_offset + axis_split_step);

            dsts.emplace_back(std::move(reference::slice(src, start_coords, end_coords)));

            axis_offset += axis_split_step;
        }

        return dsts;
    }

    std::vector<TensorType>      _target{};
    std::vector<SimpleTensor<T>> _reference{};
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_SPLIT_FIXTURE */
