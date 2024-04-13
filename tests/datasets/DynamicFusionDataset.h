/*
 * Copyright (c) 2022 Arm Limited.
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
#ifndef TESTS_DATASETS_DYNAMICFUSIONDATASET
#define TESTS_DATASETS_DYNAMICFUSIONDATASET

#include "utils/TypePrinter.h"

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"

namespace arm_compute
{
namespace test
{
namespace datasets
{
class DynamicFusionThreeInputs
{
public:
    using type = std::tuple<TensorShape, TensorShape, TensorShape>;

    struct iterator
    {
        iterator(std::vector<TensorShape>::const_iterator shape0_it,
                 std::vector<TensorShape>::const_iterator shape1_it,
                 std::vector<TensorShape>::const_iterator shape2_it)
            : _shape0_it{ std::move(shape0_it) },
              _shape1_it{ std::move(shape1_it) },
              _shape2_it{ std::move(shape2_it) }
        {
        }

        std::string description() const
        {
            std::stringstream description;
            description << "shape0=" << *_shape0_it << ":";
            description << "shape1=" << *_shape1_it << ":";
            description << "shape2=" << *_shape2_it << ":";

            return description.str();
        }

        DynamicFusionThreeInputs::type operator*() const
        {
            return std::make_tuple(*_shape0_it, *_shape1_it, *_shape2_it);
        }

        iterator &operator++()
        {
            ++_shape0_it;
            ++_shape1_it;
            ++_shape2_it;

            return *this;
        }

    private:
        std::vector<TensorShape>::const_iterator _shape0_it;
        std::vector<TensorShape>::const_iterator _shape1_it;
        std::vector<TensorShape>::const_iterator _shape2_it;
    };

    iterator begin() const
    {
        return iterator(_shape0_shapes.begin(), _shape1_shapes.begin(), _shape2_shapes.begin());
    }

    int size() const
    {
        return std::min(_shape0_shapes.size(), std::min(_shape1_shapes.size(), _shape2_shapes.size()));
    }

    void add_config(TensorShape shape0, TensorShape shape1, TensorShape shape2)
    {
        _shape0_shapes.emplace_back(std::move(shape0));
        _shape1_shapes.emplace_back(std::move(shape1));
        _shape2_shapes.emplace_back(std::move(shape2));
    }

protected:
    DynamicFusionThreeInputs()                            = default;
    DynamicFusionThreeInputs(DynamicFusionThreeInputs &&) = default;

private:
    std::vector<TensorShape> _shape0_shapes{};
    std::vector<TensorShape> _shape1_shapes{};
    std::vector<TensorShape> _shape2_shapes{};
};

class DynamicFusionElementwiseBinaryTwoOpsSmallShapes final : public DynamicFusionThreeInputs
{
public:
    DynamicFusionElementwiseBinaryTwoOpsSmallShapes()
    {
        add_config(TensorShape{ 9U, 9U, 5U }, TensorShape{ 9U, 9U, 5U }, TensorShape{ 9U, 9U, 5U });
        add_config(TensorShape{ 9U, 9U, 5U }, TensorShape{ 1U, 1U, 1U } /* Broadcast in X, Y, Z*/, TensorShape{ 9U, 9U, 5U });
        add_config(TensorShape{ 27U, 13U, 2U }, TensorShape{ 27U, 1U, 1U } /* Broadcast in Y and Z*/, TensorShape{ 27U, 13U, 2U });
        add_config(TensorShape{ 27U, 13U, 2U }, TensorShape{ 27U, 13U, 2U }, TensorShape{ 27U, 1U, 1U } /* Broadcast in Y and Z*/);
    }
};

} // namespace datasets
} // namespace test
} // namespace arm_compute
#endif /* TESTS_DATASETS_DYNAMICFUSIONDATASET */
