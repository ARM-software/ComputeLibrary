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
#ifndef ARM_COMPUTE_TEST_STRIDED_SLICE_DATASET
#define ARM_COMPUTE_TEST_STRIDED_SLICE_DATASET

#include "utils/TypePrinter.h"

#include "arm_compute/core/Types.h"

namespace arm_compute
{
namespace test
{
namespace datasets
{
class SliceDataset
{
public:
    using type = std::tuple<TensorShape, Coordinates, Coordinates>;

    struct iterator
    {
        iterator(std::vector<TensorShape>::const_iterator tensor_shapes_it,
                 std::vector<Coordinates>::const_iterator starts_values_it,
                 std::vector<Coordinates>::const_iterator ends_values_it)
            : _tensor_shapes_it{ std::move(tensor_shapes_it) },
              _starts_values_it{ std::move(starts_values_it) },
              _ends_values_it{ std::move(ends_values_it) }
        {
        }

        std::string description() const
        {
            std::stringstream description;
            description << "Shape=" << *_tensor_shapes_it << ":";
            description << "Starts=" << *_starts_values_it << ":";
            description << "Ends=" << *_ends_values_it << ":";
            return description.str();
        }

        SliceDataset::type operator*() const
        {
            return std::make_tuple(*_tensor_shapes_it, *_starts_values_it, *_ends_values_it);
        }

        iterator &operator++()
        {
            ++_tensor_shapes_it;
            ++_starts_values_it;
            ++_ends_values_it;
            return *this;
        }

    private:
        std::vector<TensorShape>::const_iterator _tensor_shapes_it;
        std::vector<Coordinates>::const_iterator _starts_values_it;
        std::vector<Coordinates>::const_iterator _ends_values_it;
    };

    iterator begin() const
    {
        return iterator(_tensor_shapes.begin(), _starts_values.begin(), _ends_values.begin());
    }

    int size() const
    {
        return std::min(_tensor_shapes.size(), std::min(_starts_values.size(), _ends_values.size()));
    }

    void add_config(TensorShape shape, Coordinates starts, Coordinates ends)
    {
        _tensor_shapes.emplace_back(std::move(shape));
        _starts_values.emplace_back(std::move(starts));
        _ends_values.emplace_back(std::move(ends));
    }

protected:
    SliceDataset()                = default;
    SliceDataset(SliceDataset &&) = default;

private:
    std::vector<TensorShape> _tensor_shapes{};
    std::vector<Coordinates> _starts_values{};
    std::vector<Coordinates> _ends_values{};
};

class StridedSliceDataset
{
public:
    using type = std::tuple<TensorShape, Coordinates, Coordinates, BiStrides, int32_t, int32_t, int32_t>;

    struct iterator
    {
        iterator(std::vector<TensorShape>::const_iterator tensor_shapes_it,
                 std::vector<Coordinates>::const_iterator starts_values_it,
                 std::vector<Coordinates>::const_iterator ends_values_it,
                 std::vector<BiStrides>::const_iterator   strides_values_it,
                 std::vector<int32_t>::const_iterator     begin_mask_values_it,
                 std::vector<int32_t>::const_iterator     end_mask_values_it,
                 std::vector<int32_t>::const_iterator     shrink_mask_values_it)
            : _tensor_shapes_it{ std::move(tensor_shapes_it) },
              _starts_values_it{ std::move(starts_values_it) },
              _ends_values_it{ std::move(ends_values_it) },
              _strides_values_it{ std::move(strides_values_it) },
              _begin_mask_values_it{ std::move(begin_mask_values_it) },
              _end_mask_values_it{ std::move(end_mask_values_it) },
              _shrink_mask_values_it{ std::move(shrink_mask_values_it) }
        {
        }

        std::string description() const
        {
            std::stringstream description;
            description << "Shape=" << *_tensor_shapes_it << ":";
            description << "Starts=" << *_starts_values_it << ":";
            description << "Ends=" << *_ends_values_it << ":";
            description << "Strides=" << *_strides_values_it << ":";
            description << "BeginMask=" << *_begin_mask_values_it << ":";
            description << "EndMask=" << *_end_mask_values_it << ":";
            description << "ShrinkMask=" << *_shrink_mask_values_it << ":";
            return description.str();
        }

        StridedSliceDataset::type operator*() const
        {
            return std::make_tuple(*_tensor_shapes_it,
                                   *_starts_values_it, *_ends_values_it, *_strides_values_it,
                                   *_begin_mask_values_it, *_end_mask_values_it, *_shrink_mask_values_it);
        }

        iterator &operator++()
        {
            ++_tensor_shapes_it;
            ++_starts_values_it;
            ++_ends_values_it;
            ++_strides_values_it;
            ++_begin_mask_values_it;
            ++_end_mask_values_it;
            ++_shrink_mask_values_it;

            return *this;
        }

    private:
        std::vector<TensorShape>::const_iterator _tensor_shapes_it;
        std::vector<Coordinates>::const_iterator _starts_values_it;
        std::vector<Coordinates>::const_iterator _ends_values_it;
        std::vector<BiStrides>::const_iterator   _strides_values_it;
        std::vector<int32_t>::const_iterator     _begin_mask_values_it;
        std::vector<int32_t>::const_iterator     _end_mask_values_it;
        std::vector<int32_t>::const_iterator     _shrink_mask_values_it;
    };

    iterator begin() const
    {
        return iterator(_tensor_shapes.begin(),
                        _starts_values.begin(), _ends_values.begin(), _strides_values.begin(),
                        _begin_mask_values.begin(), _end_mask_values.begin(), _shrink_mask_values.begin());
    }

    int size() const
    {
        return std::min(_tensor_shapes.size(), std::min(_starts_values.size(), std::min(_ends_values.size(), _strides_values.size())));
    }

    void add_config(TensorShape shape,
                    Coordinates starts, Coordinates ends, BiStrides strides,
                    int32_t begin_mask = 0, int32_t end_mask = 0, int32_t shrink_mask = 0)
    {
        _tensor_shapes.emplace_back(std::move(shape));
        _starts_values.emplace_back(std::move(starts));
        _ends_values.emplace_back(std::move(ends));
        _strides_values.emplace_back(std::move(strides));
        _begin_mask_values.emplace_back(std::move(begin_mask));
        _end_mask_values.emplace_back(std::move(end_mask));
        _shrink_mask_values.emplace_back(std::move(shrink_mask));
    }

protected:
    StridedSliceDataset()                       = default;
    StridedSliceDataset(StridedSliceDataset &&) = default;

private:
    std::vector<TensorShape> _tensor_shapes{};
    std::vector<Coordinates> _starts_values{};
    std::vector<Coordinates> _ends_values{};
    std::vector<BiStrides>   _strides_values{};
    std::vector<int32_t>     _begin_mask_values{};
    std::vector<int32_t>     _end_mask_values{};
    std::vector<int32_t>     _shrink_mask_values{};
};

class SmallSliceDataset final : public SliceDataset
{
public:
    SmallSliceDataset()
    {
        // 1D
        add_config(TensorShape(15U), Coordinates(4), Coordinates(9));
        add_config(TensorShape(15U), Coordinates(0), Coordinates(-1));
        // 2D
        add_config(TensorShape(15U, 16U), Coordinates(0, 1), Coordinates(5, -1));
        add_config(TensorShape(15U, 16U), Coordinates(4, 1), Coordinates(12, -1));
        // 3D
        add_config(TensorShape(15U, 16U, 4U), Coordinates(0, 1, 2), Coordinates(5, -1, 4));
        add_config(TensorShape(15U, 16U, 4U), Coordinates(0, 1, 2), Coordinates(5, -1, 4));
        // 4D
        add_config(TensorShape(15U, 16U, 4U, 12U), Coordinates(0, 1, 2, 2), Coordinates(5, -1, 4, 5));
    }
};

class LargeSliceDataset final : public SliceDataset
{
public:
    LargeSliceDataset()
    {
        // 1D
        add_config(TensorShape(1025U), Coordinates(128), Coordinates(-100));
        // 2D
        add_config(TensorShape(372U, 68U), Coordinates(128, 7), Coordinates(368, -1));
        // 3D
        add_config(TensorShape(372U, 68U, 12U), Coordinates(128, 7, 2), Coordinates(368, -1, 4));
        // 4D
        add_config(TensorShape(372U, 68U, 7U, 4U), Coordinates(128, 7, 2), Coordinates(368, 17, 5));
    }
};

class SmallStridedSliceDataset final : public StridedSliceDataset
{
public:
    SmallStridedSliceDataset()
    {
        // 1D
        add_config(TensorShape(15U), Coordinates(0), Coordinates(5), BiStrides(2));
        add_config(TensorShape(15U), Coordinates(-1), Coordinates(-8), BiStrides(-2));
        // 2D
        add_config(TensorShape(15U, 16U), Coordinates(0, 1), Coordinates(5, -1), BiStrides(2, 1));
        add_config(TensorShape(15U, 16U), Coordinates(4, 1), Coordinates(12, -1), BiStrides(2, 1), 1);
        // 3D
        add_config(TensorShape(15U, 16U, 4U), Coordinates(0, 1, 2), Coordinates(5, -1, 4), BiStrides(2, 1, 2));
        add_config(TensorShape(15U, 16U, 4U), Coordinates(0, 1, 2), Coordinates(5, -1, 4), BiStrides(2, 1, 2), 0, 1);
        // 4D
        add_config(TensorShape(15U, 16U, 4U, 12U), Coordinates(0, 1, 2, 2), Coordinates(5, -1, 4, 5), BiStrides(2, 1, 2, 3));

        // Shrink axis
        add_config(TensorShape(1U, 3U, 2U, 3U), Coordinates(0, 1, 0, 0), Coordinates(1, 1, 1, 1), BiStrides(1, 1, 1, 1), 0, 15, 6);
        add_config(TensorShape(3U, 2U), Coordinates(0, 0), Coordinates(3U, 1U), BiStrides(1, 1), 0, 0, 2);
        add_config(TensorShape(4U, 7U, 7U), Coordinates(0, 0, 0), Coordinates(1U, 1U, 1U), BiStrides(1, 1, 1), 0, 6, 1);
        add_config(TensorShape(4U, 7U, 7U), Coordinates(0, 1, 0), Coordinates(1U, 1U, 1U), BiStrides(1, 1, 1), 0, 5, 3);
    }
};

class LargeStridedSliceDataset final : public StridedSliceDataset
{
public:
    LargeStridedSliceDataset()
    {
        // 1D
        add_config(TensorShape(1025U), Coordinates(128), Coordinates(-100), BiStrides(20));
        // 2D
        add_config(TensorShape(372U, 68U), Coordinates(128, 7), Coordinates(368, -30), BiStrides(10, 7));
        // 3D
        add_config(TensorShape(372U, 68U, 12U), Coordinates(128, 7, -1), Coordinates(368, -30, -5), BiStrides(14, 7, -2));
        // 4D
        add_config(TensorShape(372U, 68U, 7U, 4U), Coordinates(128, 7, 2), Coordinates(368, -30, 5), BiStrides(20, 7, 2), 1, 1);
    }
};
} // namespace datasets
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_STRIDED_SLICE_DATASET */
