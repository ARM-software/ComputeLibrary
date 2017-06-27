/*
 * Copyright (c) 2017 ARM Limited.
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
#ifndef __ARM_COMPUTE_TEST_SHAPE_DATASETS_H__
#define __ARM_COMPUTE_TEST_SHAPE_DATASETS_H__

#include "arm_compute/core/TensorShape.h"

#include <type_traits>

#ifdef BOOST
#include "boost_wrapper.h"
#endif /* BOOST */

namespace arm_compute
{
namespace test
{
/** Abstract data set containing tensor shapes.
 *
 * Can be used as input for Boost data test cases to automatically run a test
 * case on different tensor shapes.
 */
template <unsigned int Size>
class ShapeDataset
{
public:
    /** Type of the samples in the data set. */
    using sample = TensorShape;

    /** Dimensionality of the data set. */
    enum
    {
        arity = 1
    };

    /** Number of samples in the data set. */
#ifdef BOOST
    boost::unit_test::data::size_t size() const
#else  /* BOOST */
    unsigned int size() const
#endif /* BOOST */
    {
        return _shapes.size();
    }

    /** Type of the iterator used to step through all samples in the data set.
     * Needs to support operator*() and operator++() which a pointer does.
     */
    using iterator = const TensorShape *;

    /** Iterator to the first sample in the data set. */
    iterator begin() const
    {
        return _shapes.data();
    }

protected:
    /** Protected constructor to make the class abstract. */
    template <typename... Ts>
    ShapeDataset(Ts... shapes)
        : _shapes{ { shapes... } }
    {
    }

    /** Protected destructor to prevent deletion of derived class through a
     * pointer to the base class.
     */
    ~ShapeDataset() = default;

private:
    std::array<TensorShape, Size> _shapes;
};

/** Data set containing one 1D tensor shape. */
class Small1DShape final : public ShapeDataset<1>
{
public:
    Small1DShape()
        : ShapeDataset(TensorShape(256U))
    {
    }
};

/** Data set containing two small 2D tensor shapes. */
class Small2DShapes final : public ShapeDataset<2>
{
public:
    Small2DShapes()
        : ShapeDataset(TensorShape(17U, 17U),
                       TensorShape(640U, 480U))
    {
    }
};

/** Data set containing small tensor shapes. */
class SmallShapes final : public ShapeDataset<3>
{
public:
    SmallShapes()
        : ShapeDataset(TensorShape(7U, 7U),
                       TensorShape(27U, 13U, 2U),
                       TensorShape(128U, 64U, 1U, 3U))
    {
    }
};

/** Data set containing large tensor shapes. */
class LargeShapes final : public ShapeDataset<3>
{
public:
    LargeShapes()
        : ShapeDataset(TensorShape(1920U, 1080U),
                       TensorShape(1245U, 652U, 1U, 3U),
                       TensorShape(4160U, 3120U))
    {
    }
};

/** Data set containing two 2D large tensor shapes. */
class Large2DShapes final : public ShapeDataset<2>
{
public:
    Large2DShapes()
        : ShapeDataset(TensorShape(1920U, 1080U),
                       TensorShape(4160U, 3120U))
    {
    }
};
} // namespace test
} // namespace arm_compute
#endif /* __ARM_COMPUTE_TEST_SHAPE_DATASETS_H__ */
