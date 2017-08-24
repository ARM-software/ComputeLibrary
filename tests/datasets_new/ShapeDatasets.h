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
#include "framework/datasets/Datasets.h"

#include <type_traits>

namespace arm_compute
{
namespace test
{
namespace datasets
{
/** Data set containing 1D tensor shapes. */
class Small1DShape final : public framework::dataset::SingletonDataset<TensorShape>
{
public:
    Small1DShape()
        : SingletonDataset("Shape", TensorShape{ 256U })
    {
    }
};

/** Parent type for all for shape datasets. */
using ShapeDataset = framework::dataset::ContainerDataset<std::vector<TensorShape>>;

/** Data set containing small 2D tensor shapes. */
class Small2DShapes final : public ShapeDataset
{
public:
    Small2DShapes()
        : ShapeDataset("Shape",
    {
        TensorShape{ 7U, 7U },
                     TensorShape{ 27U, 13U },
                     TensorShape{ 128U, 64U }
    })
    {
    }
};

/** Data set containing small tensor shapes. */
class SmallShapes final : public ShapeDataset
{
public:
    SmallShapes()
        : ShapeDataset("Shape",
    {
        // Batch size 1
        TensorShape{ 7U, 7U },
                     TensorShape{ 27U, 13U, 2U },
                     TensorShape{ 128U, 64U, 1U, 3U },
                     // Batch size 4
                     TensorShape{ 7U, 7U, 4U },
                     TensorShape{ 27U, 13U, 2U, 4U },
                     // Arbitrary batch size
                     TensorShape{ 7U, 7U, 5U }
    })
    {
    }
};

/** Data set containing large tensor shapes. */
class LargeShapes final : public ShapeDataset
{
public:
    LargeShapes()
        : ShapeDataset("Shape",
    {
        // Batch size 1
        TensorShape{ 1920U, 1080U },
                     TensorShape{ 1245U, 652U, 1U, 3U },
                     TensorShape{ 4160U, 3120U },
                     // Batch size 4
                     TensorShape{ 1245U, 652U, 1U, 4U },
                     // Batch size 8
                     TensorShape{ 1245U, 652U, 1U, 8U },
    })
    {
    }
};

/** Data set containing large 2D tensor shapes. */
class Large2DShapes final : public ShapeDataset
{
public:
    Large2DShapes()
        : ShapeDataset("Shape",
    {
        TensorShape{ 1920U, 1080U },
                     TensorShape{ 1245U, 652U },
                     TensorShape{ 4160U, 3120U }
    })
    {
    }
};

/** Data set containing small tensor shapes for direct convolution. */
class SmallDirectConvolutionShapes final : public ShapeDataset
{
public:
    SmallDirectConvolutionShapes()
        : ShapeDataset("InputShape",
    {
        // Batch size 1
        TensorShape{ 5U, 5U, 3U },
                     TensorShape{ 32U, 37U, 3U },
                     TensorShape{ 13U, 15U, 8U },
                     // Batch size 4
                     TensorShape{ 5U, 5U, 3U, 4U },
                     TensorShape{ 32U, 37U, 3U, 4U },
                     TensorShape{ 13U, 15U, 8U, 4U },
                     // Batch size 8
                     TensorShape{ 5U, 5U, 3U, 8U },
                     TensorShape{ 32U, 37U, 3U, 8U },
                     TensorShape{ 13U, 15U, 8U, 8U },
                     // Arbitrary batch size
                     TensorShape{ 32U, 37U, 3U, 8U }
    })
    {
    }
};
} // namespace datasets
} // namespace test
} // namespace arm_compute
#endif /* __ARM_COMPUTE_TEST_SHAPE_DATASETS_H__ */
