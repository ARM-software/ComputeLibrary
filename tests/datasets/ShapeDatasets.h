/*
 * Copyright (c) 2017, 2018 ARM Limited.
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
#include "tests/framework/datasets/Datasets.h"

#include <type_traits>

namespace arm_compute
{
namespace test
{
namespace datasets
{
/** Parent type for all for shape datasets. */
using ShapeDataset = framework::dataset::ContainerDataset<std::vector<TensorShape>>;

/** Data set containing small 1D tensor shapes. */
class Small1DShapes final : public ShapeDataset
{
public:
    Small1DShapes()
        : ShapeDataset("Shape",
    {
        TensorShape{ 256U }
    })
    {
    }
};

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

/** Data set containing small 3D tensor shapes. */
class Small3DShapes final : public ShapeDataset
{
public:
    Small3DShapes()
        : ShapeDataset("Shape",
    {
        TensorShape{ 7U, 7U, 5U },
                     TensorShape{ 27U, 13U, 37U },
                     TensorShape{ 128U, 64U, 21U }
    })
    {
    }
};

/** Data set containing small 4D tensor shapes. */
class Small4DShapes final : public ShapeDataset
{
public:
    Small4DShapes()
        : ShapeDataset("Shape",
    {
        TensorShape{ 7U, 7U, 5U, 3U },
                     TensorShape{ 27U, 13U, 37U, 2U },
                     TensorShape{ 128U, 64U, 21U, 3U }
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
        TensorShape{ 9U, 9U },
                     TensorShape{ 27U, 13U, 2U },
                     TensorShape{ 128U, 64U, 1U, 3U },
                     // Batch size 4
                     TensorShape{ 9U, 9U, 3U, 4U },
                     TensorShape{ 27U, 13U, 2U, 4U },
                     // Arbitrary batch size
                     TensorShape{ 9U, 9U, 3U, 5U }
    })
    {
    }
};

/** Data set containing medium tensor shapes. */
class MediumShapes final : public ShapeDataset
{
public:
    MediumShapes()
        : ShapeDataset("Shape",
    {
        // Batch size 1
        TensorShape{ 37U, 37U },
                     TensorShape{ 27U, 33U, 2U },
                     TensorShape{ 128U, 64U, 1U, 3U },
                     // Batch size 4
                     TensorShape{ 37U, 37U, 3U, 4U },
                     TensorShape{ 27U, 33U, 2U, 4U },
                     // Arbitrary batch size
                     TensorShape{ 37U, 37U, 3U, 5U }
    })
    {
    }
};

/** Data set containing medium 2D tensor shapes. */
class Medium2DShapes final : public ShapeDataset
{
public:
    Medium2DShapes()
        : ShapeDataset("Shape",
    {
        TensorShape{ 42U, 37U },
                     TensorShape{ 57U, 60U },
                     TensorShape{ 128U, 64U },
                     TensorShape{ 83U, 72U }
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
        TensorShape{ 1921U, 1083U },
                     TensorShape{ 641U, 485U, 2U, 3U },
                     TensorShape{ 4159U, 3117U },
                     // Batch size 4
                     TensorShape{ 799U, 595U, 1U, 4U },
    })
    {
    }
};

/** Data set containing large 1D tensor shapes. */
class Large1DShapes final : public ShapeDataset
{
public:
    Large1DShapes()
        : ShapeDataset("Shape",
    {
        TensorShape{ 1921U },
                     TensorShape{ 1245U },
                     TensorShape{ 4160U }
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

/** Data set containing large 3D tensor shapes. */
class Large3DShapes final : public ShapeDataset
{
public:
    Large3DShapes()
        : ShapeDataset("Shape",
    {
        TensorShape{ 320U, 240U, 3U },
                     TensorShape{ 383U, 653U, 2U },
                     TensorShape{ 721U, 123U, 13U }
    })
    {
    }
};

/** Data set containing large 4D tensor shapes. */
class Large4DShapes final : public ShapeDataset
{
public:
    Large4DShapes()
        : ShapeDataset("Shape",
    {
        TensorShape{ 320U, 123U, 3U, 3U },
                     TensorShape{ 383U, 413U, 2U, 3U },
                     TensorShape{ 517U, 123U, 13U, 2U }
    })
    {
    }
};

/** Data set containing small tensor shapes for deconvolution. */
class SmallDeconvolutionShapes final : public ShapeDataset
{
public:
    SmallDeconvolutionShapes()
        : ShapeDataset("InputShape",
    {
        TensorShape{ 4U, 3U, 3U, 2U },
                     TensorShape{ 5U, 5U, 3U },
                     TensorShape{ 11U, 13U, 4U, 3U }
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
        TensorShape{ 35U, 35U, 3U },
                     TensorShape{ 32U, 37U, 3U },
                     // Batch size 4
                     TensorShape{ 32U, 37U, 3U, 4U },
                     // Batch size 8
                     TensorShape{ 32U, 37U, 3U, 8U },
                     TensorShape{ 33U, 35U, 8U, 8U },
                     // Arbitrary batch size
                     TensorShape{ 32U, 37U, 3U, 8U }
    })
    {
    }
};

/** Data set containing 2D tensor shapes for DepthConcatenateLayer. */
class DepthConcatenateLayerShapes final : public ShapeDataset
{
public:
    DepthConcatenateLayerShapes()
        : ShapeDataset("Shape",
    {
        TensorShape{ 322U, 243U },
                     TensorShape{ 463U, 879U },
                     TensorShape{ 416U, 651U }
    })
    {
    }
};

/** Data set containing global pooling tensor shapes. */
class GlobalPoolingShapes final : public ShapeDataset
{
public:
    GlobalPoolingShapes()
        : ShapeDataset("Shape",
    {
        // Batch size 1
        TensorShape{ 9U, 9U },
                     TensorShape{ 13U, 13U, 2U },
                     TensorShape{ 27U, 27U, 1U, 3U },
                     // Batch size 4
                     TensorShape{ 31U, 31U, 3U, 4U },
                     TensorShape{ 34U, 34U, 2U, 4U }
    })
    {
    }
};

/** Data set containing small softmax layer shapes. */
class SoftmaxLayerSmallShapes final : public ShapeDataset
{
public:
    SoftmaxLayerSmallShapes()
        : ShapeDataset("Shape",
    {
        TensorShape{ 9U, 9U },
                     TensorShape{ 256U, 10U, 2U },
                     TensorShape{ 353U, 8U, 2U, 2U },
                     TensorShape{ 512U, 7U, 2U, 2U },
                     TensorShape{ 633U, 10U, 1U, 2U },
                     TensorShape{ 781U, 5U, 2U },
    })
    {
    }
};

/** Data set containing large softmax layer shapes. */
class SoftmaxLayerLargeShapes final : public ShapeDataset
{
public:
    SoftmaxLayerLargeShapes()
        : ShapeDataset("Shape",
    {
        TensorShape{ 1000U, 10U },
                     TensorShape{ 3989U, 10U, 2U },
                     TensorShape{ 4098U, 8U, 1U, 2U },
                     TensorShape{ 7339U, 11U },
    })
    {
    }
};

/** Data set containing 2D tensor shapes relative to an image size. */
class SmallImageShapes final : public ShapeDataset
{
public:
    SmallImageShapes()
        : ShapeDataset("Shape",
    {
        TensorShape{ 640U, 480U },
                     TensorShape{ 800U, 600U },
                     TensorShape{ 1200U, 800U }
    })
    {
    }
};

/** Data set containing 2D tensor shapes relative to an image size. */
class LargeImageShapes final : public ShapeDataset
{
public:
    LargeImageShapes()
        : ShapeDataset("Shape",
    {
        TensorShape{ 1920U, 1080U },
                     TensorShape{ 2560U, 1536U },
                     TensorShape{ 3584U, 2048U }
    })
    {
    }
};
} // namespace datasets
} // namespace test
} // namespace arm_compute
#endif /* __ARM_COMPUTE_TEST_SHAPE_DATASETS_H__ */
