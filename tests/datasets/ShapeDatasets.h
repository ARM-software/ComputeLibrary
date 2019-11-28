/*
 * Copyright (c) 2017-2019 ARM Limited.
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

/** Data set containing tiny 1D tensor shapes. */
class Tiny1DShapes final : public ShapeDataset
{
public:
    Tiny1DShapes()
        : ShapeDataset("Shape",
    {
        TensorShape{ 2U },
                     TensorShape{ 3U },
    })
    {
    }
};

/** Data set containing small 1D tensor shapes. */
class Small1DShapes final : public ShapeDataset
{
public:
    Small1DShapes()
        : ShapeDataset("Shape",
    {
        TensorShape{ 128U },
                     TensorShape{ 256U },
                     TensorShape{ 512U },
                     TensorShape{ 1024U }
    })
    {
    }
};

/** Data set containing tiny 2D tensor shapes. */
class Tiny2DShapes final : public ShapeDataset
{
public:
    Tiny2DShapes()
        : ShapeDataset("Shape",
    {
        TensorShape{ 7U, 7U },
                     TensorShape{ 11U, 13U },
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

/** Data set containing tiny 3D tensor shapes. */
class Tiny3DShapes final : public ShapeDataset
{
public:
    Tiny3DShapes()
        : ShapeDataset("Shape",
    {
        TensorShape{ 7U, 7U, 5U },
                     TensorShape{ 23U, 13U, 9U },
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
        TensorShape{ 1U, 7U, 7U },
                     TensorShape{ 2U, 5U, 4U },

                     TensorShape{ 7U, 7U, 5U },
                     TensorShape{ 27U, 13U, 37U },
    })
    {
    }
};

/** Data set containing tiny 4D tensor shapes. */
class Tiny4DShapes final : public ShapeDataset
{
public:
    Tiny4DShapes()
        : ShapeDataset("Shape",
    {
        TensorShape{ 7U, 7U, 5U, 3U },
                     TensorShape{ 17U, 13U, 7U, 2U },
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
        TensorShape{ 2U, 7U, 1U, 3U },
                     TensorShape{ 7U, 7U, 5U, 3U },
                     TensorShape{ 27U, 13U, 37U, 2U },
                     TensorShape{ 128U, 64U, 21U, 3U }
    })
    {
    }
};

/** Data set containing small tensor shapes. */
class TinyShapes final : public ShapeDataset
{
public:
    TinyShapes()
        : ShapeDataset("Shape",
    {
        // Batch size 1
        TensorShape{ 9U, 9U },
                     TensorShape{ 27U, 13U, 2U },
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
        TensorShape{ 11U, 11U },
                     TensorShape{ 27U, 13U, 7U },
                     TensorShape{ 31U, 27U, 17U, 2U },
                     // Batch size 4
                     TensorShape{ 27U, 13U, 2U, 4U },
                     // Arbitrary batch size
                     TensorShape{ 11U, 11U, 3U, 5U }
    })
    {
    }
};

/** Data set containing pairs of small tensor shapes that are broadcast compatible. */
class SmallShapesBroadcast final : public framework::dataset::ZipDataset<ShapeDataset, ShapeDataset>
{
public:
    SmallShapesBroadcast()
        : ZipDataset<ShapeDataset, ShapeDataset>(
              ShapeDataset("Shape0",
    {
        TensorShape{ 9U, 9U },
                     TensorShape{ 27U, 13U, 2U },
                     TensorShape{ 128U, 1U, 5U, 3U },
                     TensorShape{ 9U, 9U, 3U, 4U },
                     TensorShape{ 27U, 13U, 2U, 4U },
                     TensorShape{ 1U, 1U, 1U, 5U },
                     TensorShape{ 1U, 16U, 10U, 2U, 128U },
                     TensorShape{ 1U, 16U, 10U, 2U, 128U }
    }),
    ShapeDataset("Shape1",
    {
        TensorShape{ 9U, 1U, 2U },
        TensorShape{ 1U, 13U, 2U },
        TensorShape{ 128U, 64U, 1U, 3U },
        TensorShape{ 9U, 1U, 3U },
        TensorShape{ 1U },
        TensorShape{ 9U, 9U, 3U, 5U },
        TensorShape{ 1U, 1U, 1U, 1U, 128U },
        TensorShape{ 128U }
    }))
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
                     TensorShape{ 83U, 72U },
                     TensorShape{ 40U, 40U }
    })
    {
    }
};

/** Data set containing medium 3D tensor shapes. */
class Medium3DShapes final : public ShapeDataset
{
public:
    Medium3DShapes()
        : ShapeDataset("Shape",
    {
        TensorShape{ 42U, 37U, 8U },
                     TensorShape{ 57U, 60U, 13U },
                     TensorShape{ 83U, 72U, 14U }
    })
    {
    }
};

/** Data set containing medium 4D tensor shapes. */
class Medium4DShapes final : public ShapeDataset
{
public:
    Medium4DShapes()
        : ShapeDataset("Shape",
    {
        TensorShape{ 42U, 37U, 8U, 15U },
                     TensorShape{ 57U, 60U, 13U, 8U },
                     TensorShape{ 83U, 72U, 14U, 5U }
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
        TensorShape{ 582U, 131U, 1U, 4U },
    })
    {
    }
};

/** Data set containing pairs of large tensor shapes that are broadcast compatible. */
class LargeShapesBroadcast final : public framework::dataset::ZipDataset<ShapeDataset, ShapeDataset>
{
public:
    LargeShapesBroadcast()
        : ZipDataset<ShapeDataset, ShapeDataset>(
              ShapeDataset("Shape0",
    {
        TensorShape{ 1921U, 541U },
                     TensorShape{ 1U, 485U, 2U, 3U },
                     TensorShape{ 4159U, 1U },
                     TensorShape{ 799U }
    }),
    ShapeDataset("Shape1",
    {
        TensorShape{ 1921U, 1U, 2U },
        TensorShape{ 641U, 1U, 2U, 3U },
        TensorShape{ 1U, 127U, 25U },
        TensorShape{ 799U, 595U, 1U, 4U }
    }))
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

/** Data set containing small 3x3 tensor shapes. */
class Small3x3Shapes final : public ShapeDataset
{
public:
    Small3x3Shapes()
        : ShapeDataset("Shape",
    {
        TensorShape{ 3U, 3U, 7U, 4U },
                     TensorShape{ 3U, 3U, 4U, 13U },
                     TensorShape{ 3U, 3U, 3U, 5U },
    })
    {
    }
};

/** Data set containing small 3x1 tensor shapes. */
class Small3x1Shapes final : public ShapeDataset
{
public:
    Small3x1Shapes()
        : ShapeDataset("Shape",
    {
        TensorShape{ 3U, 1U, 7U, 4U },
                     TensorShape{ 3U, 1U, 4U, 13U },
                     TensorShape{ 3U, 1U, 3U, 5U },
    })
    {
    }
};

/** Data set containing small 1x3 tensor shapes. */
class Small1x3Shapes final : public ShapeDataset
{
public:
    Small1x3Shapes()
        : ShapeDataset("Shape",
    {
        TensorShape{ 1U, 3U, 7U, 4U },
                     TensorShape{ 1U, 3U, 4U, 13U },
                     TensorShape{ 1U, 3U, 3U, 5U },
    })
    {
    }
};

/** Data set containing large 3x3 tensor shapes. */
class Large3x3Shapes final : public ShapeDataset
{
public:
    Large3x3Shapes()
        : ShapeDataset("Shape",
    {
        TensorShape{ 3U, 3U, 32U, 64U },
                     TensorShape{ 3U, 3U, 51U, 13U },
                     TensorShape{ 3U, 3U, 53U, 47U },
    })
    {
    }
};

/** Data set containing large 3x1 tensor shapes. */
class Large3x1Shapes final : public ShapeDataset
{
public:
    Large3x1Shapes()
        : ShapeDataset("Shape",
    {
        TensorShape{ 3U, 1U, 32U, 64U },
                     TensorShape{ 3U, 1U, 51U, 13U },
                     TensorShape{ 3U, 1U, 53U, 47U },
    })
    {
    }
};

/** Data set containing large 1x3 tensor shapes. */
class Large1x3Shapes final : public ShapeDataset
{
public:
    Large1x3Shapes()
        : ShapeDataset("Shape",
    {
        TensorShape{ 1U, 3U, 32U, 64U },
                     TensorShape{ 1U, 3U, 51U, 13U },
                     TensorShape{ 1U, 3U, 53U, 47U },
    })
    {
    }
};

/** Data set containing small 5x5 tensor shapes. */
class Small5x5Shapes final : public ShapeDataset
{
public:
    Small5x5Shapes()
        : ShapeDataset("Shape",
    {
        TensorShape{ 5U, 5U, 7U, 4U },
                     TensorShape{ 5U, 5U, 4U, 13U },
                     TensorShape{ 5U, 5U, 3U, 5U },
    })
    {
    }
};

/** Data set containing large 5x5 tensor shapes. */
class Large5x5Shapes final : public ShapeDataset
{
public:
    Large5x5Shapes()
        : ShapeDataset("Shape",
    {
        TensorShape{ 5U, 5U, 32U, 64U },
                     TensorShape{ 5U, 5U, 51U, 13U },
                     TensorShape{ 5U, 5U, 53U, 47U },
    })
    {
    }
};

/** Data set containing small 5x1 tensor shapes. */
class Small5x1Shapes final : public ShapeDataset
{
public:
    Small5x1Shapes()
        : ShapeDataset("Shape",
    {
        TensorShape{ 5U, 1U, 7U, 4U },
                     TensorShape{ 5U, 1U, 4U, 13U },
                     TensorShape{ 5U, 1U, 3U, 5U },
    })
    {
    }
};

/** Data set containing large 5x1 tensor shapes. */
class Large5x1Shapes final : public ShapeDataset
{
public:
    Large5x1Shapes()
        : ShapeDataset("Shape",
    {
        TensorShape{ 5U, 1U, 32U, 64U },
                     TensorShape{ 5U, 1U, 51U, 13U },
                     TensorShape{ 5U, 1U, 53U, 47U },
    })
    {
    }
};

/** Data set containing small 1x5 tensor shapes. */
class Small1x5Shapes final : public ShapeDataset
{
public:
    Small1x5Shapes()
        : ShapeDataset("Shape",
    {
        TensorShape{ 1U, 5U, 7U, 4U },
                     TensorShape{ 1U, 5U, 4U, 13U },
                     TensorShape{ 1U, 5U, 3U, 5U },
    })
    {
    }
};

/** Data set containing large 1x5 tensor shapes. */
class Large1x5Shapes final : public ShapeDataset
{
public:
    Large1x5Shapes()
        : ShapeDataset("Shape",
    {
        TensorShape{ 1U, 5U, 32U, 64U },
                     TensorShape{ 1U, 5U, 51U, 13U },
                     TensorShape{ 1U, 5U, 53U, 47U },
    })
    {
    }
};

/** Data set containing small 1x7 tensor shapes. */
class Small1x7Shapes final : public ShapeDataset
{
public:
    Small1x7Shapes()
        : ShapeDataset("Shape",
    {
        TensorShape{ 1U, 7U, 7U, 4U },
                     TensorShape{ 1U, 7U, 4U, 13U },
                     TensorShape{ 1U, 7U, 3U, 5U },
    })
    {
    }
};

/** Data set containing large 1x7 tensor shapes. */
class Large1x7Shapes final : public ShapeDataset
{
public:
    Large1x7Shapes()
        : ShapeDataset("Shape",
    {
        TensorShape{ 1U, 7U, 32U, 64U },
                     TensorShape{ 1U, 7U, 51U, 13U },
                     TensorShape{ 1U, 7U, 53U, 47U },
    })
    {
    }
};

/** Data set containing small 7x7 tensor shapes. */
class Small7x7Shapes final : public ShapeDataset
{
public:
    Small7x7Shapes()
        : ShapeDataset("Shape",
    {
        TensorShape{ 7U, 7U, 7U, 4U },
                     TensorShape{ 7U, 7U, 4U, 13U },
                     TensorShape{ 7U, 7U, 3U, 5U },
    })
    {
    }
};

/** Data set containing large 7x7 tensor shapes. */
class Large7x7Shapes final : public ShapeDataset
{
public:
    Large7x7Shapes()
        : ShapeDataset("Shape",
    {
        TensorShape{ 7U, 7U, 32U, 64U },
                     TensorShape{ 7U, 7U, 51U, 13U },
                     TensorShape{ 7U, 7U, 53U, 47U },
    })
    {
    }
};

/** Data set containing small 7x1 tensor shapes. */
class Small7x1Shapes final : public ShapeDataset
{
public:
    Small7x1Shapes()
        : ShapeDataset("Shape",
    {
        TensorShape{ 7U, 1U, 7U, 4U },
                     TensorShape{ 7U, 1U, 4U, 13U },
                     TensorShape{ 7U, 1U, 3U, 5U },
    })
    {
    }
};

/** Data set containing large 7x1 tensor shapes. */
class Large7x1Shapes final : public ShapeDataset
{
public:
    Large7x1Shapes()
        : ShapeDataset("Shape",
    {
        TensorShape{ 7U, 1U, 32U, 64U },
                     TensorShape{ 7U, 1U, 51U, 13U },
                     TensorShape{ 7U, 1U, 53U, 47U },
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
        TensorShape{ 5U, 4U, 3U, 2U },
                     TensorShape{ 5U, 5U, 3U },
                     TensorShape{ 11U, 13U, 4U, 3U }
    })
    {
    }
};

/** Data set containing tiny tensor shapes for direct convolution. */
class TinyDirectConvolutionShapes final : public ShapeDataset
{
public:
    TinyDirectConvolutionShapes()
        : ShapeDataset("InputShape",
    {
        // Batch size 1
        TensorShape{ 11U, 13U, 3U },
                     TensorShape{ 7U, 27U, 3U }
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
        TensorShape{ 32U, 37U, 3U },
                     // Batch size 4
                     TensorShape{ 32U, 37U, 3U, 4U },
    })
    {
    }
};

/** Data set containing small tensor shapes for direct convolution. */
class SmallDirectConvolutionTensorShiftShapes final : public ShapeDataset
{
public:
    SmallDirectConvolutionTensorShiftShapes()
        : ShapeDataset("InputShape",
    {
        // Batch size 1
        TensorShape{ 32U, 37U, 3U },
                     // Batch size 4
                     TensorShape{ 32U, 37U, 3U, 4U },
                     // Arbitrary batch size
                     TensorShape{ 32U, 37U, 3U, 8U }
    })
    {
    }
};

/** Data set containing small grouped im2col tensor shapes. */
class GroupedIm2ColSmallShapes final : public ShapeDataset
{
public:
    GroupedIm2ColSmallShapes()
        : ShapeDataset("Shape",
    {
        TensorShape{ 11U, 11U, 48U },
                     TensorShape{ 27U, 13U, 24U },
                     TensorShape{ 128U, 64U, 12U, 3U },
                     TensorShape{ 11U, 11U, 48U, 4U },
                     TensorShape{ 27U, 13U, 24U, 4U },
                     TensorShape{ 11U, 11U, 48U, 5U }
    })
    {
    }
};

/** Data set containing large grouped im2col tensor shapes. */
class GroupedIm2ColLargeShapes final : public ShapeDataset
{
public:
    GroupedIm2ColLargeShapes()
        : ShapeDataset("Shape",
    {
        TensorShape{ 153U, 231U, 12U },
                     TensorShape{ 123U, 191U, 12U, 2U },
    })
    {
    }
};

/** Data set containing small grouped weights tensor shapes. */
class GroupedWeightsSmallShapes final : public ShapeDataset
{
public:
    GroupedWeightsSmallShapes()
        : ShapeDataset("Shape",
    {
        TensorShape{ 3U, 3U, 48U, 120U },
                     TensorShape{ 1U, 3U, 24U, 240U },
                     TensorShape{ 3U, 1U, 12U, 480U },
                     TensorShape{ 5U, 5U, 48U, 120U },
                     TensorShape{ 1U, 5U, 24U, 240U },
                     TensorShape{ 5U, 1U, 48U, 480U }
    })
    {
    }
};

/** Data set containing large grouped weights tensor shapes. */
class GroupedWeightsLargeShapes final : public ShapeDataset
{
public:
    GroupedWeightsLargeShapes()
        : ShapeDataset("Shape",
    {
        TensorShape{ 9U, 9U, 96U, 240U },
                     TensorShape{ 7U, 9U, 48U, 480U },
                     TensorShape{ 9U, 7U, 24U, 960U },
                     TensorShape{ 13U, 13U, 96U, 240U },
                     TensorShape{ 11U, 13U, 48U, 480U },
                     TensorShape{ 13U, 11U, 24U, 960U }
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

/** Data set containing tensor shapes for ConcatenateLayer. */
class ConcatenateLayerShapes final : public ShapeDataset
{
public:
    ConcatenateLayerShapes()
        : ShapeDataset("Shape",
    {
        TensorShape{ 232U, 65U, 3U },
                     TensorShape{ 432U, 65U, 3U },
                     TensorShape{ 124U, 65U, 3U },
                     TensorShape{ 124U, 65U, 3U, 4U }
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
/** Data set containing tiny softmax layer shapes. */
class SoftmaxLayerTinyShapes final : public ShapeDataset
{
public:
    SoftmaxLayerTinyShapes()
        : ShapeDataset("Shape",
    {
        TensorShape{ 9U, 9U },
                     TensorShape{ 128U, 10U },
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
                     TensorShape{ 256U, 10U },
                     TensorShape{ 353U, 8U },
                     TensorShape{ 781U, 5U },
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
                     TensorShape{ 3989U, 10U },
                     TensorShape{ 7339U, 11U },

    })
    {
    }
};

/** Data set containing large and small softmax layer 4D shapes. */
class SoftmaxLayer4DShapes final : public ShapeDataset
{
public:
    SoftmaxLayer4DShapes()
        : ShapeDataset("Shape",
    {
        TensorShape{ 9U, 9U, 9U, 9U },
                     TensorShape{ 256U, 10U, 1U, 9U },
                     TensorShape{ 353U, 8U, 2U },
                     TensorShape{ 781U, 5U, 2U, 2U },
                     TensorShape{ 781U, 11U, 1U, 2U },
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

/** Data set containing small YOLO tensor shapes. */
class SmallYOLOShapes final : public ShapeDataset
{
public:
    SmallYOLOShapes()
        : ShapeDataset("Shape",
    {
        // Batch size 1
        TensorShape{ 11U, 11U, 270U },
                     TensorShape{ 27U, 13U, 90U },
                     TensorShape{ 13U, 12U, 45U, 2U },
    })
    {
    }
};

/** Data set containing large YOLO tensor shapes. */
class LargeYOLOShapes final : public ShapeDataset
{
public:
    LargeYOLOShapes()
        : ShapeDataset("Shape",
    {
        TensorShape{ 24U, 23U, 270U },
                     TensorShape{ 51U, 63U, 90U, 2U },
                     TensorShape{ 76U, 91U, 45U, 3U }
    })
    {
    }
};

/** Data set containing small tensor shapes to be used with the GEMM reshaping kernel */
class SmallGEMMReshape2DShapes final : public ShapeDataset
{
public:
    SmallGEMMReshape2DShapes()
        : ShapeDataset("Shape",
    {
        TensorShape{ 63U, 72U },
    })
    {
    }
};

/** Data set containing small tensor shapes to be used with the GEMM reshaping kernel when the input has to be reinterpreted as 3D */
class SmallGEMMReshape3DShapes final : public ShapeDataset
{
public:
    SmallGEMMReshape3DShapes()
        : ShapeDataset("Shape",
    {
        TensorShape{ 63U, 9U, 8U },
    })
    {
    }
};

/** Data set containing large tensor shapes to be used with the GEMM reshaping kernel */
class LargeGEMMReshape2DShapes final : public ShapeDataset
{
public:
    LargeGEMMReshape2DShapes()
        : ShapeDataset("Shape",
    {
        TensorShape{ 16U, 27U },
                     TensorShape{ 533U, 171U },
                     TensorShape{ 345U, 612U }
    })
    {
    }
};

/** Data set containing large tensor shapes to be used with the GEMM reshaping kernel when the input has to be reinterpreted as 3D */
class LargeGEMMReshape3DShapes final : public ShapeDataset
{
public:
    LargeGEMMReshape3DShapes()
        : ShapeDataset("Shape",
    {
        TensorShape{ 16U, 3U, 9U },
                     TensorShape{ 533U, 19U, 9U },
                     TensorShape{ 345U, 34U, 18U }
    })
    {
    }
};

/** Data set containing small 2D tensor shapes. */
class Small2DNonMaxSuppressionShapes final : public ShapeDataset
{
public:
    Small2DNonMaxSuppressionShapes()
        : ShapeDataset("Shape",
    {
        TensorShape{ 4U, 7U },
                     TensorShape{ 4U, 13U },
                     TensorShape{ 4U, 64U }
    })
    {
    }
};

/** Data set containing large 2D tensor shapes. */
class Large2DNonMaxSuppressionShapes final : public ShapeDataset
{
public:
    Large2DNonMaxSuppressionShapes()
        : ShapeDataset("Shape",
    {
        TensorShape{ 4U, 207U },
                     TensorShape{ 4U, 113U },
                     TensorShape{ 4U, 264U }
    })
    {
    }
};

} // namespace datasets
} // namespace test
} // namespace arm_compute
#endif /* __ARM_COMPUTE_TEST_SHAPE_DATASETS_H__ */
