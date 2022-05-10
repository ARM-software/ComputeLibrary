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
#ifdef ENABLE_EXPERIMENTAL_DYNAMIC_FUSION
#ifndef ARM_COMPUTE_EXPERIMENTAL_DYNAMICFUSION_CLKERNELDESCRIPTORS_H
#define ARM_COMPUTE_EXPERIMENTAL_DYNAMICFUSION_CLKERNELDESCRIPTORS_H

#include "arm_compute/core/experimental/OperatorGraph.h"

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
struct ClDirectConv2dKernelDescriptor
{
    friend bool operator==(const ClDirectConv2dKernelDescriptor &desc0, const ClDirectConv2dKernelDescriptor &desc1)
    {
        return desc0.conv2d == desc1.conv2d;
    }
    Conv2dDescriptor conv2d{};
};

struct ClEltwiseAddKernelDescriptor
{
    friend bool operator==(const ClEltwiseAddKernelDescriptor &desc0, const ClEltwiseAddKernelDescriptor &desc1)
    {
        return desc0.add == desc1.add;
    }
    AddDescriptor add{};
};
struct ClActivationKernelDescriptor
{
    friend bool operator==(const ClActivationKernelDescriptor &, const ClActivationKernelDescriptor &)
    {
        return true;
    }
};

enum class ClippingStrategy
{
    TOP_LEFT,
    TOP_RIGHT,
    BOTTOM_LEFT,
    BOTTOM_RIGHT,
};
/** Component: Store */
struct TileDescriptor
{
    Size2D           tile_dims{};
    Size2D           boundaries{};
    ClippingStrategy clipping{ ClippingStrategy::TOP_LEFT };

    TileDescriptor()
    {
    }

    TileDescriptor(Size2D dims, const Size2D &bound, const ClippingStrategy &clip)
        : tile_dims(dims), boundaries(bound), clipping(clip)
    {
    }

    bool empty() const
    {
        return (tile_dims.area() == 0) || (boundaries.area() == 0);
    }
    friend bool operator==(const TileDescriptor &tile0, const TileDescriptor &tile1)
    {
        return tile0.tile_dims == tile1.tile_dims && tile0.boundaries == tile1.boundaries && tile0.clipping == tile1.clipping;
    }
};
enum class StoreType
{
    VStore,
    VStorePartial,
    StoreRow,
    ConvertStoreRow,
    StoreBlock,
    ConvertStoreBlock,
    StoreRowPartial,
    StoreBlockPartial,
    StoreBlockBoundaryAware,
    StoreVectorSelect,
    TStoreIndirectWidthSelect
};
} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
#endif //ARM_COMPUTE_EXPERIMENTAL_DYNAMICFUSION_CLKERNELDESCRIPTORS_H
#endif /* ENABLE_EXPERIMENTAL_DYNAMIC_FUSION */