/*
 * Copyright (c) 2023-2024 Arm Limited.
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

#include "src/dynamic_fusion/sketch/gpu/ckw_driver/GpuCkwComponentArgument.h"

#include "compute_kernel_writer/include/ckw/Error.h"

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{

GpuCkwComponentArgument::GpuCkwComponentArgument(ckw::TensorOperand tensor) : _tensor(tensor)
{
}

GpuCkwComponentArgument &GpuCkwComponentArgument::init_virtual_tensor(ckw::TileOperand         &tile,
                                                                      const ckw::TensorSampler &sampler)
{
    CKW_ASSERT(_tile == nullptr);

    _tile    = tile;
    _sampler = sampler;

    return *this;
}

bool GpuCkwComponentArgument::has_tensor() const
{
    return _tensor.is_valid();
}

ckw::TensorOperand &GpuCkwComponentArgument::tensor()
{
    CKW_ASSERT(_tensor.is_valid());

    return _tensor;
}

const ckw::TensorOperand &GpuCkwComponentArgument::tensor() const
{
    CKW_ASSERT(_tensor.is_valid());

    return _tensor;
}

bool GpuCkwComponentArgument::has_tile() const
{
    return _tile.is_valid();
}

ckw::TileOperand &GpuCkwComponentArgument::tile()
{
    CKW_ASSERT(_tile.is_valid());

    return _tile;
}

const ckw::TileOperand &GpuCkwComponentArgument::tile() const
{
    CKW_ASSERT(_tile.is_valid());

    return _tile;
}

ckw::TensorSampler &GpuCkwComponentArgument::tensor_sampler()
{
    CKW_ASSERT(_tile.is_valid());

    return _sampler;
}

const ckw::TensorSampler &GpuCkwComponentArgument::tensor_sampler() const
{
    CKW_ASSERT(_tile.is_valid());

    return _sampler;
}

} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
