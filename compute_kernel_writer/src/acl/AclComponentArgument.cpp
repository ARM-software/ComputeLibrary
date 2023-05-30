/*
 * Copyright (c) 2023 Arm Limited.
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

#include "acl/AclComponentArgument.h"
#include "ckw/Error.h"

AclComponentArgument::AclComponentArgument()
{
}

AclComponentArgument::AclComponentArgument(ckw::TensorOperand &tensor)
    : _tensor(&tensor)
{
}

AclComponentArgument &AclComponentArgument::init_virtual_tensor(ckw::TileOperand &tile, const ckw::TensorTileSampler &tile_sampler)
{
    CKW_ASSERT(_tile == nullptr);

    _tile         = &tile;
    _tile_sampler = tile_sampler;

    return *this;
}

bool AclComponentArgument::has_tensor() const
{
    return _tensor != nullptr;
}

ckw::TensorOperand &AclComponentArgument::tensor()
{
    CKW_ASSERT(_tensor != nullptr);

    return *_tensor;
}

const ckw::TensorOperand &AclComponentArgument::tensor() const
{
    CKW_ASSERT(_tensor != nullptr);

    return *_tensor;
}

bool AclComponentArgument::has_tile() const
{
    return _tile != nullptr;
}

ckw::TileOperand &AclComponentArgument::tile()
{
    CKW_ASSERT(_tile != nullptr);

    return *_tile;
}

const ckw::TileOperand &AclComponentArgument::tile() const
{
    CKW_ASSERT(_tile != nullptr);

    return *_tile;
}

ckw::TensorTileSampler &AclComponentArgument::tile_sampler()
{
    CKW_ASSERT(_tile != nullptr);

    return _tile_sampler;
}

const ckw::TensorTileSampler &AclComponentArgument::tile_sampler() const
{
    CKW_ASSERT(_tile != nullptr);

    return _tile_sampler;
}
