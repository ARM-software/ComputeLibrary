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

#include "ckw/TensorTileSampler.h"

#include "ckw/TileOperand.h"
#include "ckw/types/TensorSamplerTypes.h"

namespace ckw
{

TensorTileSampler::TensorTileSampler()
{
}

TensorTileSampler::TensorTileSampler(TileOperand              &x,
                                     TileOperand              &y,
                                     TileOperand              &z,
                                     TileOperand              &b,
                                     TensorSamplerFormat       format,
                                     TensorSamplerAddressModeX address_mode_x,
                                     TensorSamplerAddressModeY address_mode_y,
                                     TensorSamplerAddressModeZ address_mode_z)
    : _x(&x),
      _y(&y),
      _z(&z),
      _b(&b),
      _height(0),
      _width(0),
      _format(format),
      _address_mode_x(address_mode_x),
      _address_mode_y(address_mode_y),
      _address_mode_z(address_mode_z)
{
}

TensorTileSampler::TensorTileSampler(TileOperand              &x,
                                     TileOperand              &y,
                                     TileOperand              &z,
                                     TileOperand              &b,
                                     int32_t                   height,
                                     int32_t                   width,
                                     TensorSamplerFormat       format,
                                     TensorSamplerAddressModeX address_mode_x,
                                     TensorSamplerAddressModeY address_mode_y,
                                     TensorSamplerAddressModeZ address_mode_z)
    : _x(&x),
      _y(&y),
      _z(&z),
      _b(&b),
      _height(height),
      _width(width),
      _format(format),
      _address_mode_x(address_mode_x),
      _address_mode_y(address_mode_y),
      _address_mode_z(address_mode_z)
{
}

const TileOperand &TensorTileSampler::x() const
{
    return *_x;
}

TensorTileSampler &TensorTileSampler::x(TileOperand &x)
{
    _x = &x;
    return *this;
}

const TileOperand &TensorTileSampler::y() const
{
    return *_y;
}

TensorTileSampler &TensorTileSampler::y(TileOperand &y)
{
    _y = &y;
    return *this;
}

const TileOperand &TensorTileSampler::z() const
{
    return *_z;
}

TensorTileSampler &TensorTileSampler::z(TileOperand &z)
{
    _z = &z;
    return *this;
}

const TileOperand &TensorTileSampler::b() const
{
    return *_b;
}

TensorTileSampler &TensorTileSampler::b(TileOperand &b)
{
    _b = &b;
    return *this;
}

int32_t TensorTileSampler::width() const
{
    return _width;
}

TensorTileSampler &TensorTileSampler::width(int32_t width)
{
    _width = width;
    return *this;
}

int32_t TensorTileSampler::height() const
{
    return _height;
}

TensorTileSampler &TensorTileSampler::height(int32_t height)
{
    _height = height;
    return *this;
}

TensorSamplerFormat TensorTileSampler::format() const
{
    return _format;
}

TensorTileSampler &TensorTileSampler::format(TensorSamplerFormat format)
{
    _format = format;
    return *this;
}

TensorSamplerAddressModeX TensorTileSampler::address_mode_x() const
{
    return _address_mode_x;
}

TensorTileSampler &TensorTileSampler::address_mode_x(TensorSamplerAddressModeX address_mode_x)
{
    _address_mode_x = address_mode_x;
    return *this;
}

TensorSamplerAddressModeY TensorTileSampler::address_mode_y() const
{
    return _address_mode_y;
}

TensorTileSampler &TensorTileSampler::address_mode_y(TensorSamplerAddressModeY address_mode_y)
{
    _address_mode_y = address_mode_y;
    return *this;
}

TensorSamplerAddressModeZ TensorTileSampler::address_mode_z() const
{
    return _address_mode_z;
}

TensorTileSampler &TensorTileSampler::address_mode_z(TensorSamplerAddressModeZ address_mode_z)
{
    _address_mode_z = address_mode_z;
    return *this;
}

} // namespace ckw
