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

#include "ckw/TensorSampler.h"

namespace ckw
{

TensorSampler::TensorSampler(TensorStorageType         storage,
                             TensorSamplerFormat       format,
                             TensorSamplerAddressModeX address_mode_x,
                             TensorSamplerAddressModeY address_mode_y,
                             TensorSamplerAddressModeZ address_mode_z)
    : _storage(storage),
      _format(format),
      _address_mode_x(address_mode_x),
      _address_mode_y(address_mode_y),
      _address_mode_z(address_mode_z)
{
}

TensorStorageType TensorSampler::storage() const
{
    return _storage;
}

TensorSampler &TensorSampler::storage(TensorStorageType storage)
{
    _storage = storage;
    return *this;
}

/** Get the format of the tensor. */
TensorSamplerFormat TensorSampler::format() const
{
    return _format;
}

/** Set the format of the tensor. */
TensorSampler &TensorSampler::format(TensorSamplerFormat format)
{
    _format = format;
    return *this;
}

/** Get the address mode of the x dimension. */
TensorSamplerAddressModeX TensorSampler::address_mode_x() const
{
    return _address_mode_x;
}

/** Set the address mode of the x-dimension. */
TensorSampler &TensorSampler::address_mode_x(TensorSamplerAddressModeX address_mode_x)
{
    _address_mode_x = address_mode_x;
    return *this;
}

/** Get the address mode of the y dimension. */
TensorSamplerAddressModeY TensorSampler::address_mode_y() const
{
    return _address_mode_y;
}

/** Set the address mode of the y dimension. */
TensorSampler &TensorSampler::address_mode_y(TensorSamplerAddressModeY address_mode_y)
{
    _address_mode_y = address_mode_y;
    return *this;
}

/** Get the address mode of the z dimension. */
TensorSamplerAddressModeZ TensorSampler::address_mode_z() const
{
    return _address_mode_z;
}

/** Set the address mode of the z dimension. */
TensorSampler &TensorSampler::address_mode_z(TensorSamplerAddressModeZ address_mode_z)
{
    _address_mode_z = address_mode_z;
    return *this;
}

} // namespace ckw
