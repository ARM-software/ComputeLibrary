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

#ifndef CKW_INCLUDE_CKW_TENSORSAMPLER_H
#define CKW_INCLUDE_CKW_TENSORSAMPLER_H

#include "ckw/types/TensorStorageType.h"
#include "ckw/types/TensorSamplerTypes.h"

namespace ckw
{

/** Tensor sampler
 *
 * It contains information about how the tensor is sampled. It can be used to
 * tell how a tile should be stored to tensor memory, and how a tensor should be
 * sampled to get the values stored in a tile. Where to sample the tensor is
 * defined with the coordinates respecting the addressing modes, storage type and
 * the tensor format defined in this class.
 */
class TensorSampler
{
public:
    /** Initialize a new instance of @ref TensorSampler class. */
    TensorSampler();

    /** Initialize a new instance of @ref TensorSampler class.
     *
     * @param[in] storage        Tensor storage to load/store the tensor from/to
     * @param[in] format         The tensor data format.
     * @param[in] address_mode_x The address mode of the x dimension.
     * @param[in] address_mode_y The address mode of the y dimension.
     * @param[in] address_mode_z The address mode of the z dimension.
     */
    TensorSampler(
        TensorStorageType         storage,
        TensorSamplerFormat       format,
        TensorSamplerAddressModeX address_mode_x,
        TensorSamplerAddressModeY address_mode_y,
        TensorSamplerAddressModeZ address_mode_z);

    /** Get the storage for the tensor */
    TensorStorageType storage() const;

    /** Set the storage for the tensor */
    TensorSampler &storage(TensorStorageType storage);

    /** Get the format of the tensor. */
    TensorSamplerFormat format() const;

    /** Set the format of the tensor. */
    TensorSampler &format(TensorSamplerFormat format);

    /** Get the address mode of the x dimension. */
    TensorSamplerAddressModeX address_mode_x() const;

    /** Set the address mode of the x dimension. */
    TensorSampler &address_mode_x(TensorSamplerAddressModeX address_mode_x);

    /** Get the address mode of the y dimension. */
    TensorSamplerAddressModeY address_mode_y() const;

    /** Set the address mode of the y dimension. */
    TensorSampler &address_mode_y(TensorSamplerAddressModeY address_mode_y);

    /** Get the address mode of the z dimension. */
    TensorSamplerAddressModeZ address_mode_z() const;

    /** Set the address mode of the z dimension. */
    TensorSampler &address_mode_z(TensorSamplerAddressModeZ address_mode_z);

private:
    TensorStorageType                _storage { TensorStorageType::BufferUint8Ptr };
    TensorSamplerFormat              _format  { TensorSamplerFormat::Unknown };
    TensorSamplerAddressModeX _address_mode_x { TensorSamplerAddressModeX::Unknown };
    TensorSamplerAddressModeY _address_mode_y { TensorSamplerAddressModeY::Unknown };
    TensorSamplerAddressModeZ _address_mode_z { TensorSamplerAddressModeZ::Unknown };
};

} // namespace ckw

#endif // CKW_PROTOTYPE_INCLUDE_CKW_TENSORSAMPLER_H
