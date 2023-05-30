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

#ifndef CKW_INCLUDE_CKW_TENSORTILESAMPLER_H
#define CKW_INCLUDE_CKW_TENSORTILESAMPLER_H

#include "ckw/Types.h"
#include <functional>

namespace ckw
{

class TileOperand;

/** Tensor sampler
 *
 * It contains information about how the result tile should be stored to tensor memory.
 * It can also be used to dictate how the subsequent operators fetch the input tensor.
 */
class TensorTileSampler
{
public:
    /** Initialize a new instance of @ref TensorSampler class. */
    TensorTileSampler();

    /** Initialize a new instance of @ref TensorSampler class.
     *
     * @param[in] x              The coordinate in the x dimension.
     * @param[in] y              The coordinate in the y dimension.
     * @param[in] z              The coordinate in the z dimension.
     * @param[in] b              The coordinate in the batch dimension.
     * @param[in] format         The tensor data format.
     * @param[in] address_mode_x The address mode of the x dimension.
     * @param[in] address_mode_y The address mode of the y dimension.
     * @param[in] address_mode_z The address mode of the z dimension.
     */
    TensorTileSampler(
        TileOperand &x, TileOperand &y, TileOperand &z, TileOperand &b,
        TensorSamplerFormat       format,
        TensorSamplerAddressModeX address_mode_x,
        TensorSamplerAddressModeY address_mode_y,
        TensorSamplerAddressModeZ address_mode_z);

    /** Initialize a new instance of @ref TensorSampler class.
     *
     * @param[in] x              The coordinate in the x dimension.
     * @param[in] y              The coordinate in the y dimension.
     * @param[in] z              The coordinate in the z dimension.
     * @param[in] b              The coordinate in the batch dimension.
     * @param[in] height         The height of the tile.
     * @param[in] width          The width of the tile.
     * @param[in] format         The tensor data format.
     * @param[in] address_mode_x The address mode of the x dimension.
     * @param[in] address_mode_y The address mode of the y dimension.
     * @param[in] address_mode_z The address mode of the z dimension.
     */
    TensorTileSampler(
        TileOperand &x, TileOperand &y, TileOperand &z, TileOperand &b,
        int32_t height, int32_t width,
        TensorSamplerFormat       format,
        TensorSamplerAddressModeX address_mode_x,
        TensorSamplerAddressModeY address_mode_y,
        TensorSamplerAddressModeZ address_mode_z);

    /** Get the coordinate in the x dimension. */
    const TileOperand &x() const;

    /** Set the coordinate in the x dimension. */
    TensorTileSampler &x(TileOperand &x);

    /** Get the coordinate in the y dimension. */
    const TileOperand &y() const;

    /** Set the coordinate in the y dimension. */
    TensorTileSampler &y(TileOperand &y);

    /** Get the coordinate in the z dimension. */
    const TileOperand &z() const;

    /** Set the coordinate in the z dimension. */
    TensorTileSampler &z(TileOperand &z);

    /** Get the coordinate in the batch dimension. */
    const TileOperand &b() const;

    /** Set the coordinate in the batch dimension. */
    TensorTileSampler &b(TileOperand &b);

    /** Get the width of the tile. */
    int32_t width() const;

    /** Set the width of the tile. */
    TensorTileSampler &width(int32_t width);

    /** Get the height of the tile. */
    int32_t height() const;

    /** Set the height of the tile. */
    TensorTileSampler &height(int32_t height);

    /** Get the format of the tensor. */
    TensorSamplerFormat format() const;

    /** Set the format of the tensor. */
    TensorTileSampler &format(TensorSamplerFormat format);

    /** Get the address mode of the x dimension. */
    TensorSamplerAddressModeX address_mode_x() const;

    /** Set the address mode of the x-dimension. */
    TensorTileSampler &address_mode_x(TensorSamplerAddressModeX address_mode_x);

    /** Get the address mode of the y dimension. */
    TensorSamplerAddressModeY address_mode_y() const;

    /** Set the address mode of the y dimension. */
    TensorTileSampler &address_mode_y(TensorSamplerAddressModeY address_mode_y);

    /** Get the address mode of the z dimension. */
    TensorSamplerAddressModeZ address_mode_z() const;

    /** Set the address mode of the z dimension. */
    TensorTileSampler &address_mode_z(TensorSamplerAddressModeZ address_mode_z);

private:
    TileOperand *_x{ nullptr };
    TileOperand *_y{ nullptr };
    TileOperand *_z{ nullptr };
    TileOperand *_b{ nullptr };

    int32_t _height{ 0 };
    int32_t _width{ 0 };

    TensorSamplerFormat       _format{ TensorSamplerFormat::Unknown };
    TensorSamplerAddressModeX _address_mode_x{ TensorSamplerAddressModeX::Unknown };
    TensorSamplerAddressModeY _address_mode_y{ TensorSamplerAddressModeY::Unknown };
    TensorSamplerAddressModeZ _address_mode_z{ TensorSamplerAddressModeZ::Unknown };
};

} // namespace ckw

#endif // CKW_INCLUDE_CKW_TENSORTILESAMPLER_H
