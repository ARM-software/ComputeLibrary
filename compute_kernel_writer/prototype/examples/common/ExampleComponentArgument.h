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

#ifndef CKW_PROTOTYPE_EXAMPLES_COMMON_EXAMPLECOMPONENTARGUMENT_H
#define CKW_PROTOTYPE_EXAMPLES_COMMON_EXAMPLECOMPONENTARGUMENT_H

#include "ckw/TensorTileSampler.h"

namespace ckw
{
class TensorOperand;

class TileOperand;
} // namespace ckw

/** The argument of a dynamic fusion component which can be either user tensor or virtual tensor. */
class ExampleComponentArgument
{
public:
    /** Initialize a new instance of @ref ExampleComponentArgument class for empty virtual tensor. */
    ExampleComponentArgument();

    /** Initialize a new instance of @ref ExampleComponentArgument class for user tensor.
     *
     * @param[in] tensor The user tensor.
     */
    explicit ExampleComponentArgument(ckw::TensorOperand &tensor);

    /** Set virtual tensor information (tile, sampler) for the argument.
     *
     * If the component is a user tensor, it can be treated as virtual tensor as well
     * and won't be loaded again using @ref ExampleKernelWriter::op_load_once method.
     *
     * @param[in] tile    The tile that has been loaded.
     * @param[in] sampler The tensor sampling information that has been used to load the tile.
     */
    ExampleComponentArgument &init_virtual_tensor(ckw::TileOperand &tile, const ckw::TensorTileSampler &sampler);

    /** Get whether the argument is a user tensor. */
    bool has_tensor() const;

    /** Get the tensor operand.
     *
     * If the tensor is not available, throw an error.
     */
    ckw::TensorOperand &tensor();

    /** Get the tensor operand.
     *
     * If the tensor is not available, throw an error.
     */
    const ckw::TensorOperand &tensor() const;

    /** Get whether the argument contains a tile.
     *
     * The argument can be either a user tensor that has been loaded,
     * or a virtual tensor (i.e. a tile with tensor sampling information).
     */
    bool has_tile() const;

    /** Get the tile operand.
     *
     * If the tile is not available, throw an error.
     */
    ckw::TileOperand &tile();

    /** Get the tile operand.
     *
     * If the tile is not available, throw an error.
     */
    const ckw::TileOperand &tile() const;

    /** Get the tensor sampling information for the tile.
     *
     * If the tile is not available, throw an error.
     */
    ckw::TensorTileSampler &tile_sampler();

    /** Get the tensor sampling information for the tile.
     *
     * If the tile is not available, throw an error.
     */
    const ckw::TensorTileSampler &tile_sampler() const;

private:
    ckw::TensorOperand    *_tensor{nullptr};
    ckw::TileOperand      *_tile{nullptr};
    ckw::TensorTileSampler _tile_sampler{};
};

#endif // CKW_PROTOTYPE_EXAMPLES_COMMON_EXAMPLECOMPONENTARGUMENT_H
