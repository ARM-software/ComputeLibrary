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

#ifndef ACL_SRC_DYNAMIC_FUSION_SKETCH_GPU_CKW_DRIVER_GPUCKWCOMPONENTARGUMENT_H
#define ACL_SRC_DYNAMIC_FUSION_SKETCH_GPU_CKW_DRIVER_GPUCKWCOMPONENTARGUMENT_H

#include "compute_kernel_writer/include/ckw/TensorOperand.h"
#include "compute_kernel_writer/include/ckw/TensorSampler.h"
#include "compute_kernel_writer/include/ckw/TileOperand.h"

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{

/** The argument of a dynamic fusion component which can be either user tensor or virtual tensor. */
class GpuCkwComponentArgument
{
public:
    /** Default constructor */
    GpuCkwComponentArgument() = default;

    /** Initialize a new instance of @ref GpuCkwComponentArgument class for user tensor.
     *
     * @param[in] tensor The user tensor.
     */
    explicit GpuCkwComponentArgument(ckw::TensorOperand tensor);

    /** Bind the tile and sampler to the tensor argument.
     *
     * This method can be used to share a tile and sampler associated to a tensor
     * among different kernel components. For example, when we create the destination
     * tile and destination sampler for the first time (root component), this method can be
     * used to bind these two information to the destination tensor so that the following
     * simple components know the tile size and how to access the elements from memory.
     *
     * @param[in] tile    The tile that has been loaded.
     * @param[in] sampler The tensor sampling information that has been used to load the tile.
     */
    GpuCkwComponentArgument &init_virtual_tensor(ckw::TileOperand &tile, const ckw::TensorSampler &sampler);

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
    ckw::TensorSampler &tensor_sampler();

    /** Get the tensor sampling information for the tile.
     *
     * If the tile is not available, throw an error.
     */
    const ckw::TensorSampler &tensor_sampler() const;

private:
    ckw::TensorOperand _tensor{};
    ckw::TileOperand   _tile{};
    ckw::TensorSampler _sampler{};
};

} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute

#endif // ACL_SRC_DYNAMIC_FUSION_SKETCH_GPU_CKW_DRIVER_GPUCKWCOMPONENTARGUMENT_H
