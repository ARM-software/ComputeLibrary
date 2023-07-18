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
#ifndef ACL_SRC_DYNAMIC_FUSION_SKETCH_GPU_CKW_DRIVER_COMPONENTS_UTILS_WRITERHELPER
#define ACL_SRC_DYNAMIC_FUSION_SKETCH_GPU_CKW_DRIVER_COMPONENTS_UTILS_WRITERHELPER

#include "arm_compute/core/utils/misc/Utility.h"
#include "ckw/TensorTileSampler.h"
#include "src/dynamic_fusion/sketch/gpu/ckw_driver/GpuCkwComponentArgument.h"
#include "src/dynamic_fusion/sketch/gpu/ckw_driver/GpuCkwScopedKernelWriter.h"

#include <algorithm>
#include <functional>

using namespace ckw;
namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
using SamplerCreator = std::function<TensorTileSampler(GpuCkwScopedKernelWriter &, int32_t /* m0 */, int32_t /* n0 */)>;

/** Load src and dst tiles of dimension [m0, n0] only when not loaded and prepare the sampler
 */
inline void load_src_dst_tiles_and_prepare_sampler(GpuCkwScopedKernelWriter &writer, GpuCkwComponentArgument *src, GpuCkwComponentArgument *dst, int32_t m0, int32_t n0, SamplerCreator create_sampler)
{
    if(!src->has_tile())
    {
        const auto sampler = create_sampler(writer, m0, n0);
        writer->op_load_once(src, sampler);
    }
    else
    {
        const auto &sampler = src->tile_sampler();
        writer->op_load_once(src, sampler);
    }

    auto       &src_tile = src->tile();
    const auto &sampler  = src->tile_sampler();

    // Prepare the output tile.
    if(!dst->has_tile())
    {
        auto &tile = writer->declare_tile("dst_tile", src_tile.tile_info());
        dst->init_virtual_tensor(tile, sampler);
    }
}

/** Get boundary aware coordinate along one axis. Load and store of size step_v at the coordinate will not be out of bound
 *
 * @param[in,out] writer          Writer
 * @param[out]    coord           Resultant coordinate
 * @param[in]     gid             Global work item id
 * @param[in]     step_v          Step size / vector size
 * @param[in]     leftover_step_v Leftover step size at the boundary
 * @param[in]     prefix          Prefix to all the tiles declared within this function
 * @param[in]     const_0         Constant tile of value 0
 */
inline void get_coord(GpuCkwScopedKernelWriter writer, TileOperand &coord, TileOperand &gid, int32_t step_v, int32_t leftover_step_v, const std::string &prefix, TileOperand &const_0)
{
    auto &step          = writer->declare_tile(prefix + "step", step_v);
    auto &leftover_step = writer->declare_tile(prefix + "leftover_step", leftover_step_v);

    // step - leftover_step
    auto &step_minus_leftover = writer->declare_tile(prefix + "step_minus_leftover", ckw::DataType::Int32);
    writer->op_binary_expression(step_minus_leftover, step, ckw::BinaryOp::Sub, leftover_step);

    // (step - leftover_step) % step
    auto &coord_correction = writer->declare_tile(prefix + "coord_correction", ckw::DataType::Int32);
    writer->op_binary_expression(coord_correction, step_minus_leftover, ckw::BinaryOp::Mod, step);

    // (gid * step)
    auto &raw_coord = writer->declare_tile(prefix + "raw_coord", ckw::DataType::Int32);
    writer->op_binary_expression(raw_coord, gid, ckw::BinaryOp::Mul, step);

    // (gid * step) - (step - leftover_step) % step
    auto &corrected_coord = writer->declare_tile(prefix + "corrected_coord", ckw::DataType::Int32);
    writer->op_binary_expression(corrected_coord, raw_coord, ckw::BinaryOp::Sub, coord_correction);

    // max((gid * step) - (step - leftover_step) % step, 0)
    writer->op_binary_elementwise_function(coord, ckw::BinaryFunction::Max, corrected_coord, const_0);
}

/** Declare coordinate tiles "{prefix}_dim0_coord" and "{prefix}_dim1_coord", and create a boundary-aware sampler from tile of size [n0, m0], against the overall dimensions [dim0, dim1]
 * The load and store of tile [n0, m0] will never be out of bound of [dim0, dim1]
 */

/** Declare coordinate tiles "{prefix}_dim0_coord" and "{prefix}_dim1_coord", and create a boundary-aware sampler from tile of size [n0, m0], against the overall dimensions [dim0, dim1]
 * The load and store of tile [n0, m0] will never be out of bound of [dim0, dim1]
 *
 * @param[in,out] writer  Writer
 * @param[in]     gid_0   Global work item id 0
 * @param[in]     gid_1   Global work item id 1
 * @param[in]     dim0_v  Dimension 0
 * @param[in]     dim1_v  Dimension 1
 * @param[in]     n0_v    Tile size dimension 0
 * @param[in]     m0_v    Tile size dimension 1
 * @param[in]     prefix  Prefix to all the tiles declared within this function
 * @param[in]     const_0 Constant tile of value 0
 *
 * @return TensorTileSampler
 */
inline TensorTileSampler create_boundary_aware_2d_sampler(GpuCkwScopedKernelWriter writer, TileOperand &gid_0, TileOperand &gid_1, int32_t dim0_v, int32_t dim1_v, int32_t n0_v, int32_t m0_v,
                                                          const std::string prefix, TileOperand &const_0)
{
    // Clamp tile size [n0, m0] against dimension [dim0, dim1]
    // This is needed to:
    // * Guard against tile sizes are bigger than the tensor dimensions
    // * Handle broadcasting tiles (e.g. src tensor is of size 1 in one of the dimensions)
    n0_v                       = utility::clamp(n0_v, 1, dim0_v);
    m0_v                       = utility::clamp(m0_v, 1, dim1_v);
    const int32_t partial_n0_v = dim0_v % n0_v;
    const int32_t partial_m0_v = dim1_v % m0_v;

    // Declare #prefix_dim0_coord and #prefix_dim1_coord
    auto &dim0_coord = writer->declare_tile(prefix + "dim0_coord", ckw::DataType::Int32);
    get_coord(writer, dim0_coord, gid_0, n0_v, partial_n0_v, prefix + "dim0_", const_0);
    auto &dim1_coord = writer->declare_tile(prefix + "dim1_coord", ckw::DataType::Int32);
    get_coord(writer, dim1_coord, gid_1, m0_v, partial_m0_v, prefix + "dim1_", const_0);

    // Set sampler
    // Only set fields related to boundary aware loading/storing. Other info (e.g. format) is not responsibility of this function
    TensorTileSampler sampler;

    sampler.x(dim0_coord);
    sampler.y(dim1_coord);

    sampler.width(n0_v);
    sampler.height(m0_v);

    sampler.address_mode_x(TensorSamplerAddressModeX::None);
    sampler.address_mode_y(TensorSamplerAddressModeY::None);
    sampler.address_mode_z(TensorSamplerAddressModeZ::None);

    return sampler;
}
} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
#endif /* ACL_SRC_DYNAMIC_FUSION_SKETCH_GPU_CKW_DRIVER_COMPONENTS_UTILS_WRITERHELPER */
