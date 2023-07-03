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

#include "src/dynamic_fusion/sketch/gpu/ckw_driver/GpuCkwComponentArgument.h"
#include "src/dynamic_fusion/sketch/gpu/ckw_driver/GpuCkwScopedKernelWriter.h"
#include "ckw/TensorTileSampler.h"

#include <functional>

using namespace ckw;
namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
using SamplerCreator = std::function<TensorTileSampler(GpuCkwScopedKernelWriter &, int32_t /* m0 */, int32_t /* n0 */)>;

/** Load lhs and rhs tiles of dimension [m0, n0] only when not loaded and prepare the sampler
 */
inline void load_lhs_rhs_tiles_and_prepare_sampler(GpuCkwScopedKernelWriter &writer, GpuCkwComponentArgument *lhs, GpuCkwComponentArgument *rhs, int32_t m0, int32_t n0, SamplerCreator create_sampler)
{
    if(!lhs->has_tile() && !rhs->has_tile())
    {
        const auto sampler = create_sampler(writer, m0, n0);

        writer->op_load_once(lhs, sampler);
        writer->op_load_once(rhs, sampler);
    }
    else if(lhs->has_tile())
    {
        const auto &sampler = lhs->tile_sampler();
        writer->op_load_once(rhs, sampler);
    }
    else
    {
        const auto &sampler = rhs->tile_sampler();
        writer->op_load_once(lhs, sampler);
    }
}

} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
#endif /* ACL_SRC_DYNAMIC_FUSION_SKETCH_GPU_CKW_DRIVER_COMPONENTS_UTILS_WRITERHELPER */
