/*
 * Copyright (c) 2019-2021, 2023 Arm Limited.
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

#include "arm_compute/runtime/NEON/functions/NEBatchToSpaceLayer.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"
#include "src/common/utils/Log.h"
#include "src/core/NEON/kernels/NEBatchToSpaceLayerKernel.h"

namespace arm_compute
{
void NEBatchToSpaceLayer::configure(const ITensor *input, const ITensor *block_shape, ITensor *output, const CropInfo &crop_info)
{
    ARM_COMPUTE_UNUSED(crop_info);
    ARM_COMPUTE_LOG_PARAMS(input, block_shape, output);
    auto k = std::make_unique<NEBatchToSpaceLayerKernel>();
    k->configure(input, block_shape, output);
    _kernel = std::move(k);
}

void NEBatchToSpaceLayer::configure(const ITensor *input, int32_t block_shape_x, int32_t block_shape_y, ITensor *output, const CropInfo &crop_info)
{
    ARM_COMPUTE_UNUSED(crop_info);
    auto k = std::make_unique<NEBatchToSpaceLayerKernel>();
    k->configure(input, block_shape_x, block_shape_y, output);
    _kernel = std::move(k);
}

Status NEBatchToSpaceLayer::validate(const ITensorInfo *input, const ITensorInfo *block_shape, const ITensorInfo *output, const CropInfo &crop_info)
{
    ARM_COMPUTE_UNUSED(crop_info);
    return NEBatchToSpaceLayerKernel::validate(input, block_shape, output);
}

Status NEBatchToSpaceLayer::validate(const ITensorInfo *input, int32_t block_shape_x, int32_t block_shape_y, const ITensorInfo *output, const CropInfo &crop_info)
{
    ARM_COMPUTE_UNUSED(crop_info);
    return NEBatchToSpaceLayerKernel::validate(input, block_shape_x, block_shape_y, output);
}
} // namespace arm_compute
