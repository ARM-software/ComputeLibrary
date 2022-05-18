/*
 * Copyright (c) 2020-2022 Arm Limited.
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
#include "arm_compute/runtime/NEON/functions/NEMaxUnpoolingLayer.h"

#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "arm_compute/runtime/NEON/functions/NEFill.h"
#include "src/common/utils/Log.h"
#include "src/cpu/kernels/CpuMaxUnpoolingLayerKernel.h"
#include "src/cpu/operators/CpuMaxUnpooling.h"

namespace arm_compute
{
struct NEMaxUnpoolingLayer::Impl
{
    const ITensor                        *src{ nullptr };
    const ITensor                        *indices{ nullptr };
    ITensor                              *dst{ nullptr };
    std::unique_ptr<cpu::CpuMaxUnpooling> op{ nullptr };
};

NEMaxUnpoolingLayer::~NEMaxUnpoolingLayer() = default;

NEMaxUnpoolingLayer::NEMaxUnpoolingLayer()
    : _fill_func(), _impl()
{
}

void NEMaxUnpoolingLayer::configure(ITensor *input, ITensor *indices, ITensor *output, const PoolingLayerInfo &pool_info)
{
    ARM_COMPUTE_LOG_PARAMS(input, indices, output, pool_info);

    const PixelValue zero_value(0.f);
    _fill_func     = std::make_unique<NEFill>();
    _impl          = std::make_unique<Impl>();
    _impl->src     = input;
    _impl->indices = indices;
    _impl->dst     = output;

    _impl->op = std::make_unique<cpu::CpuMaxUnpooling>();
    _fill_func->configure(output, zero_value);
    _impl->op->configure(input->info(), indices->info(), output->info(), pool_info);
}

Status NEMaxUnpoolingLayer::validate(const ITensorInfo *input, const ITensorInfo *indices, const ITensorInfo *output, const PoolingLayerInfo &pool_info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output, indices);
    ARM_COMPUTE_RETURN_ON_ERROR(cpu::CpuMaxUnpooling::validate(input, indices, output, pool_info));
    return Status{};
}

void NEMaxUnpoolingLayer::run()
{
    ITensorPack pack;
    pack.add_tensor(TensorType::ACL_SRC_0, _impl->src);
    pack.add_tensor(TensorType::ACL_SRC_1, _impl->indices);
    pack.add_tensor(TensorType::ACL_DST, _impl->dst);

    _fill_func->run();
    _impl->op->run(pack);
}
} /* namespace arm_compute */
