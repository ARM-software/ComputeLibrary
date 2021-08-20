/*
 * Copyright (c) 2017-2021 Arm Limited.
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
#include "arm_compute/runtime/NEON/functions/NEFlattenLayer.h"

#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/cpu/operators/CpuFlatten.h"

namespace arm_compute
{
struct NEFlattenLayer::Impl
{
    const ITensor                   *src{ nullptr };
    ITensor                         *dst{ nullptr };
    std::unique_ptr<cpu::CpuFlatten> op{ nullptr };
};

NEFlattenLayer::NEFlattenLayer()
    : _impl(std::make_unique<Impl>())
{
}
NEFlattenLayer::NEFlattenLayer(NEFlattenLayer &&) = default;
NEFlattenLayer &NEFlattenLayer::operator=(NEFlattenLayer &&) = default;
NEFlattenLayer::~NEFlattenLayer()                            = default;

void NEFlattenLayer::configure(const ITensor *input, ITensor *output)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    _impl->src = input;
    _impl->dst = output;
    auto_init_if_empty(*output->info(), input->info()->clone()->set_tensor_shape(misc::shape_calculator::compute_flatten_shape(input->info())));

    _impl->op = std::make_unique<cpu::CpuFlatten>();
    _impl->op->configure(_impl->src->info(), _impl->dst->info());
}

Status NEFlattenLayer::validate(const ITensorInfo *input, const ITensorInfo *output)
{
    // Checks performed when output is configured
    if(output->total_size() != 0)
    {
        const TensorInfo tensor_info_output = input->clone()->set_tensor_shape(misc::shape_calculator::compute_flatten_shape(input));
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(output, &tensor_info_output);
    }
    return cpu::CpuFlatten::validate(input, output);
}
void NEFlattenLayer::run()
{
    ITensorPack pack;
    pack.add_tensor(TensorType::ACL_SRC, _impl->src);
    pack.add_tensor(TensorType::ACL_DST, _impl->dst);
    _impl->op->run(pack);
}
} // namespace arm_compute
