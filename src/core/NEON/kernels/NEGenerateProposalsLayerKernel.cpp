/*
 * Copyright (c) 2019-2022 Arm Limited.
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
#include "src/core/NEON/kernels/NEGenerateProposalsLayerKernel.h"

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Window.h"
#include "src/core/CPP/Validate.h"
#include "src/core/common/Registrars.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"
#include "src/cpu/kernels/genproposals/list.h"
#include <arm_neon.h>

namespace arm_compute
{
namespace
{
struct ComputeAllAnchorsData
{
    DataType dt;
};

using ComputeAllAnchorsSelectorPtr = std::add_pointer<bool(const ComputeAllAnchorsData &data)>::type;
using ComputeAllAnchorsUKernelPtr  = std::add_pointer<void(const ITensor *anchors, ITensor *all_anchors, ComputeAnchorsInfo anchors_info, const Window &window)>::type;

struct ComputeAllAnchorsKernel
{
    const char                        *name;
    const ComputeAllAnchorsSelectorPtr is_selected;
    ComputeAllAnchorsUKernelPtr        ukernel;
};

static const ComputeAllAnchorsKernel available_kernels[] =
{
#if defined(ARM_COMPUTE_ENABLE_NEON)
    {
        "neon_qu16_computeallanchors",
        [](const ComputeAllAnchorsData & data) { return data.dt == DataType::QSYMM16; },
        REGISTER_QSYMM16_NEON(arm_compute::cpu::neon_qu16_computeallanchors)
    },
#endif //defined(ARM_COMPUTE_ENABLE_NEON)
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    {
        "neon_fp16_computeallanchors",
        [](const ComputeAllAnchorsData & data) { return data.dt == DataType::F16; },
        REGISTER_FP16_NEON(arm_compute::cpu::neon_fp16_computeallanchors)
    },
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    {
        "neon_fp32_computeallanchors",
        [](const ComputeAllAnchorsData & data) { return data.dt == DataType::F32; },
        REGISTER_FP32_NEON(arm_compute::cpu::neon_fp32_computeallanchors)
    },
};

/** Micro-kernel selector
 *
 * @param[in] data Selection data passed to help pick the appropriate micro-kernel
 *
 * @return A matching micro-kernel else nullptr
 */
const ComputeAllAnchorsKernel *get_implementation(const ComputeAllAnchorsData &data)
{
    for(const auto &uk : available_kernels)
    {
        if(uk.is_selected(data))
        {
            return &uk;
        }
    }
    return nullptr;
}

Status validate_arguments(const ITensorInfo *anchors, const ITensorInfo *all_anchors, const ComputeAnchorsInfo &info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(anchors, all_anchors);
    ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(anchors);
    ARM_COMPUTE_RETURN_ERROR_ON(anchors->dimension(0) != info.values_per_roi());
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_NOT_IN(anchors, DataType::QSYMM16, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON(anchors->num_dimensions() > 2);
    if(all_anchors->total_size() > 0)
    {
        const size_t feature_height = info.feat_height();
        const size_t feature_width  = info.feat_width();
        const size_t num_anchors    = anchors->dimension(1);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(all_anchors, anchors);
        ARM_COMPUTE_RETURN_ERROR_ON(all_anchors->num_dimensions() > 2);
        ARM_COMPUTE_RETURN_ERROR_ON(all_anchors->dimension(0) != info.values_per_roi());
        ARM_COMPUTE_RETURN_ERROR_ON(all_anchors->dimension(1) != feature_height * feature_width * num_anchors);

        if(is_data_type_quantized(anchors->data_type()))
        {
            ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_QUANTIZATION_INFO(anchors, all_anchors);
        }
    }
    return Status{};
}

} // namespace

NEComputeAllAnchorsKernel::NEComputeAllAnchorsKernel()
    : _anchors(nullptr), _all_anchors(nullptr), _anchors_info(0.f, 0.f, 0.f)
{
}

void NEComputeAllAnchorsKernel::configure(const ITensor *anchors, ITensor *all_anchors, const ComputeAnchorsInfo &info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(anchors, all_anchors);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(anchors->info(), all_anchors->info(), info));

    // Metadata
    const size_t   num_anchors = anchors->info()->dimension(1);
    const DataType data_type   = anchors->info()->data_type();
    const float    width       = info.feat_width();
    const float    height      = info.feat_height();

    // Initialize the output if empty
    const TensorShape output_shape(info.values_per_roi(), width * height * num_anchors);
    auto_init_if_empty(*all_anchors->info(), TensorInfo(output_shape, 1, data_type, anchors->info()->quantization_info()));

    // Set instance variables
    _anchors      = anchors;
    _all_anchors  = all_anchors;
    _anchors_info = info;

    Window win = calculate_max_window(*all_anchors->info(), Steps(info.values_per_roi()));

    INEKernel::configure(win);
}

Status NEComputeAllAnchorsKernel::validate(const ITensorInfo *anchors, const ITensorInfo *all_anchors, const ComputeAnchorsInfo &info)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(anchors, all_anchors, info));
    return Status{};
}

void NEComputeAllAnchorsKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);

    const auto *uk = get_implementation(ComputeAllAnchorsData{ _anchors->info()->data_type() });
    ARM_COMPUTE_ERROR_ON(uk == nullptr || uk->ukernel == nullptr);

    uk->ukernel(_anchors, _all_anchors, _anchors_info, window);
}
} // namespace arm_compute
