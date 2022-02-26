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
#include "src/core/NEON/kernels/NEBoundingBoxTransformKernel.h"

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Window.h"
#include "src/core/CPP/Validate.h"
#include "src/core/common/Registrars.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"
#include "src/cpu/kernels/boundingboxtransform/list.h"

#include <arm_neon.h>

namespace arm_compute
{
namespace
{
struct BoundingBoxTransformSelectorData
{
    DataType dt;
};

using BoundingBoxTransformSelctorPtr = std::add_pointer<bool(const BoundingBoxTransformSelectorData &data)>::type;
using BoundingBoxTransformUKernelPtr = std::add_pointer<void(const ITensor *boxes, ITensor *pred_boxes, const ITensor *deltas, BoundingBoxTransformInfo bbinfo, const Window &window)>::type;

struct BoundingBoxTransformKernel
{
    const char                          *name;
    const BoundingBoxTransformSelctorPtr is_selected;
    BoundingBoxTransformUKernelPtr       ukernel;
};

static const BoundingBoxTransformKernel available_kernels[] =
{
    {
        "fp32_neon_boundingboxtransform",
        [](const BoundingBoxTransformSelectorData & data) { return data.dt == DataType::F32; },
        REGISTER_FP32_NEON(arm_compute::cpu::neon_fp32_boundingboxtransform)
    },
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    {
        "fp16_neon_boundingboxtransform",
        [](const BoundingBoxTransformSelectorData & data) { return data.dt == DataType::F16; },
        REGISTER_FP16_NEON(arm_compute::cpu::neon_fp16_boundingboxtransform)
    },
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
#if defined(ARM_COMPUTE_ENABLE_NEON)
    {
        "qu16_neon_boundingboxtransform",
        [](const BoundingBoxTransformSelectorData & data) { return data.dt == DataType::QASYMM16; },
        REGISTER_QSYMM16_NEON(arm_compute::cpu::neon_qu16_boundingboxtransform)
    },
#endif //defined(ARM_COMPUTE_ENABLE_NEON)
};

/** Micro-kernel selector
 *
 * @param[in] data Selection data passed to help pick the appropriate micro-kernel
 *
 * @return A matching micro-kernel else nullptr
 */
const BoundingBoxTransformKernel *get_implementation(const BoundingBoxTransformSelectorData &data)
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

Status validate_arguments(const ITensorInfo *boxes, const ITensorInfo *pred_boxes, const ITensorInfo *deltas, const BoundingBoxTransformInfo &info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(boxes, pred_boxes, deltas);
    ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(boxes);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_NOT_IN(boxes, DataType::QASYMM16, DataType::F32, DataType::F16);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_NOT_IN(deltas, DataType::QASYMM8, DataType::F32, DataType::F16);
    ARM_COMPUTE_RETURN_ERROR_ON(deltas->tensor_shape()[1] != boxes->tensor_shape()[1]);
    ARM_COMPUTE_RETURN_ERROR_ON(deltas->tensor_shape()[0] % 4 != 0);
    ARM_COMPUTE_RETURN_ERROR_ON(boxes->tensor_shape()[0] != 4);
    ARM_COMPUTE_RETURN_ERROR_ON(deltas->num_dimensions() > 2);
    ARM_COMPUTE_RETURN_ERROR_ON(boxes->num_dimensions() > 2);
    ARM_COMPUTE_RETURN_ERROR_ON(info.scale() <= 0);

    if(boxes->data_type() == DataType::QASYMM16)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(deltas, 1, DataType::QASYMM8);
        const UniformQuantizationInfo deltas_qinfo = deltas->quantization_info().uniform();
        ARM_COMPUTE_RETURN_ERROR_ON(deltas_qinfo.scale != 0.125f);
        ARM_COMPUTE_RETURN_ERROR_ON(deltas_qinfo.offset != 0);
    }
    else
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(boxes, deltas);
    }

    if(pred_boxes->total_size() > 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(pred_boxes->tensor_shape(), deltas->tensor_shape());
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(pred_boxes, deltas);
        ARM_COMPUTE_RETURN_ERROR_ON(pred_boxes->num_dimensions() > 2);
        if(pred_boxes->data_type() == DataType::QASYMM16)
        {
            const UniformQuantizationInfo pred_qinfo = pred_boxes->quantization_info().uniform();
            ARM_COMPUTE_RETURN_ERROR_ON(pred_qinfo.scale != 0.125f);
            ARM_COMPUTE_RETURN_ERROR_ON(pred_qinfo.offset != 0);
        }
    }

    return Status{};
}
} // namespace

NEBoundingBoxTransformKernel::NEBoundingBoxTransformKernel()
    : _boxes(nullptr), _pred_boxes(nullptr), _deltas(nullptr), _bbinfo(0, 0, 0)
{
}

void NEBoundingBoxTransformKernel::configure(const ITensor *boxes, ITensor *pred_boxes, const ITensor *deltas, const BoundingBoxTransformInfo &info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(boxes, pred_boxes, deltas);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(boxes->info(), pred_boxes->info(), deltas->info(), info));

    // Configure kernel window
    auto_init_if_empty(*pred_boxes->info(), deltas->info()->clone()->set_data_type(boxes->info()->data_type()).set_quantization_info(boxes->info()->quantization_info()));

    // Set instance variables
    _boxes      = boxes;
    _pred_boxes = pred_boxes;
    _deltas     = deltas;
    _bbinfo     = info;

    const unsigned int num_boxes = boxes->info()->dimension(1);
    Window             win       = calculate_max_window(*pred_boxes->info(), Steps());
    win.set(Window::DimX, Window::Dimension(0, 1u));
    win.set(Window::DimY, Window::Dimension(0, num_boxes));

    INEKernel::configure(win);
}

Status NEBoundingBoxTransformKernel::validate(const ITensorInfo *boxes, const ITensorInfo *pred_boxes, const ITensorInfo *deltas, const BoundingBoxTransformInfo &info)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(boxes, pred_boxes, deltas, info));
    return Status{};
}

void NEBoundingBoxTransformKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);

    const auto *uk = get_implementation(BoundingBoxTransformSelectorData{ _boxes->info()->data_type() });
    ARM_COMPUTE_ERROR_ON(uk == nullptr || uk->ukernel == nullptr);

    uk->ukernel(_boxes, _pred_boxes, _deltas, _bbinfo, window);
}
} // namespace arm_compute
