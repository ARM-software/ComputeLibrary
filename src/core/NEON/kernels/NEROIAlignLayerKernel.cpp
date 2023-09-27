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
#include "src/core/NEON/kernels/NEROIAlignLayerKernel.h"

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/core/utils/misc/Utility.h"
#include "arm_compute/core/Window.h"

#include "src/core/common/Registrars.h"
#include "src/core/CPP/Validate.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"
#include "src/cpu/kernels/roialign/list.h"

#include <arm_neon.h>

using namespace arm_compute::misc::shape_calculator;

namespace arm_compute
{
namespace
{
struct ROIAlignSelectorData
{
    DataType dt;
};

using ROIAlignSelctorPtr = std::add_pointer<bool(const ROIAlignSelectorData &data)>::type;
using ROIAlignUKernelPtr = std::add_pointer<void(const ITensor      *input,
                                                 ITensor            *output,
                                                 const ITensor      *rois,
                                                 ROIPoolingLayerInfo pool_info,
                                                 const Window       &window,
                                                 const ThreadInfo   &info)>::type;

struct ROIAlignKernel
{
    const char              *name;
    const ROIAlignSelctorPtr is_selected;
    ROIAlignUKernelPtr       ukernel;
};

static const ROIAlignKernel available_kernels[] = {
    {"fp32_neon_roialign", [](const ROIAlignSelectorData &data) { return data.dt == DataType::F32; },
     REGISTER_FP32_NEON(arm_compute::cpu::neon_fp32_roialign)},
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    {"fp16_neon_roialign", [](const ROIAlignSelectorData &data) { return data.dt == DataType::F16; },
     REGISTER_FP16_NEON(arm_compute::cpu::neon_fp16_roialign)},
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
#if defined(ARM_COMPUTE_ENABLE_NEON)
    {"qu8_neon_roialign", [](const ROIAlignSelectorData &data) { return data.dt == DataType::QASYMM8; },
     REGISTER_QASYMM8_NEON(arm_compute::cpu::neon_qu8_roialign)},
    {"qs8_neon_roialign", [](const ROIAlignSelectorData &data) { return data.dt == DataType::QASYMM8_SIGNED; },
     REGISTER_QASYMM8_SIGNED_NEON(arm_compute::cpu::neon_qs8_roialign)},
#endif //defined(ARM_COMPUTE_ENABLE_NEON)
};

/** Micro-kernel selector
 *
 * @param[in] data Selection data passed to help pick the appropriate micro-kernel
 *
 * @return A matching micro-kernel else nullptr
 */
const ROIAlignKernel *get_implementation(const ROIAlignSelectorData &data)
{
    for (const auto &uk : available_kernels)
    {
        if (uk.is_selected(data))
        {
            return &uk;
        }
    }
    return nullptr;
}

Status validate_arguments(const ITensorInfo         *input,
                          const ITensorInfo         *rois,
                          ITensorInfo               *output,
                          const ROIPoolingLayerInfo &pool_info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, rois, output);
    ARM_COMPUTE_RETURN_ERROR_ON(rois->dimension(0) != 5);
    ARM_COMPUTE_RETURN_ERROR_ON(rois->num_dimensions() > 2);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QASYMM8, DataType::QASYMM8_SIGNED,
                                                         DataType::F32, DataType::F16);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_LAYOUT_NOT_IN(input, DataLayout::NHWC, DataLayout::NCHW);
    ARM_COMPUTE_RETURN_ERROR_ON((pool_info.pooled_width() == 0) || (pool_info.pooled_height() == 0));
    ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(input);

    if (output->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_LAYOUT(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(compute_roi_align_shape(*input, *rois, pool_info),
                                                           output->tensor_shape());
    }

    if (input->data_type() == DataType::QASYMM8 || input->data_type() == DataType::QASYMM8_SIGNED)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(rois, 1, DataType::QASYMM16);

        const UniformQuantizationInfo rois_qinfo = rois->quantization_info().uniform();
        ARM_COMPUTE_RETURN_ERROR_ON(rois_qinfo.scale != 0.125f);
        ARM_COMPUTE_RETURN_ERROR_ON(rois_qinfo.offset != 0);
    }
    else
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, rois);
    }

    return Status{};
}
} // namespace

NEROIAlignLayerKernel::NEROIAlignLayerKernel()
    : _input(nullptr), _output(nullptr), _rois(nullptr), _pool_info(0, 0, 0.f)
{
}

void NEROIAlignLayerKernel::configure(const ITensor             *input,
                                      const ITensor             *rois,
                                      ITensor                   *output,
                                      const ROIPoolingLayerInfo &pool_info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output, rois);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), rois->info(), output->info(), pool_info));
    // Output auto inizialitation if not yet initialized
    const TensorShape output_shape = compute_roi_align_shape(*input->info(), *rois->info(), pool_info);
    auto_init_if_empty((*output->info()), output_shape, 1, input->info()->data_type(),
                       input->info()->quantization_info());
    output->info()->set_data_layout(input->info()->data_layout());

    // Configure kernel window
    const unsigned int num_rois = rois->info()->dimension(1);
    Window             window;
    window.set(Window::DimX, Window::Dimension(0, num_rois));
    window.set(Window::DimY, Window::Dimension(0, 1));

    // Set instance variables
    _input     = input;
    _rois      = rois;
    _output    = output;
    _pool_info = pool_info;

    INEKernel::configure(window);
}

Status NEROIAlignLayerKernel::validate(const ITensorInfo         *input,
                                       const ITensorInfo         *rois,
                                       ITensorInfo               *output,
                                       const ROIPoolingLayerInfo &pool_info)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, rois, output, pool_info));
    return Status{};
}

void NEROIAlignLayerKernel::run(const Window &window, const ThreadInfo &info)
{
    const DataLayout data_layout = _input->info()->data_layout();
    if (data_layout == DataLayout::NCHW || data_layout == DataLayout::NHWC)
    {
        const auto *uk = get_implementation(ROIAlignSelectorData{_input->info()->data_type()});
        ARM_COMPUTE_ERROR_ON(uk == nullptr || uk->ukernel == nullptr);

        uk->ukernel(_input, _output, _rois, _pool_info, window, info);
    }
    else
    {
        ARM_COMPUTE_ERROR("Invalid layout");
    }
}
} // namespace arm_compute
