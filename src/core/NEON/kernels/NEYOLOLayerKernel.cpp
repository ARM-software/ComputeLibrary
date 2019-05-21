/*
 * Copyright (c) 2018-2019 ARM Limited.
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
#include "arm_compute/core/NEON/kernels/NEYOLOLayerKernel.h"

#include "arm_compute/core/CPP/Validate.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/NEON/NEAsymm.h"
#include "arm_compute/core/NEON/NEFixedPoint.h"
#include "arm_compute/core/NEON/NEMath.h"
#include "arm_compute/core/NEON/kernels/detail/NEActivationFunctionDetail.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

#include <arm_neon.h>

using namespace arm_compute;
namespace
{
Status validate_arguments(const ITensorInfo *input, const ITensorInfo *output, const ActivationLayerInfo &act_info, int32_t num_classes)
{
    ARM_COMPUTE_UNUSED(act_info);
    ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(input);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON(input->data_layout() == DataLayout::UNKNOWN);
    ARM_COMPUTE_RETURN_ERROR_ON(act_info.activation() != ActivationLayerInfo::ActivationFunction::LOGISTIC);

    const unsigned int channel_idx = get_data_layout_dimension_index(input->data_layout(), DataLayoutDimension::CHANNEL);
    ARM_COMPUTE_RETURN_ERROR_ON(num_classes <= 0);
    ARM_COMPUTE_RETURN_ERROR_ON((input->dimension(channel_idx) % (num_classes + 5)) != 0);

    // Checks performed when output is configured
    if((output != nullptr) && (output->total_size() != 0))
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    }

    return Status{};
}

std::pair<Status, Window> validate_and_configure_window(ITensorInfo *input, ITensorInfo *output)
{
    if(output != nullptr)
    {
        ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);

        // Output auto inizialitation if not yet initialized
        auto_init_if_empty(*output, *input);
    }

    const bool         is_nchw                           = input->data_layout() == DataLayout::NCHW;
    const unsigned int num_elems_processed_per_iteration = is_nchw ? 16 / input->element_size() : 1;

    Window win            = calculate_max_window(*input, Steps(num_elems_processed_per_iteration));
    bool   window_changed = false;

    if(output != nullptr)
    {
        AccessWindowHorizontal input_access(input, 0, num_elems_processed_per_iteration);
        AccessWindowHorizontal output_access(output, 0, num_elems_processed_per_iteration);
        window_changed = update_window_and_padding(win, input_access, output_access);
        output_access.set_valid_region(win, input->valid_region());
    }
    else
    {
        window_changed = update_window_and_padding(win, AccessWindowHorizontal(input, 0, num_elems_processed_per_iteration));
    }

    Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
    return std::make_pair(err, win);
}
} // namespace

NEYOLOLayerKernel::NEYOLOLayerKernel()
    : _func(nullptr), _input(nullptr), _output(nullptr), _act_info(), _num_classes()
{
}

void NEYOLOLayerKernel::yolo_layer_fp32_nchw(const Window &window)
{
    Iterator input(_input, window);
    Iterator output(_output, window);

    execute_window_loop(window, [&](const Coordinates & id)
    {
        float32x4_t res = vld1q_f32(reinterpret_cast<float *>(input.ptr()));

        const int  box_ch_id = id.z() % (_num_classes + 5);
        const bool activate  = box_ch_id != 2 && box_ch_id != 3;

        // Perform activation
        if(activate)
        {
            auto activation = ::detail::logistic<float, 4>(_act_info);
            activation(res);
        }

        // Store results
        vst1q_f32(reinterpret_cast<float *>(output.ptr()), res);
    },
    input, output);
}

void NEYOLOLayerKernel::yolo_layer_fp32_nhwc(const Window &window)
{
    Iterator input(_input, window);
    Iterator output(_output, window);

    execute_window_loop(window, [&](const Coordinates & id)
    {
        float res = *(reinterpret_cast<float *>(input.ptr()));

        const int  box_ch_id = id.x() % (_num_classes + 5);
        const bool activate  = box_ch_id != 2 && box_ch_id != 3;

        // Perform activation
        if(activate)
        {
            res = 1.f / (1.f + std::exp(-res));
        }

        // Store result
        *(reinterpret_cast<float *>(output.ptr())) = res;
    },
    input, output);
}

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
void NEYOLOLayerKernel::yolo_layer_fp16_nchw(const Window &window)
{
    Iterator input(_input, window);
    Iterator output(_output, window);

    execute_window_loop(window, [&](const Coordinates & id)
    {
        float16x8_t res = vld1q_f16(reinterpret_cast<float16_t *>(input.ptr()));

        const int  box_ch_id = id.z() % (_num_classes + 5);
        const bool activate  = box_ch_id != 2 && box_ch_id != 3;

        // Perform activation
        if(activate)
        {
            auto activation = ::detail::logistic<float16_t, 8>(_act_info);
            activation(res);
        }

        // Store results
        vst1q_f16(reinterpret_cast<float16_t *>(output.ptr()), res);
    },
    input, output);
}

void NEYOLOLayerKernel::yolo_layer_fp16_nhwc(const Window &window)
{
    Iterator input(_input, window);
    Iterator output(_output, window);

    execute_window_loop(window, [&](const Coordinates & id)
    {
        float16_t res = *(reinterpret_cast<float16_t *>(input.ptr()));

        const int  box_ch_id = id.x() % (_num_classes + 5);
        const bool activate  = box_ch_id != 2 && box_ch_id != 3;

        // Perform activation
        if(activate)
        {
            res = 1.f / (1.f + std::exp(-res));
        }

        // Store result
        *(reinterpret_cast<float16_t *>(output.ptr())) = res;
    },
    input, output);
}
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */

void NEYOLOLayerKernel::configure(ITensor *input, ITensor *output, const ActivationLayerInfo &act_info, int32_t num_classes)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), (output != nullptr) ? output->info() : nullptr, act_info, num_classes));

    _input       = input;
    _output      = output;
    _act_info    = act_info;
    _num_classes = num_classes;

    switch(_input->info()->data_type())
    {
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        case DataType::F16:
            _func = (_input->info()->data_layout() == DataLayout::NHWC) ? &NEYOLOLayerKernel::yolo_layer_fp16_nhwc : &NEYOLOLayerKernel::yolo_layer_fp16_nchw;
            break;
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        case DataType::F32:
            _func = (_input->info()->data_layout() == DataLayout::NHWC) ? &NEYOLOLayerKernel::yolo_layer_fp32_nhwc : &NEYOLOLayerKernel::yolo_layer_fp32_nchw;
            break;
        default:
            ARM_COMPUTE_ERROR("Element size not supported");
            break;
    }

    // Configure kernel window
    auto win_config = validate_and_configure_window(input->info(), (output == nullptr) ? nullptr : output->info());
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    ICPPKernel::configure(win_config.second);
}

Status NEYOLOLayerKernel::validate(const ITensorInfo *input, const ITensorInfo *output, const ActivationLayerInfo &act_info, int32_t num_classes)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output, act_info, num_classes));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(input->clone().get(), (output == nullptr) ? nullptr : output->clone().get()).first);

    return Status{};
}

void NEYOLOLayerKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);
    ARM_COMPUTE_ERROR_ON(_func == nullptr);

    (this->*_func)(window);
}
