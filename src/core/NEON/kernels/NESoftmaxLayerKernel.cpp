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
#include "src/core/NEON/kernels/NESoftmaxLayerKernel.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"
#include "src/core/CPP/Validate.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"

#include "src/core/NEON/kernels/softmax/impl/NEON/list.h"
#include "src/core/NEON/kernels/softmax/impl/SVE/list.h"
#include "src/core/common/Registrars.h"

namespace arm_compute
{
namespace
{
struct SoftmaxSelectorData
{
    DataType dt;
};
using SoftmaxSelectorPtr          = std::add_pointer<bool(const SoftmaxSelectorData &data)>::type;
using SoftmaxLogits1DMaxKernelPtr = std::add_pointer<void(const ITensor *, ITensor *, const Window &)>::type;
using SoftmaxLogits1DKernelPtr    = std::add_pointer<void(const ITensor *, const ITensor *, void *const, ITensor *, float, bool, const Window &)>::type;

struct SoftmaxLogits1DKernel
{
    const char              *name;
    const SoftmaxSelectorPtr is_selected;
    SoftmaxLogits1DKernelPtr ukernel;
};

struct SoftmaxLogits1DMaxKernel
{
    const char                 *name;
    const SoftmaxSelectorPtr    is_selected;
    SoftmaxLogits1DMaxKernelPtr ukernel;
};

static const SoftmaxLogits1DKernel available_logits_1d_kernels[] =
{
#if defined(__ARM_FEATURE_SVE)
    {
        "sve_softmax_logits_1d_float",
        [](const SoftmaxSelectorData & data) { return (data.dt == DataType::F32); },
        REGISTER_FP32_SVE(arm_compute::cpu::sve_softmax_logits_1d_float<float>)
    },
    {
        "sve_softmax_logits_1d_float",
        [](const SoftmaxSelectorData & data) { return (data.dt == DataType::F16); },
        REGISTER_FP16_SVE(arm_compute::cpu::sve_softmax_logits_1d_float<float16_t>)
    },
#else /* !defined(__ARM_FEATURE_SVE) */
    {
        "neon_softmax_logits_1d_float",
        [](const SoftmaxSelectorData & data) { return (data.dt == DataType::F32); },
        REGISTER_FP32_NEON(arm_compute::cpu::neon_softmax_logits_1d_float<float>)
    },
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    {
        "neon_softmax_logits_1d_float",
        [](const SoftmaxSelectorData & data) { return (data.dt == DataType::F16); },
        REGISTER_FP16_NEON(arm_compute::cpu::neon_softmax_logits_1d_float<float16_t>)
    },
#endif /* defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) */
#endif /* defined(__ARM_FEATURE_SVE) */

#if defined(__ARM_FEATURE_SVE2)
    {
        "sve_softmax_logits_1d_quantized",
        [](const SoftmaxSelectorData & data) { return (data.dt == DataType::QASYMM8); },
        REGISTER_QASYMM8_SVE(arm_compute::cpu::sve_softmax_logits_1d_quantized<qasymm8_t>)
    },
    {
        "sve_softmax_logits_1d_quantized",
        [](const SoftmaxSelectorData & data) { return (data.dt == DataType::QASYMM8_SIGNED); },
        REGISTER_QASYMM8_SIGNED_SVE(arm_compute::cpu::sve_softmax_logits_1d_quantized<qasymm8_signed_t>)
    },
#else  /* !defined(__ARM_FEATURE_SVE2) */
    {
        "neon_softmax_logits_1d_quantized",
        [](const SoftmaxSelectorData & data) { return (data.dt == DataType::QASYMM8); },
        REGISTER_QASYMM8_NEON(arm_compute::cpu::neon_softmax_logits_1d_quantized<qasymm8_t>)
    },
    {
        "neon_softmax_logits_1d_quantized",
        [](const SoftmaxSelectorData & data) { return (data.dt == DataType::QASYMM8_SIGNED); },
        REGISTER_QASYMM8_SIGNED_NEON(arm_compute::cpu::neon_softmax_logits_1d_quantized<qasymm8_signed_t>)
    },
#endif /* defined(__ARM_FEATURE_SVE2) */

};

static const SoftmaxLogits1DMaxKernel available_logits_1d_max_kernels[] =
{
#if defined(__ARM_FEATURE_SVE)
    {
        "sve_logits_1d_max",
        [](const SoftmaxSelectorData & data) { return (data.dt == DataType::F32); },
        REGISTER_FP32_SVE(arm_compute::cpu::sve_logits_1d_max<float>)
    },
    {
        "sve_logits_1d_max",
        [](const SoftmaxSelectorData & data) { return (data.dt == DataType::F16); },
        REGISTER_FP16_SVE(arm_compute::cpu::sve_logits_1d_max<float16_t>)
    },
    {
        "sve_logits_1d_max",
        [](const SoftmaxSelectorData & data) { return (data.dt == DataType::QASYMM8); },
        REGISTER_QASYMM8_SVE(arm_compute::cpu::sve_logits_1d_max<qasymm8_t>)
    },
    {
        "sve_logits_1d_max",
        [](const SoftmaxSelectorData & data) { return (data.dt == DataType::QASYMM8_SIGNED); },
        REGISTER_QASYMM8_SIGNED_SVE(arm_compute::cpu::sve_logits_1d_max<qasymm8_signed_t>)
    },
#else /* !defined(__ARM_FEATURE_SVE) */
    {
        "neon_logits_1d_max",
        [](const SoftmaxSelectorData & data) { return (data.dt == DataType::F32); },
        REGISTER_FP32_NEON(arm_compute::cpu::neon_logits_1d_max<float>)
    },
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    {
        "neon_logits_1d_max",
        [](const SoftmaxSelectorData & data) { return (data.dt == DataType::F16); },
        REGISTER_FP16_NEON(arm_compute::cpu::neon_logits_1d_max<float16_t>)
    },
#endif /* defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) */
    {
        "neon_logits_1d_max",
        [](const SoftmaxSelectorData & data) { return (data.dt == DataType::QASYMM8); },
        REGISTER_QASYMM8_NEON(arm_compute::cpu::neon_logits_1d_max<qasymm8_t>)
    },
    {
        "neon_logits_1d_max",
        [](const SoftmaxSelectorData & data) { return (data.dt == DataType::QASYMM8_SIGNED); },
        REGISTER_QASYMM8_SIGNED_NEON(arm_compute::cpu::neon_logits_1d_max<qasymm8_signed_t>)
    },
#endif /* defined(__ARM_FEATURE_SVE) */
};

const SoftmaxLogits1DKernel *get_implementation_logits(const SoftmaxSelectorData &data)
{
    for(const auto &uk : available_logits_1d_kernels)
    {
        if(uk.is_selected({ data.dt }))
        {
            return &uk;
        }
    }
    return nullptr;
}

const SoftmaxLogits1DMaxKernel *get_implementation_logits_max(const SoftmaxSelectorData &data)
{
    for(const auto &uk : available_logits_1d_max_kernels)
    {
        if(uk.is_selected({ data.dt }))
        {
            return &uk;
        }
    }
    return nullptr;
}

Status validate_arguments_logits_1d_max(const ITensorInfo &input, const ITensorInfo &output)
{
    ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(&input);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(&input, 1, DataType::QASYMM8, DataType::QASYMM8_SIGNED, DataType::F16, DataType::F32);

    // Validate in case of configured output
    if(output.total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(&input, &output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_QUANTIZATION_INFO(&input, &output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(output.tensor_shape(), TensorShape(input.tensor_shape()).set(0, 1));
    }

    return Status{};
}

} // namespace

NELogits1DMaxKernel::NELogits1DMaxKernel()
    : _border_size()
{
}

BorderSize NELogits1DMaxKernel::border_size() const
{
    return _border_size;
}

void NELogits1DMaxKernel::configure(const ITensor *input, ITensor *output)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_ERROR_ON_NULLPTR(input->info(), output->info());
    // Perform validation step
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments_logits_1d_max(*input->info(), *output->info()));
    // Configure kernel window

    // Softmax across the x dimension
    const TensorShape output_shape = TensorShape(input->info()->tensor_shape()).set(0, 1);
    // Output auto initialization if not yet initialized
    auto_init_if_empty(*output->info(), output_shape, 1, input->info()->data_type(), input->info()->quantization_info());

    Window      win = calculate_max_window(*input->info(), Steps());
    Coordinates coord;
    coord.set_num_dimensions(output->info()->num_dimensions());
    output->info()->set_valid_region(ValidRegion(coord, output->info()->tensor_shape()));

    _input  = input;
    _output = output;

    const int input_width                       = input->info()->valid_region().shape.x();
    const int num_elems_processed_per_iteration = 16U / data_size_from_type(input->info()->data_type());
    const int num_elems_read_per_iteration      = ceil_to_multiple(input_width, num_elems_processed_per_iteration);

    _border_size = BorderSize(0, num_elems_read_per_iteration - input_width, 0, 0);

    INEKernel::configure(win);
}

Status NELogits1DMaxKernel::validate(const ITensorInfo *input, const ITensorInfo *output)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments_logits_1d_max(*input, *output));

    return Status{};
}

void NELogits1DMaxKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);

    const auto *uk = get_implementation_logits_max(SoftmaxSelectorData{ _input->info()->data_type() });
    uk->ukernel(_input, _output, window);
}

namespace
{
Status validate_arguments_logits_softmax(const ITensorInfo &input, const ITensorInfo &max,
                                         const ITensorInfo &output, const float beta, const ITensorInfo &tmp, bool is_log)
{
    ARM_COMPUTE_UNUSED(beta);
    // Check input
    ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(&input);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(&input, 1, DataType::QASYMM8, DataType::QASYMM8_SIGNED, DataType::F16, DataType::F32);

    const bool is_quantized_asymmetric = is_data_type_quantized_asymmetric(input.data_type());

    // Check max
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(&input, &max);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(TensorShape(input.tensor_shape()).set(0, 1), max.tensor_shape());
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_QUANTIZATION_INFO(&input, &max);

    // Check output if configured
    if(output.total_size() != 0)
    {
        const QuantizationInfo output_quantization = is_quantized_asymmetric ? arm_compute::get_softmax_output_quantization_info(input.data_type(), is_log) : output.quantization_info();
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(&input, &output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(&input, &output);
        ARM_COMPUTE_RETURN_ERROR_ON(output.quantization_info() != output_quantization);
    }

    // Check tmp if configured
    if(tmp.total_size() != 0)
    {
        const DataType tmp_data_type = is_quantized_asymmetric ? DataType::F32 : input.data_type();
        ARM_COMPUTE_RETURN_ERROR_ON(tmp.data_type() != tmp_data_type);
        // We could potentially reduce tmp memory if we could predict or make an assumption
        // on the maximum number of threads that will run in parallel.
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(&input, &tmp);
    }

    return Status{};
}
} // namespace

template <bool IS_LOG>
NELogits1DSoftmaxKernel<IS_LOG>::NELogits1DSoftmaxKernel()
    : _input(nullptr), _max(nullptr), _output(nullptr), _beta(1.0f), _tmp(nullptr)
{
}

template <bool IS_LOG>
void NELogits1DSoftmaxKernel<IS_LOG>::configure(const ITensor *input, const ITensor *max, ITensor *output, const float beta, ITensor *tmp)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, max, output, tmp);
    ARM_COMPUTE_ERROR_ON_NULLPTR(input->info(), max->info(), output->info(), tmp->info());
    // Perform validation step
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments_logits_softmax(*input->info(), *max->info(), *output->info(), beta, *tmp->info(), IS_LOG));

    // Configure kernel window
    const bool is_quantized_asymmetric = is_data_type_quantized_asymmetric(input->info()->data_type());

    // Output auto initialization if not yet initialized
    const QuantizationInfo output_quantization = is_quantized_asymmetric ? arm_compute::get_softmax_output_quantization_info(input->info()->data_type(), IS_LOG) : output->info()->quantization_info();
    auto_init_if_empty(*output->info(), TensorInfo(*input->info()).set_quantization_info(output_quantization).reset_padding());

    // Tmp auto initialization if not yet initialized
    const DataType tmp_data_type = is_quantized_asymmetric ? DataType::F32 : input->info()->data_type();
    auto_init_if_empty(*tmp->info(), TensorInfo(*input->info()).set_data_type(tmp_data_type).reset_padding());

    // Configure kernel window
    Window      win = calculate_max_window(*max->info(), Steps());
    Coordinates coord;
    coord.set_num_dimensions(output->info()->num_dimensions());
    output->info()->set_valid_region(ValidRegion(coord, output->info()->tensor_shape()));

    _input  = input;
    _max    = max;
    _output = output;
    _beta   = beta;
    _tmp    = tmp;

    INEKernel::configure(win);
}

template <bool IS_LOG>
Status NELogits1DSoftmaxKernel<IS_LOG>::validate(const ITensorInfo *input, const ITensorInfo *max,
                                                 const ITensorInfo *output, const float beta, const ITensorInfo *tmp)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, max, output, tmp);
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments_logits_softmax(*input, *max, *output, beta, *tmp, IS_LOG));

    return Status{};
}

template <bool IS_LOG>
void NELogits1DSoftmaxKernel<IS_LOG>::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);

    const unsigned int num_elems_processed_per_iteration = _input->info()->valid_region().shape.x();
    const unsigned int tmp_size_for_thread               = _tmp->info()->element_size() * num_elems_processed_per_iteration;

    ARM_COMPUTE_ERROR_ON(_tmp->info()->total_size() < (info.num_threads * tmp_size_for_thread));

    void *tmp_for_thread = _tmp->buffer() + (info.thread_id * tmp_size_for_thread);

    const auto *uk = get_implementation_logits(SoftmaxSelectorData{ _input->info()->data_type() });
    uk->ukernel(_input, _max, tmp_for_thread, _output, _beta, IS_LOG, window);
}

template class NELogits1DSoftmaxKernel<true>;
template class NELogits1DSoftmaxKernel<false>;

} // namespace arm_compute
