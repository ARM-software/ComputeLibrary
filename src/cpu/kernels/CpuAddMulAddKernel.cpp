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
#include "src/cpu/kernels/CpuAddMulAddKernel.h"

#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Validate.h"

#include "src/core/common/Registrars.h"
#include "src/core/CPP/Validate.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"
#include "src/cpu/kernels/addmuladd/list.h"

namespace arm_compute
{
namespace cpu
{
namespace kernels
{
namespace
{
static const std::vector<CpuAddMulAddKernel::AddMulAddKernel> available_kernels = {
#ifdef __aarch64__
    {"neon_fp32_add_mul_add", [](const DataTypeISASelectorData &data) { return (data.dt == DataType::F32); },
     REGISTER_FP32_NEON(arm_compute::cpu::add_mul_add_fp32_neon)},
    {"neon_fp16_add_mul_add", [](const DataTypeISASelectorData &data) { return (data.dt == DataType::F16); },
     REGISTER_FP16_NEON(arm_compute::cpu::add_mul_add_fp16_neon)},
    {"neon_qasymm8_add_mul_add", [](const DataTypeISASelectorData &data) { return (data.dt == DataType::QASYMM8); },
     REGISTER_QASYMM8_NEON(arm_compute::cpu::add_mul_add_u8_neon)},
    {"neon_qasymm8_signed_add_mul_add",
     [](const DataTypeISASelectorData &data) { return (data.dt == DataType::QASYMM8_SIGNED); },
     REGISTER_QASYMM8_SIGNED_NEON(arm_compute::cpu::add_mul_add_s8_neon)}
#endif // __aarch64__
};

Status validate_arguments(const ITensorInfo         *input1,
                          const ITensorInfo         *input2,
                          const ITensorInfo         *bn_mul,
                          const ITensorInfo         *bn_add,
                          const ITensorInfo         *add_output,
                          const ITensorInfo         *final_output,
                          ConvertPolicy              policy,
                          const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input1, input2, bn_mul, bn_add, final_output);

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(policy != ConvertPolicy::SATURATE, "Only Saturate Policy is supported");

    using ActFunction          = ActivationLayerInfo::ActivationFunction;
    const ActFunction act_func = act_info.activation();
    ARM_COMPUTE_RETURN_ERROR_ON_MSG((act_func != ActFunction::BOUNDED_RELU && act_func != ActFunction::RELU &&
                                     act_func != ActFunction::LU_BOUNDED_RELU && act_func != ActFunction::IDENTITY),
                                    "Only RELU Family activations, or no activation, is supported");

    ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(input1);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input1, 1, DataType::QASYMM8, DataType::QASYMM8_SIGNED,
                                                         DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input1, input2);

    if (is_data_type_quantized(input1->data_type()))
    {
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(bn_mul, 1, DataType::F32);
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(bn_add, 1, DataType::F32);
    }
    else
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input1, bn_mul);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input1, bn_add);
    }

    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(input1, input2); // No broadcasting
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(bn_mul, bn_add);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(bn_mul->num_dimensions() != 1, "BatchNorm coefficients should be 1D array");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(bn_mul->tensor_shape()[0] != input1->tensor_shape()[0],
                                    "First dimensions of inputs and batchNorm coefs should match");

    // Validate in case we have add layer's output (intermediate) initialized
    if (add_output != nullptr && add_output->total_size() > 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input1, add_output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(input1, add_output);
    }

    // Validate in case final output has been initialized
    if (final_output->total_size() > 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input1, final_output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(input1, final_output);
    }

    const auto uk = CpuAddMulAddKernel::get_implementation<DataTypeISASelectorData>(
        DataTypeISASelectorData{input1->data_type(), CPUInfo::get().get_isa()});
    ARM_COMPUTE_RETURN_ERROR_ON(uk == nullptr || uk->ukernel == nullptr);

    return Status{};
}
} // namespace

void CpuAddMulAddKernel::configure(const ITensorInfo         *input1,
                                   const ITensorInfo         *input2,
                                   const ITensorInfo         *bn_mul,
                                   const ITensorInfo         *bn_add,
                                   ITensorInfo               *add_output,
                                   ITensorInfo               *final_output,
                                   ConvertPolicy              policy,
                                   const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_UNUSED(bn_mul, bn_add, input2);
    ARM_COMPUTE_ERROR_ON_NULLPTR(input1, input2, bn_add, bn_mul, final_output);
    ARM_COMPUTE_ERROR_THROW_ON(
        validate_arguments(input1, input2, bn_mul, bn_add, add_output, final_output, policy, act_info));

    const auto uk = CpuAddMulAddKernel::get_implementation<DataTypeISASelectorData>(
        DataTypeISASelectorData{input1->data_type(), CPUInfo::get().get_isa()});
    ARM_COMPUTE_ERROR_ON_NULLPTR(uk);
    ARM_COMPUTE_ERROR_ON(uk->ukernel == nullptr);

    _policy     = policy;
    _act_info   = act_info;
    _run_method = uk->ukernel;
    _name       = std::string("CpuAddMulAddKernel/").append(uk->name);

    // Auto initialize outputs if not initialized
    set_shape_if_empty(*final_output, input1->tensor_shape());
    set_data_type_if_unknown(*final_output, input1->data_type());

    if (add_output != nullptr)
    {
        set_shape_if_empty(*add_output, input1->tensor_shape());
        set_data_type_if_unknown(*add_output, input1->data_type());
    }

    // Configure kernel window
    Window win;
    win = calculate_max_window(*final_output, Steps());
    ICpuKernel::configure(win);
}

Status CpuAddMulAddKernel::validate(const ITensorInfo         *input1,
                                    const ITensorInfo         *input2,
                                    const ITensorInfo         *bn_mul,
                                    const ITensorInfo         *bn_add,
                                    const ITensorInfo         *add_output,
                                    const ITensorInfo         *final_output,
                                    ConvertPolicy              policy,
                                    const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input1, input2, bn_mul, bn_add, final_output);

    ARM_COMPUTE_RETURN_ON_ERROR(
        validate_arguments(input1, input2, bn_mul, bn_add, add_output, final_output, policy, act_info));

    return Status{};
}

void CpuAddMulAddKernel::run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);

    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICpuKernel::window(), window);

    ARM_COMPUTE_ERROR_ON(tensors.empty());
    ARM_COMPUTE_ERROR_ON(_run_method == nullptr);

    const ITensor *input1       = tensors.get_const_tensor(TensorType::ACL_SRC_0);
    const ITensor *input2       = tensors.get_const_tensor(TensorType::ACL_SRC_1);
    const ITensor *bn_mul       = tensors.get_const_tensor(TensorType::ACL_SRC_2);
    const ITensor *bn_add       = tensors.get_const_tensor(TensorType::ACL_SRC_3);
    ITensor       *add_output   = tensors.get_tensor(TensorType::ACL_DST_0);
    ITensor       *final_output = tensors.get_tensor(TensorType::ACL_DST_1);

    _run_method(input1, input2, bn_mul, bn_add, add_output, final_output, _policy, _act_info, window);
}

const char *CpuAddMulAddKernel::name() const
{
    return _name.c_str();
}

const std::vector<CpuAddMulAddKernel::AddMulAddKernel> &CpuAddMulAddKernel::get_available_kernels()
{
    return available_kernels;
}
} // namespace kernels
} // namespace cpu
} // namespace arm_compute
