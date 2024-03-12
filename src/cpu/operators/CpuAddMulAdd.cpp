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
#include "src/cpu/operators/CpuAddMulAdd.h"

#include "arm_compute/core/experimental/Types.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"

#include "src/common/utils/Log.h"
#include "src/core/helpers/MemoryHelpers.h"
#include "src/cpu/kernels/CpuAddMulAddKernel.h"
#include "src/cpu/utils/CpuAuxTensorHandler.h"

namespace arm_compute
{
namespace cpu
{
void CpuAddMulAdd::configure(const ITensorInfo         *input1,
                             const ITensorInfo         *input2,
                             const ITensorInfo         *bn_mul,
                             const ITensorInfo         *bn_add,
                             ITensorInfo               *add_output,
                             ITensorInfo               *final_output,
                             ConvertPolicy              policy,
                             const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_LOG_PARAMS(input1, input2, bn_mul, bn_add, add_output, final_output, policy, act_info);

    auto k = std::make_unique<kernels::CpuAddMulAddKernel>();

    const DataType data_type = input1->data_type();
    if (is_data_type_quantized(data_type))
    {
        _dequantize_bn_mul.configure(bn_mul, &_dequantized_bn_mul);
        _dequantize_bn_add.configure(bn_add, &_dequantized_bn_add);

        k->configure(input1, input2, &_dequantized_bn_mul, &_dequantized_bn_add, add_output, final_output, policy,
                     act_info);

        // Save auxilary memory requirements after configuration
        _aux_mem[DequantizedBnMul] =
            experimental::MemoryInfo(offset_int_vec(DequantizedBnMul), experimental::MemoryLifetime::Temporary,
                                     _dequantized_bn_mul.total_size());
        _aux_mem[DequantizedBnAdd] =
            experimental::MemoryInfo(offset_int_vec(DequantizedBnAdd), experimental::MemoryLifetime::Temporary,
                                     _dequantized_bn_add.total_size());
    }
    else
    {
        k->configure(input1, input2, bn_mul, bn_add, add_output, final_output, policy, act_info);
    }

    _kernel = std::move(k);
}

Status CpuAddMulAdd::validate(const ITensorInfo         *input1,
                              const ITensorInfo         *input2,
                              const ITensorInfo         *bn_mul,
                              const ITensorInfo         *bn_add,
                              const ITensorInfo         *add_output,
                              const ITensorInfo         *final_output,
                              ConvertPolicy              policy,
                              const ActivationLayerInfo &act_info)
{
    const DataType data_type = input1->data_type();
    if (is_data_type_quantized(data_type))
    {
        TensorInfo dequantized_bn_mul = bn_mul->clone()->set_data_type(DataType::F32);
        TensorInfo dequantized_bn_add = bn_add->clone()->set_data_type(DataType::F32);

        ARM_COMPUTE_RETURN_ON_ERROR(CpuDequantize::validate(bn_mul, &dequantized_bn_mul));
        ARM_COMPUTE_RETURN_ON_ERROR(CpuDequantize::validate(bn_add, &dequantized_bn_add));

        return kernels::CpuAddMulAddKernel::validate(input1, input2, &dequantized_bn_mul, &dequantized_bn_add,
                                                     add_output, final_output, policy, act_info);
    }
    else
    {
        return kernels::CpuAddMulAddKernel::validate(input1, input2, bn_mul, bn_add, add_output, final_output, policy,
                                                     act_info);
    }
}

void CpuAddMulAdd::run(ITensorPack &tensors)
{
    const DataType data_type = tensors.get_const_tensor(TensorType::ACL_SRC_0)->info()->data_type();

    if (is_data_type_quantized(data_type))
    {
        const ITensor *bn_mul = tensors.get_const_tensor(TensorType::ACL_SRC_2);
        const ITensor *bn_add = tensors.get_const_tensor(TensorType::ACL_SRC_3);

        CpuAuxTensorHandler dequantized_bn_mul_handler(offset_int_vec(DequantizedBnMul), _dequantized_bn_mul, tensors,
                                                       true);
        CpuAuxTensorHandler dequantized_bn_add_handler(offset_int_vec(DequantizedBnAdd), _dequantized_bn_add, tensors,
                                                       true);

        ITensorPack dequantize_mul_pack = {{TensorType::ACL_SRC_0, bn_mul},
                                           {TensorType::ACL_DST_0, dequantized_bn_mul_handler.get()}};

        ITensorPack dequantize_add_pack = {{TensorType::ACL_SRC_0, bn_add},
                                           {TensorType::ACL_DST_0, dequantized_bn_add_handler.get()}};

        _dequantize_bn_mul.run(dequantize_mul_pack);
        _dequantize_bn_add.run(dequantize_add_pack);

        ITensorPack add_mul_add_pack = {
            {TensorType::ACL_SRC_0, tensors.get_const_tensor(TensorType::ACL_SRC_0)},
            {TensorType::ACL_SRC_1, tensors.get_const_tensor(TensorType::ACL_SRC_1)},
            {TensorType::ACL_SRC_2, dequantized_bn_mul_handler.get()},
            {TensorType::ACL_SRC_3, dequantized_bn_add_handler.get()},
            {TensorType::ACL_DST_0, tensors.get_tensor(TensorType::ACL_DST_0)},
            {TensorType::ACL_DST_1, tensors.get_tensor(TensorType::ACL_DST_1)},
        };

        NEScheduler::get().schedule_op(_kernel.get(), Window::DimY, _kernel->window(), add_mul_add_pack);
    }
    else
    {
        NEScheduler::get().schedule_op(_kernel.get(), Window::DimY, _kernel->window(), tensors);
    }
}

experimental::MemoryRequirements CpuAddMulAdd::workspace() const
{
    return _aux_mem;
}

} // namespace cpu
} // namespace arm_compute
