/*
 * Copyright (c) 2017 ARM Limited.
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
#include "arm_compute/runtime/CL/functions/CLSoftmaxLayer.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/ICLKernel.h"
#include "arm_compute/core/CL/kernels/CLSoftmaxLayerKernel.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/runtime/CL/CLMemoryGroup.h"
#include "arm_compute/runtime/CL/CLScheduler.h"

using namespace arm_compute;

CLSoftmaxLayer::CLSoftmaxLayer(std::shared_ptr<IMemoryManager> memory_manager)
    : _memory_group(std::move(memory_manager)), _max_kernel(), _shift_exp_sum_kernel(), _max_shift_exp_sum_kernel(), _norm_kernel(), _max(), _sum(), _tmp(), _run_legacy_path(false)
{
}

void CLSoftmaxLayer::configure(const ICLTensor *input, ICLTensor *output, float beta)
{
    // Perform validation step
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_ERROR_THROW_ON(CLSoftmaxLayer::validate(input->info(), output->info()));

    // Create intermediate tensors shapes
    const TensorInfo input_info    = input->info()->clone()->reset_padding().set_is_resizable(true);
    DataType         tmp_data_type = is_data_type_quantized_asymmetric(input->info()->data_type()) ? DataType::S32 : input->info()->data_type();
    TensorInfo       tensor_info_tmp(input_info.clone()->set_data_type(tmp_data_type));
    _tmp.allocator()->init(tensor_info_tmp);

    TensorShape max_sum_shape = input->info()->tensor_shape();
    max_sum_shape.set(0, 1);
    _max.allocator()->init(input_info.clone()->set_tensor_shape(max_sum_shape));
    _sum.allocator()->init(input_info.clone()->set_tensor_shape(max_sum_shape).set_data_type(tmp_data_type));

    // Set GPU target to kernels
    _max_shift_exp_sum_kernel.set_target(CLScheduler::get().target());

    // Manage intermediate buffers
    _memory_group.manage(&_tmp);
    _memory_group.manage(&_max);
    _memory_group.manage(&_sum);

    // Configure kernels
    _run_legacy_path = is_data_type_quantized_asymmetric(input->info()->data_type());
    if(_run_legacy_path)
    {
        _max_kernel.configure(input, &_max);
        _shift_exp_sum_kernel.configure(input, &_max, &_tmp, &_sum, beta);
    }
    else
    {
        _max_shift_exp_sum_kernel.configure(input, &_max, &_tmp, &_sum, beta);
    }
    _norm_kernel.configure(&_tmp, &_sum, output, beta);

    // Allocate intermediate buffers
    _tmp.allocator()->allocate();
    _max.allocator()->allocate();
    _sum.allocator()->allocate();
}

Status CLSoftmaxLayer::validate(const ITensorInfo *input, const ITensorInfo *output)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output);

    // Create intermediate tensor info
    DataType   tmp_data_type = is_data_type_quantized_asymmetric(input->data_type()) ? DataType::S32 : input->data_type();
    TensorInfo tensor_info_tmp(input->clone()->set_data_type(tmp_data_type));

    TensorShape max_sum_shape = input->tensor_shape();
    max_sum_shape.set(0, 1);
    TensorInfo tensor_info_max(input->clone()->set_tensor_shape(max_sum_shape));
    TensorInfo tensor_info_sum(input->clone()->set_tensor_shape(max_sum_shape).set_data_type(tmp_data_type).set_quantization_info(QuantizationInfo()));

    bool run_legacy_path = is_data_type_quantized_asymmetric(input->data_type());
    if(run_legacy_path)
    {
        ARM_COMPUTE_RETURN_ON_ERROR(CLLogits1DMaxKernel::validate(input, &tensor_info_max));
        ARM_COMPUTE_RETURN_ON_ERROR(CLLogits1DShiftExpSumKernel::validate(input, &tensor_info_max, &tensor_info_tmp, &tensor_info_sum));
    }
    else
    {
        ARM_COMPUTE_RETURN_ON_ERROR(CLLogits1DMaxShiftExpSumKernel::validate(input, &tensor_info_max, &tensor_info_tmp, &tensor_info_sum));
    }
    ARM_COMPUTE_RETURN_ON_ERROR(CLLogits1DNormKernel::validate(&tensor_info_tmp, &tensor_info_sum, output));

    return Status{};
}

void CLSoftmaxLayer::run()
{
    _memory_group.acquire();

    // Force to use the new fused kernel
    if(_run_legacy_path)
    {
        CLScheduler::get().enqueue(_max_kernel, false);
        CLScheduler::get().enqueue(_shift_exp_sum_kernel, false);
    }
    else
    {
        CLScheduler::get().enqueue(_max_shift_exp_sum_kernel, false);
    }
    CLScheduler::get().enqueue(_norm_kernel);

    _memory_group.release();
}
