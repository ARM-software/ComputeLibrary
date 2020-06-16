/*
 * Copyright (c) 2017-2020 ARM Limited.
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
#include "arm_compute/core/NEON/kernels/NEReshapeLayerKernel.h"

#include "arm_compute/core/AccessWindowStatic.h"
#include "arm_compute/core/CPP/Validate.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/IAccessWindow.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/NEON/INEKernel.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"

#include <cstdint>

/** [NEReshapeLayerKernel Kernel] **/
namespace arm_compute
{
namespace
{
Status validate_arguments(const ITensorInfo *input, const ITensorInfo *output)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output);
    // Note: ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(input) is not needed here as this kernel doesn't use NEON FP16 instructions.
    ARM_COMPUTE_RETURN_ERROR_ON(input->data_type() == DataType::UNKNOWN);

    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    ARM_COMPUTE_RETURN_ERROR_ON(input->tensor_shape().total_size() != output->tensor_shape().total_size());
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_QUANTIZATION_INFO(input, output);

    return Status{};
}

template <typename T>
inline void reshape_tensor(const Window &window, const ITensor *input, ITensor *output)
{
    const TensorShape &input_shape  = input->info()->tensor_shape();
    const TensorShape &output_shape = output->info()->tensor_shape();
    Coordinates        output_coord{};

    Iterator in(input, window);

    execute_window_loop(window, [&](const Coordinates & id)
    {
        output_coord                                                 = index2coords(output_shape, coords2index(input_shape, id));
        *reinterpret_cast<T *>(output->ptr_to_element(output_coord)) = *reinterpret_cast<T *>(in.ptr());
    },
    in);
}
} // namespace

void NEReshapeLayerKernel::configure(const ITensorInfo *input, ITensorInfo *output)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input, output));

    // Configure kernel window
    Window win = calculate_max_window(*input);

    // Set the output valid region
    output->set_valid_region(ValidRegion(Coordinates(), output->tensor_shape()));

    INEKernel::configure(win);
}

void NEReshapeLayerKernel::run_op(const std::vector<InputTensor> &inputs, const std::vector<OutputTensor> &outputs, const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);

    const auto src = inputs[0].tensor;
    auto       dst = outputs[0].tensor;

    switch(src->info()->data_type())
    {
        case DataType::U8:
        case DataType::S8:
        case DataType::QASYMM8:
        case DataType::QASYMM8_SIGNED:
            reshape_tensor<uint8_t>(window, src, dst);
            break;
        case DataType::U16:
        case DataType::S16:
        case DataType::F16:
            reshape_tensor<uint16_t>(window, src, dst);
            break;
        case DataType::U32:
        case DataType::S32:
        case DataType::F32:
            reshape_tensor<uint32_t>(window, src, dst);
            break;
        default:
            ARM_COMPUTE_ERROR("Unsupported data type!");
    }
}

Status NEReshapeLayerKernel::validate(const ITensorInfo *input, const ITensorInfo *output)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output));

    return Status{};
}
} // namespace arm_compute
/** [NEReshapeLayerKernel Kernel] **/
