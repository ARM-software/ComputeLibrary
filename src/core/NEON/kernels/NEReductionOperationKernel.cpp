/*
 * Copyright (c) 2017-2024 Arm Limited.
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
#include "src/core/NEON/kernels/NEReductionOperationKernel.h"

#include "arm_compute/core/Coordinates.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/core/Validate.h"

#include "src/core/common/Registrars.h"
#include "src/core/CPP/Validate.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"
#include "src/core/NEON/INEKernel.h"
#include "src/core/NEON/wrapper/wrapper.h"
#include "src/cpu/kernels/reduction_layer/generic/neon/list.h"

namespace arm_compute
{

void NEReductionOperationKernel::reduce_op()
{
    const bool is_complex = (_input->info()->num_channels() == 2);

    if (is_complex)
    {
        switch (_reduction_axis)
        {
            case 2:
                switch (_input->info()->data_type())
                {
                    case DataType::F32:
                    {
                        switch (_op)
                        {
                            case ReductionOperation::SUM:
                                _func = REGISTER_FP32_NEON(cpu::reduce_RedOpYZW_complex_reduceZ_float32_4_2_SUM);
                                break;
                            default:
                                ARM_COMPUTE_ERROR("Not supported");
                                break;
                        }
                        break;
                    }
                    default:
                    {
                        ARM_COMPUTE_ERROR("Not supported");
                        break;
                    }
                }
                break;
            default:
            {
                ARM_COMPUTE_ERROR("Not supported");
                break;
            }
        }
        return;
    }

    switch (_reduction_axis)
    {
        case 0:
        {
            switch (_input->info()->data_type())
            {
                case DataType::QASYMM8:
                {
                    _func = REGISTER_QASYMM8_NEON(cpu::reduce_RedOpX_reduceX_qasymm8);
                    break;
                }
                case DataType::QASYMM8_SIGNED:
                {
                    _func = REGISTER_QASYMM8_SIGNED_NEON(cpu::reduce_RedOpX_reduceX_qasymm8_signed);
                    break;
                }
#ifdef ARM_COMPUTE_ENABLE_FP16
                case DataType::F16:
                {
                    _func = REGISTER_FP16_NEON(cpu::reduce_RedOpX_reduceX_float16_8);
                    break;
                }
#endif // ARM_COMPUTE_ENABLE_FP16
                case DataType::F32:
                {
                    _func = REGISTER_FP32_NEON(cpu::reduce_RedOpX_reduceX_float32_4);
                    break;
                }
                case DataType::S32:
                {
                    _func = REGISTER_INTEGER_NEON(cpu::reduce_RedOpX_reduceX_S32_4);
                    break;
                }
                default:
                {
                    ARM_COMPUTE_ERROR("Not supported");
                    break;
                }
            }
            break;
        }
        case 1:
        {
            switch (_input->info()->data_type())
            {
                case DataType::QASYMM8:
                {
                    _func = REGISTER_QASYMM8_NEON(cpu::reduce_RedOpYZW_reduceY_qasymm8);
                    break;
                }
                case DataType::QASYMM8_SIGNED:
                {
                    _func = REGISTER_QASYMM8_SIGNED_NEON(cpu::reduce_RedOpYZW_reduceY_qasymm8_signed);
                    break;
                }
#ifdef ARM_COMPUTE_ENABLE_FP16
                case DataType::F16:
                {
                    _func = REGISTER_FP16_NEON(cpu::reduce_RedOpYZW_reduceY_float16_8);
                    break;
                }
#endif // ARM_COMPUTE_ENABLE_FP16
                case DataType::F32:
                {
                    _func = REGISTER_FP32_NEON(cpu::reduce_RedOpYZW_reduceY_float32_4);
                    break;
                }
                case DataType::S32:
                {
                    _func = REGISTER_INTEGER_NEON(cpu::reduce_RedOpYZW_reduceY_S32_4);
                    break;
                }
                default:
                {
                    ARM_COMPUTE_ERROR("Not supported");
                    break;
                }
            }
            break;
        }
        case 2:
        {
            switch (_input->info()->data_type())
            {
                case DataType::QASYMM8:
                {
                    _func = REGISTER_QASYMM8_NEON(cpu::reduce_RedOpYZW_reduceZ_qasymm8);
                    break;
                }
                case DataType::QASYMM8_SIGNED:
                {
                    _func = REGISTER_QASYMM8_SIGNED_NEON(cpu::reduce_RedOpYZW_reduceZ_qasymm8_signed);
                    break;
                }
#ifdef ARM_COMPUTE_ENABLE_FP16
                case DataType::F16:
                {
                    _func = REGISTER_FP16_NEON(cpu::reduce_RedOpYZW_reduceZ_float16_8);
                    break;
                }
#endif // ARM_COMPUTE_ENABLE_FP16
                case DataType::F32:
                {
                    _func = REGISTER_FP32_NEON(cpu::reduce_RedOpYZW_reduceZ_float32_4);
                    break;
                }
                case DataType::S32:
                {
                    _func = REGISTER_INTEGER_NEON(cpu::reduce_RedOpYZW_reduceZ_S32_4);
                    break;
                }
                default:
                {
                    std::cout << int(_input->info()->data_type()) << std::endl;
                    ARM_COMPUTE_ERROR("Not supported");
                    break;
                }
            }
            break;
        }
        case 3:
        {
            switch (_input->info()->data_type())
            {
                case DataType::QASYMM8:
                {
                    _func = REGISTER_QASYMM8_NEON(cpu::reduce_RedOpYZW_reduceW_qasymm8);
                    break;
                }
                case DataType::QASYMM8_SIGNED:
                {
                    _func = REGISTER_QASYMM8_SIGNED_NEON(cpu::reduce_RedOpYZW_reduceW_qasymm8_signed);
                    break;
                }
#ifdef ARM_COMPUTE_ENABLE_FP16
                case DataType::F16:
                {
                    _func = REGISTER_FP16_NEON(cpu::reduce_RedOpYZW_reduceW_float16_8);
                    break;
                }
#endif // ARM_COMPUTE_ENABLE_FP16
                case DataType::F32:
                {
                    _func = REGISTER_FP32_NEON(cpu::reduce_RedOpYZW_reduceW_float32_4);
                    break;
                }
                case DataType::S32:
                {
                    _func = REGISTER_INTEGER_NEON(cpu::reduce_RedOpYZW_reduceW_S32_4);
                    break;
                }
                default:
                {
                    ARM_COMPUTE_ERROR("Not supported");
                    break;
                }
            }
            break;
        }
        default:
        {
            ARM_COMPUTE_ERROR("Unsupported reduction axis");
            break;
        }
    }
}

Status validate_arguments(const ITensorInfo *input, const ITensorInfo *output, unsigned int axis, ReductionOperation op)
{
    ARM_COMPUTE_UNUSED(op);

    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(input);

    if (input->num_channels() == 1)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QASYMM8_SIGNED, DataType::QASYMM8,
                                                             DataType::S32, DataType::F16, DataType::F32);
    }
    else
    {
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 2, DataType::F32);
        ARM_COMPUTE_RETURN_ERROR_ON(op != ReductionOperation::SUM);
        ARM_COMPUTE_RETURN_ERROR_ON(axis != 2);
    }

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(axis >= TensorShape::num_max_dimensions,
                                    "Reduction axis greater than max number of dimensions");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(axis > 3, "Unsupported reduction axis");

    if (output->total_size() != 0)
    {
        bool is_arg_min_max = (op == ReductionOperation::ARG_IDX_MAX || op == ReductionOperation::ARG_IDX_MIN);
        if (!is_arg_min_max)
        {
            ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
            ARM_COMPUTE_RETURN_ERROR_ON(input->num_channels() != output->num_channels());
        }
        else
        {
            ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::U32, DataType::S32);
        }

        const TensorShape output_shape =
            arm_compute::misc::shape_calculator::compute_reduced_shape(input->tensor_shape(), axis);
        const TensorInfo tensor_info_reshaped = input->clone()->set_tensor_shape(output_shape);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(output, &tensor_info_reshaped);
    }

    return Status{};
}

NEReductionOperationKernel::NEReductionOperationKernel()
    : _func(nullptr), _input(nullptr), _output(nullptr), _reduction_axis(0), _op(ReductionOperation::SUM_SQUARE)
{
}

void NEReductionOperationKernel::configure(const ITensor     *input,
                                           ITensor           *output,
                                           unsigned int       axis,
                                           ReductionOperation op)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);

    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), output->info(), axis, op));

    _input          = input;
    _output         = output;
    _op             = op;
    _reduction_axis = axis;

    // Configure kernel window
    Window win = calculate_max_window(*input->info(), Steps());
    INEKernel::configure(win);

    // Calculate output shape and set if empty
    const TensorShape output_shape =
        arm_compute::misc::shape_calculator::compute_reduced_shape(input->info()->tensor_shape(), axis);
    // Output auto initialization if not yet initialized
    const bool is_arg_min_max   = (op == ReductionOperation::ARG_IDX_MIN || op == ReductionOperation::ARG_IDX_MAX);
    DataType   output_data_type = is_arg_min_max ? DataType::S32 : input->info()->data_type();
    auto_init_if_empty(*output->info(), input->info()
                                            ->clone()
                                            ->set_tensor_shape(output_shape)
                                            .set_data_type(output_data_type)
                                            .reset_padding()
                                            .set_is_resizable(true));
    // Determine the reduction function
    NEReductionOperationKernel::reduce_op();
}

Status NEReductionOperationKernel::validate(const ITensorInfo *input,
                                            const ITensorInfo *output,
                                            unsigned int       axis,
                                            ReductionOperation op)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output, axis, op));

    return Status{};
}

void NEReductionOperationKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);

    (*_func)(window, _input, _output, _op);
}
} // namespace arm_compute
