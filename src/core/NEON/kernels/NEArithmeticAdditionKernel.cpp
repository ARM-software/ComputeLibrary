/*
 * Copyright (c) 2016-2021 Arm Limited.
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
#include "src/core/NEON/kernels/NEArithmeticAdditionKernel.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/Validate.h"
#include "src/core/CPP/Validate.h"
#include "src/core/NEON/kernels/arithmetic_addition/impl/NEON/list.h"
#include "src/core/NEON/kernels/arithmetic_addition/impl/SVE/list.h"
#include "src/core/NEON/wrapper/wrapper.h"
#include "src/core/common/Registrars.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"

#include <map>
#include <string>

namespace arm_compute
{
namespace
{
struct ArithmeticAdditionSelectorData
{
    DataType dt1;
    DataType dt2;
    DataType dt3;
};

using ArithmeticAdditionSelectorPtr = std::add_pointer<bool(const ArithmeticAdditionSelectorData &data)>::type;

struct ArithmeticAdditionKernel
{
    const char                                             *name;
    const ArithmeticAdditionSelectorPtr                     is_selected;
    NEArithmeticAdditionKernel::ArithmeticAdditionKernelPtr ukernel;
};

static const ArithmeticAdditionKernel available_kernels[] =
{
#if defined(__ARM_FEATURE_SVE)
    {
        "arithmetic_addition_same_sve",
        [](const ArithmeticAdditionSelectorData & data) { return ((data.dt1 == data.dt2) && (data.dt1 == DataType::F32)); },
        REGISTER_FP32_SVE(arm_compute::cpu::arithmetic_addition_same_sve<float>)
    },
    {
        "arithmetic_addition_same_sve",
        [](const ArithmeticAdditionSelectorData & data) { return ((data.dt1 == data.dt2) && (data.dt1 == DataType::F16)); },
        REGISTER_FP16_SVE(arm_compute::cpu::arithmetic_addition_same_sve<float16_t>)
    },
    {
        "arithmetic_addition_same_sve",
        [](const ArithmeticAdditionSelectorData & data) { return ((data.dt1 == data.dt2) && (data.dt1 == data.dt3) && (data.dt1 == DataType::U8)); },
        REGISTER_INTEGER_SVE(arm_compute::cpu::arithmetic_addition_same_sve<uint8_t>)
    },
    {
        "arithmetic_addition_same_sve",
        [](const ArithmeticAdditionSelectorData & data) { return ((data.dt1 == data.dt2) && (data.dt1 == data.dt3) && (data.dt1 == DataType::S16)); },
        REGISTER_INTEGER_SVE(arm_compute::cpu::arithmetic_addition_same_sve<int16_t>)
    },
    {
        "arithmetic_addition_same_sve",
        [](const ArithmeticAdditionSelectorData & data) { return ((data.dt1 == data.dt2) && (data.dt1 == data.dt3) && (data.dt1 == DataType::S32)); },
        REGISTER_INTEGER_SVE(arm_compute::cpu::arithmetic_addition_same_sve<int32_t>)
    },
    {
        "arithmetic_addition_U8_S16_S16_sve",
        [](const ArithmeticAdditionSelectorData & data) { return ((data.dt1 == DataType::U8) && (data.dt2 == DataType::S16)); },
        REGISTER_INTEGER_SVE(arm_compute::cpu::arithmetic_addition_U8_S16_S16_sve)
    },
    {
        "arithmetic_addition_S16_U8_S16_sve",
        [](const ArithmeticAdditionSelectorData & data) { return ((data.dt1 == DataType::S16) && (data.dt2 == DataType::U8)); },
        REGISTER_INTEGER_SVE(arm_compute::cpu::arithmetic_addition_S16_U8_S16_sve)
    },
    {
        "arithmetic_addition_U8_U8_S16_sve",
        [](const ArithmeticAdditionSelectorData & data) { return ((data.dt1 == data.dt2) && (data.dt3 == DataType::S16)); },
        REGISTER_INTEGER_SVE(arm_compute::cpu::arithmetic_addition_U8_U8_S16_sve)
    },
#else /* !defined(__ARM_FEATURE_SVE) */
    {
        "arithmetic_addition_same_neon",
        [](const ArithmeticAdditionSelectorData & data) { return ((data.dt1 == data.dt2) && (data.dt1 == DataType::F32)); },
        REGISTER_FP32_NEON(arm_compute::cpu::arithmetic_addition_same_neon<float>)
    },
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    {
        "arithmetic_addition_same_neon",
        [](const ArithmeticAdditionSelectorData & data) { return ((data.dt1 == data.dt2) && (data.dt1 == DataType::F16)); },
        REGISTER_FP16_NEON(arm_compute::cpu::arithmetic_addition_same_neon<float16_t>)
    },
#endif /* defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) */
    {
        "arithmetic_addition_same_neon",
        [](const ArithmeticAdditionSelectorData & data) { return ((data.dt1 == data.dt2) && (data.dt1 == data.dt3) && (data.dt1 == DataType::U8)); },
        REGISTER_INTEGER_NEON(arm_compute::cpu::arithmetic_addition_same_neon<uint8_t>)
    },
    {
        "arithmetic_addition_same_neon",
        [](const ArithmeticAdditionSelectorData & data) { return ((data.dt1 == data.dt2) && (data.dt1 == data.dt3) && (data.dt1 == DataType::S16)); },
        REGISTER_INTEGER_NEON(arm_compute::cpu::arithmetic_addition_same_neon<int16_t>)
    },
    {
        "arithmetic_addition_same_neon",
        [](const ArithmeticAdditionSelectorData & data) { return ((data.dt1 == data.dt2) && (data.dt1 == data.dt3) && (data.dt1 == DataType::S32)); },
        REGISTER_INTEGER_NEON(arm_compute::cpu::arithmetic_addition_same_neon<int32_t>)
    },
    {
        "arithmetic_addition_U8_S16_S16_neon",
        [](const ArithmeticAdditionSelectorData & data) { return ((data.dt1 == DataType::U8) && (data.dt2 == DataType::S16)); },
        REGISTER_INTEGER_NEON(arm_compute::cpu::arithmetic_addition_U8_S16_S16_neon)
    },
    {
        "arithmetic_addition_S16_U8_S16_neon",
        [](const ArithmeticAdditionSelectorData & data) { return ((data.dt1 == DataType::S16) && (data.dt2 == DataType::U8)); },
        REGISTER_INTEGER_NEON(arm_compute::cpu::arithmetic_addition_S16_U8_S16_neon)
    },
    {
        "arithmetic_addition_U8_U8_S16_neon",
        [](const ArithmeticAdditionSelectorData & data) { return ((data.dt1 == data.dt2) && (data.dt3 == DataType::S16)); },
        REGISTER_INTEGER_NEON(arm_compute::cpu::arithmetic_addition_U8_U8_S16_neon)
    },
#endif /* defined(__ARM_FEATURE_SVE) */

#if defined(__ARM_FEATURE_SVE2)
    {
        "arithmetic_addition_qasymm8_sve",
        [](const ArithmeticAdditionSelectorData & data) { return ((data.dt1 == data.dt2) && (data.dt1 == DataType::QASYMM8)); },
        REGISTER_QASYMM8_SVE(arm_compute::cpu::arithmetic_addition_qasymm8_sve)
    },
    {
        "arithmetic_addition_qasymm8_signed_sve",
        [](const ArithmeticAdditionSelectorData & data) { return ((data.dt1 == data.dt2) && (data.dt1 == DataType::QASYMM8_SIGNED)); },
        REGISTER_QASYMM8_SIGNED_SVE(arm_compute::cpu::arithmetic_addition_qasymm8_signed_sve)
    },
    {
        "arithmetic_addition_qsymm16_sve",
        [](const ArithmeticAdditionSelectorData & data) { return ((data.dt1 == data.dt2) && (data.dt1 == DataType::QSYMM16)); },
        REGISTER_QSYMM16_SVE(arm_compute::cpu::arithmetic_addition_qsymm16_sve)
    },
#else  /* !defined(__ARM_FEATURE_SVE2) */
    {
        "arithmetic_addition_qasymm8_neon",
        [](const ArithmeticAdditionSelectorData & data) { return ((data.dt1 == data.dt2) && (data.dt1 == DataType::QASYMM8)); },
        REGISTER_QASYMM8_NEON(arm_compute::cpu::arithmetic_addition_qasymm8_neon)
    },
    {
        "arithmetic_addition_qasymm8_signed_neon",
        [](const ArithmeticAdditionSelectorData & data) { return ((data.dt1 == data.dt2) && (data.dt1 == DataType::QASYMM8_SIGNED)); },
        REGISTER_QASYMM8_SIGNED_NEON(arm_compute::cpu::arithmetic_addition_qasymm8_signed_neon)
    },
    {
        "arithmetic_addition_qsymm16_neon",
        [](const ArithmeticAdditionSelectorData & data) { return ((data.dt1 == data.dt2) && (data.dt1 == DataType::QSYMM16)); },
        REGISTER_QSYMM16_NEON(arm_compute::cpu::arithmetic_addition_qsymm16_neon)
    },
#endif /* defined(__ARM_FEATURE_SVE2) */

};

const ArithmeticAdditionKernel *get_implementation(DataType dt1, DataType dt2, DataType dt3)
{
    for(const auto &uk : available_kernels)
    {
        if(uk.is_selected({ dt1, dt2, dt3 }))
        {
            return &uk;
        }
    }
    return nullptr;
}

Status validate_arguments(const ITensorInfo &input1, const ITensorInfo &input2, const ITensorInfo &output, ConvertPolicy policy)
{
    ARM_COMPUTE_UNUSED(policy);

    ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(&input1);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(&input1, 1, DataType::U8, DataType::QASYMM8, DataType::QASYMM8_SIGNED,
                                                         DataType::S16, DataType::QSYMM16, DataType::F16,
                                                         DataType::S32, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(&input2, 1, DataType::U8, DataType::QASYMM8, DataType::QASYMM8_SIGNED,
                                                         DataType::S16, DataType::QSYMM16, DataType::F16,
                                                         DataType::S32, DataType::F32);

    const TensorShape out_shape = TensorShape::broadcast_shape(input1.tensor_shape(), input2.tensor_shape());

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(out_shape.total_size() == 0, "Inputs are not broadcast compatible");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG((input1.tensor_shape().x() != input2.tensor_shape().x()) && ((input1.data_type() != input2.data_type()) || (input1.data_type() != output.data_type())
                                                                                                 || (input2.data_type() != output.data_type())),
                                    "Broadcasting across width is supported on configurations where all tensors have the same data type");

    // Validate in case of configured output
    if(output.total_size() > 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(
            !(input1.data_type() == DataType::U8 && input2.data_type() == DataType::U8 && output.data_type() == DataType::U8)
            && !(input1.data_type() == DataType::U8 && input2.data_type() == DataType::U8 && output.data_type() == DataType::S16)
            && !(input1.data_type() == DataType::U8 && input2.data_type() == DataType::S16 && output.data_type() == DataType::S16)
            && !(input1.data_type() == DataType::S16 && input2.data_type() == DataType::U8 && output.data_type() == DataType::S16)
            && !(input1.data_type() == DataType::S16 && input2.data_type() == DataType::S16 && output.data_type() == DataType::S16)
            && !(input1.data_type() == DataType::S32 && input2.data_type() == DataType::S32 && output.data_type() == DataType::S32)
            && !(input1.data_type() == DataType::F32 && input2.data_type() == DataType::F32 && output.data_type() == DataType::F32)
            && !(input1.data_type() == DataType::F16 && input2.data_type() == DataType::F16 && output.data_type() == DataType::F16)
            && !(input1.data_type() == DataType::QASYMM8 && input2.data_type() == DataType::QASYMM8 && output.data_type() == DataType::QASYMM8)
            && !(input1.data_type() == DataType::QASYMM8_SIGNED && input2.data_type() == DataType::QASYMM8_SIGNED && output.data_type() == DataType::QASYMM8_SIGNED)
            && !(input1.data_type() == DataType::QSYMM16 && input2.data_type() == DataType::QSYMM16 && output.data_type() == DataType::QSYMM16),
            "You called addition with the wrong image formats");

        ARM_COMPUTE_RETURN_ERROR_ON_MSG(detail::have_different_dimensions(out_shape, output.tensor_shape(), 0),
                                        "Wrong shape for output");
    }

    return Status{};
}

std::pair<Status, Window> validate_and_configure_window(const ITensorInfo &input1, const ITensorInfo &input2, ITensorInfo &output)
{
    const std::pair<TensorShape, ValidRegion> broadcast_pair = ITensorInfo::broadcast_shape_and_valid_region(input1, input2);
    const TensorShape &out_shape    = broadcast_pair.first;
    const ValidRegion &valid_region = broadcast_pair.second;

    // Auto initialize output if not initialized
    {
        set_shape_if_empty(output, out_shape);

        if(input1.data_type() == DataType::S16 || input2.data_type() == DataType::S16)
        {
            set_format_if_unknown(output, Format::S16);
        }
        if(input1.data_type() == DataType::S32 || input2.data_type() == DataType::S32)
        {
            set_format_if_unknown(output, Format::S32);
        }
        else if(input1.data_type() == DataType::F16 || input2.data_type() == DataType::F16)
        {
            set_format_if_unknown(output, Format::F16);
        }
        else if(input1.data_type() == DataType::F32 || input2.data_type() == DataType::F32)
        {
            set_format_if_unknown(output, Format::F32);
        }
        else if(input1.data_type() == DataType::QASYMM8 || input2.data_type() == DataType::QASYMM8)
        {
            set_data_type_if_unknown(output, DataType::QASYMM8);
        }
        else if(input1.data_type() == DataType::QASYMM8_SIGNED || input2.data_type() == DataType::QASYMM8_SIGNED)
        {
            set_data_type_if_unknown(output, DataType::QASYMM8_SIGNED);
        }
        else if(input1.data_type() == DataType::QSYMM16 || input2.data_type() == DataType::QSYMM16)
        {
            set_data_type_if_unknown(output, DataType::QSYMM16);
        }
    }

    Window win = calculate_max_window(valid_region, Steps());

    // NEArithmeticAdditionKernel doesn't need padding so update_window_and_padding() can be skipped
    Coordinates coord;
    coord.set_num_dimensions(output.num_dimensions());
    output.set_valid_region(valid_region);
    return std::make_pair(Status{}, win);
}
} // namespace

NEArithmeticAdditionKernel::NEArithmeticAdditionKernel()
    : _func(nullptr), _policy()
{
}

void NEArithmeticAdditionKernel::configure(const ITensorInfo *input1, const ITensorInfo *input2, ITensorInfo *output, ConvertPolicy policy)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input1, input2, output);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(*input1, *input2, *output, policy));

    _policy = policy;
    _func   = get_implementation(input1->data_type(), input2->data_type(), output->data_type())->ukernel;

    // Configure kernel window
    auto win_config = validate_and_configure_window(*input1, *input2, *output);
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    INEKernel::configure(win_config.second);
}

Status NEArithmeticAdditionKernel::validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, ConvertPolicy policy)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input1, input2, output);

    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(*input1, *input2, *output, policy));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(*input1->clone(), *input2->clone(), *output->clone()).first);

    return Status{};
}

void NEArithmeticAdditionKernel::run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);
    // Dispatch kernel
    (*_func)(tensors.get_const_tensor(TensorType::ACL_SRC_0),
             tensors.get_const_tensor(TensorType::ACL_SRC_1),
             tensors.get_tensor(TensorType::ACL_DST),
             _policy,
             window);
}
} // namespace arm_compute
