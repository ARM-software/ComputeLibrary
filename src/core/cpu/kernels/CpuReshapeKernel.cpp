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
#include "src/core/cpu/kernels/CpuReshapeKernel.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/IAccessWindow.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"
#include "src/core/AccessWindowStatic.h"
#include "src/core/CPP/Validate.h"
#include "src/core/NEON/INEKernel.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"

#include <cstdint>

/** [NEReshapeLayerKernel Kernel] **/
namespace arm_compute
{
namespace cpu
{
namespace kernels
{
namespace
{
Status validate_arguments(const ITensorInfo *src, const ITensorInfo *dst)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(src, dst);
    // Note: ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(src) is not needed here as this kernel doesn't use Neon FP16 instructions.
    ARM_COMPUTE_RETURN_ERROR_ON(src->data_type() == DataType::UNKNOWN);

    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src, dst);
    ARM_COMPUTE_RETURN_ERROR_ON(src->tensor_shape().total_size() != dst->tensor_shape().total_size());
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_QUANTIZATION_INFO(src, dst);

    return Status{};
}

template <typename T>
inline void reshape_tensor(const Window &window, const ITensor *src, ITensor *dst)
{
    const TensorShape &src_shape = src->info()->tensor_shape();
    const TensorShape &dst_shape = dst->info()->tensor_shape();
    Coordinates        dst_coord{};

    Iterator src_it(src, window);

    execute_window_loop(window, [&](const Coordinates & id)
    {
        dst_coord                                              = index2coords(dst_shape, coords2index(src_shape, id));
        *reinterpret_cast<T *>(dst->ptr_to_element(dst_coord)) = *reinterpret_cast<T *>(src_it.ptr());
    },
    src_it);
}
} // namespace

void CpuReshapeKernel::configure(const ITensorInfo *src, ITensorInfo *dst)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src, dst);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(src, dst));

    // Configure kernel window
    Window win = calculate_max_window(*src);

    // Set the destination valid region
    dst->set_valid_region(ValidRegion(Coordinates(), dst->tensor_shape()));

    ICpuKernel::configure(win);
}

Status CpuReshapeKernel::validate(const ITensorInfo *src, const ITensorInfo *dst)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(src, dst));

    return Status{};
}

void CpuReshapeKernel::run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICpuKernel::window(), window);

    const auto src = tensors.get_const_tensor(TensorType::ACL_SRC);
    auto       dst = tensors.get_tensor(TensorType::ACL_DST);

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

const char *CpuReshapeKernel::name() const
{
    return "CpuReshapeKernel";
}
} // namespace kernels
} // namespace cpu
} // namespace arm_compute
/** [NEReshapeLayerKernel Kernel] **/
