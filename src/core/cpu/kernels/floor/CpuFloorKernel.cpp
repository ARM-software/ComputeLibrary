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
#include "src/core/cpu/kernels/CpuFloorKernel.h"

#include "arm_compute/core/Coordinates.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/Validate.h"
#include "src/core/CPP/Validate.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"

#include "src/core/common/Registrars.h"
#include "src/core/cpu/kernels/floor/impl/list.h"

namespace arm_compute
{
namespace cpu
{
namespace kernels
{
namespace
{
struct FloorSelectorData
{
    DataType dt;
};

using FloorSelectorPtr = std::add_pointer<bool(const FloorSelectorData &data)>::type;
using FloorUKernelPtr  = std::add_pointer<void(const void *, void *, int)>::type;

struct FloorUKernel
{
    const char            *name;
    const FloorSelectorPtr is_selected;
    FloorUKernelPtr        func;
};

static const FloorUKernel available_kernels[] =
{
    {
        "fp16_neon_floor",
        [](const FloorSelectorData & data) { return data.dt == DataType::F16; },
        REGISTER_FP16_NEON(arm_compute::cpu::fp16_neon_floor)
    },
    {
        "f32_neon_floor",
        [](const FloorSelectorData & data) { return data.dt == DataType::F32; },
        REGISTER_FP32_NEON(arm_compute::cpu::fp32_neon_floor)
    },
};

/** Micro-kernel selector
 *
 * @param[in] data Selection data passed to help pick the appropriate micro-kernel
 *
 * @return A matching micro-kernel else nullptr
 */
const FloorUKernel *get_implementation(const FloorSelectorData &data)
{
    for(const auto &uk : available_kernels)
    {
        if(uk.is_selected(data))
        {
            return &uk;
        }
    }
    return nullptr;
}

Status validate_arguments(const ITensorInfo *src, const ITensorInfo *dst)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(src, dst);

    const auto *uk = get_implementation(FloorSelectorData{ src->data_type() });
    ARM_COMPUTE_RETURN_ERROR_ON(uk == nullptr || uk->func == nullptr);

    // Validate in case of configured output
    if(dst->total_size() > 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src, dst);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(src, dst);
    }

    return Status{};
}
} // namespace

void CpuFloorKernel::configure(const ITensorInfo *src, ITensorInfo *dst)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src, dst);

    // Auto initialize output
    auto_init_if_empty(*dst, src->tensor_shape(), 1, src->data_type());

    // Validate
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(src, dst));

    // Configure kernel window
    const Window win = calculate_max_window(*src, Steps());

    Coordinates coord;
    coord.set_num_dimensions(dst->num_dimensions());
    dst->set_valid_region(ValidRegion(coord, dst->tensor_shape()));

    ICPPKernel::configure(win);
}

Window CpuFloorKernel::infer_window(const ITensorInfo *src, const ITensorInfo *dst)
{
    ARM_COMPUTE_UNUSED(dst);
    ARM_COMPUTE_ERROR_ON(!bool(validate_arguments(src, dst)));

    Window win;
    win.use_tensor_dimensions(src->tensor_shape());
    return win;
}

Status CpuFloorKernel::validate(const ITensorInfo *input, const ITensorInfo *output)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output));
    return Status{};
}

void CpuFloorKernel::run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);

    ARM_COMPUTE_ERROR_ON(tensors.empty());

    const ITensor *src = tensors.get_const_tensor(TensorType::ACL_SRC);
    ITensor       *dst = tensors.get_tensor(TensorType::ACL_DST);

    const auto  len     = static_cast<int>(window.x().end()) - static_cast<int>(window.x().start());
    const auto *ukernel = get_implementation(FloorSelectorData{ src->info()->data_type() });

    Window win{ window };
    win.set(Window::DimX, Window::Dimension(0, 1, 1));

    Iterator src_it(src, win);
    Iterator dst_it(dst, win);

    execute_window_loop(win, [&](const Coordinates &)
    {
        ukernel->func(src_it.ptr(), dst_it.ptr(), len);
    },
    src_it, dst_it);
}

const char *CpuFloorKernel::name() const
{
    return "CpuFloorKernel";
}
} // namespace kernels
} // namespace cpu
} // namespace arm_compute
