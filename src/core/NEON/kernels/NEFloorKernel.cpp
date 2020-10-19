/*
 * Copyright (c) 2017-2020 Arm Limited.
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
#include "arm_compute/core/NEON/kernels/NEFloorKernel.h"

#include "arm_compute/core/Coordinates.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/NEON/INEKernel.h"
#include "arm_compute/core/Validate.h"
#include "src/core/CPP/Validate.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"

#include "src/core/NEON/kernels/floor/impl/list.h"
#include "src/core/common/Registrars.h"

namespace arm_compute
{
namespace
{
struct FloorSelectorData
{
    DataType dt;
};
using FloorSelectorPtr = std::add_pointer<bool(const FloorSelectorData &data)>::type;
using FloorUKernelPtr  = std::add_pointer<void(const void *, void *, int)>::type;

struct FloorKernel
{
    const char            *name;
    const FloorSelectorPtr is_selected;
    FloorUKernelPtr        ukernel;
};

static const FloorKernel available_kernels[] =
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

const FloorKernel *get_implementation(const FloorSelectorData &data)
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

Status validate_arguments(const ITensorInfo *input, const ITensorInfo *output)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output);

    const auto *uk = get_implementation(FloorSelectorData{ input->data_type() });
    ARM_COMPUTE_RETURN_ERROR_ON(uk == nullptr || uk->ukernel == nullptr);

    // Validate in case of configured output
    if(output->total_size() > 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(input, output);
    }

    return Status{};
}
} // namespace

void NEFloorKernel::configure(const ITensor *input, ITensor *output)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);

    // Auto initialize output
    auto_init_if_empty(*output->info(), input->info()->tensor_shape(), 1, input->info()->data_type());

    // Validate
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), output->info()));

    _input  = input;
    _output = output;

    // Configure kernel window
    Window win = calculate_max_window(*input->info(), Steps());

    Coordinates coord;
    coord.set_num_dimensions(output->info()->num_dimensions());
    output->info()->set_valid_region(ValidRegion(coord, output->info()->tensor_shape()));

    INEKernel::configure(win);
}

Status NEFloorKernel::validate(const ITensorInfo *input, const ITensorInfo *output)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output));
    return Status{};
}

void NEFloorKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);

    Window win{ window };
    win.set(Window::DimX, Window::Dimension(0, 1, 1));

    const auto  len = static_cast<int>(window.x().end()) - static_cast<int>(window.x().start());
    const auto *uk  = get_implementation(FloorSelectorData{ _input->info()->data_type() });

    Iterator input(_input, win);
    Iterator output(_output, win);

    execute_window_loop(win, [&](const Coordinates &)
    {
        uk->ukernel(input.ptr(), output.ptr(), len);
    },
    input, output);
}
} // namespace arm_compute
