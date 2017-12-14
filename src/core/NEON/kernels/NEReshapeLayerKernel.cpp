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
#include "arm_compute/core/NEON/kernels/NEReshapeLayerKernel.h"

#include "arm_compute/core/AccessWindowStatic.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/IAccessWindow.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/NEON/INEKernel.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Validate.h"

#include <arm_neon.h>
#include <cstdint>

using namespace arm_compute;

namespace
{
template <typename T>
inline void reshape_tensor(const Window &window, const ITensor *input, ITensor *output)
{
    const TensorShape &input_shape  = input->info()->tensor_shape();
    const TensorShape &output_shape = output->info()->tensor_shape();
    Coordinates        output_coord{};

    window.collapse_if_possible(window, 3);
    Iterator in(input, window);

    execute_window_loop(window, [&](const Coordinates & id)
    {
        output_coord                                                 = index2coords(output_shape, coords2index(input_shape, id));
        *reinterpret_cast<T *>(output->ptr_to_element(output_coord)) = *reinterpret_cast<T *>(in.ptr());
    },
    in);
}
} // namespace

void NEReshapeLayerKernel::configure(const ITensor *input, ITensor *output)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8, DataType::S8, DataType::QASYMM8, DataType::QS8, DataType::U16, DataType::S16, DataType::QS16,
                                                  DataType::U32, DataType::S32, DataType::F16, DataType::F32);
    ARM_COMPUTE_ERROR_ON_NULLPTR(output);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_FIXED_POINT(input, output);
    ARM_COMPUTE_ERROR_ON(input->info()->tensor_shape().total_size() != output->info()->tensor_shape().total_size());

    _input  = input;
    _output = output;

    constexpr unsigned int num_elems_processed_per_iteration = 1;

    // Configure kernel window
    Window win = calculate_max_window(*input->info(), Steps(num_elems_processed_per_iteration));

    AccessWindowHorizontal input_access(input->info(), 0, num_elems_processed_per_iteration);
    AccessWindowStatic     output_access(output->info(), 0, 0, output->info()->tensor_shape().x(), output->info()->tensor_shape().y());
    update_window_and_padding(win, input_access, output_access);

    output_access.set_valid_region(win, ValidRegion(Coordinates(), output->info()->tensor_shape()));

    INEKernel::configure(win);
}

void NEReshapeLayerKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INESimpleKernel::window(), window);

    switch(_input->info()->data_type())
    {
        case DataType::U8:
        case DataType::S8:
        case DataType::QASYMM8:
        case DataType::QS8:
            reshape_tensor<uint8_t>(window, _input, _output);
            break;
        case DataType::U16:
        case DataType::S16:
        case DataType::QS16:
        case DataType::F16:
            reshape_tensor<uint16_t>(window, _input, _output);
            break;
        case DataType::U32:
        case DataType::S32:
        case DataType::F32:
            reshape_tensor<uint32_t>(window, _input, _output);
            break;
        default:
            ARM_COMPUTE_ERROR("Unsupported data type!");
    }
}
