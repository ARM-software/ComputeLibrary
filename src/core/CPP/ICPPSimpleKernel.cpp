/*
 * Copyright (c) 2016-2018 Arm Limited.
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
#include "arm_compute/core/CPP/ICPPSimpleKernel.h"

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/IAccessWindow.h"
#include "arm_compute/core/ITensor.h"

namespace arm_compute
{
namespace
{
std::pair<Status, Window> validate_and_configure_window(ITensorInfo *input, ITensorInfo *output, unsigned int num_elems_processed_per_iteration,
                                                        bool border_undefined, const arm_compute::BorderSize &border_size)
{
    // Configure kernel window
    Window                 win = calculate_max_window(*input, Steps(num_elems_processed_per_iteration), border_undefined, border_size);
    AccessWindowHorizontal input_access(input, 0, num_elems_processed_per_iteration);
    AccessWindowHorizontal output_access(output, 0, num_elems_processed_per_iteration);

    bool window_changed = update_window_and_padding(win, input_access, output_access);

    output_access.set_valid_region(win, input->valid_region(), border_undefined, border_size);

    Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
    return std::make_pair(err, win);
}
} // namespace

ICPPSimpleKernel::ICPPSimpleKernel()
    : _input{ nullptr }, _output{ nullptr }
{
}

void ICPPSimpleKernel::configure(const ITensor *input, ITensor *output, unsigned int num_elems_processed_per_iteration, bool border_undefined, const BorderSize &border_size)
{
    _input  = input;
    _output = output;

    // Configure kernel window
    auto win_config = validate_and_configure_window(input->info(), output->info(), num_elems_processed_per_iteration, border_undefined, border_size);
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    ICPPKernel::configure(win_config.second);
}

Status ICPPSimpleKernel::validate(const ITensorInfo *input, const ITensorInfo *output, unsigned int num_elems_processed_per_iteration,
                                  bool border_undefined, const arm_compute::BorderSize &border_size)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(input->clone().get(), output->clone().get(), num_elems_processed_per_iteration, border_undefined, border_size).first);
    return Status{};
}

} // namespace arm_compute