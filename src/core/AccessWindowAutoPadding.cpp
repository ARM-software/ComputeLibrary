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
#include "arm_compute/core/AccessWindowAutoPadding.h"

#include "arm_compute/core/ITensorInfo.h"
#include "arm_compute/core/Window.h"

using namespace arm_compute;

AccessWindowAutoPadding::AccessWindowAutoPadding(ITensorInfo *info)
    : _info(info)
{
}

ValidRegion AccessWindowAutoPadding::compute_valid_region(const Window &window, ValidRegion input_valid_region, bool border_undefined, BorderSize border_size) const
{
    ARM_COMPUTE_UNUSED(window);
    ARM_COMPUTE_UNUSED(input_valid_region);
    ARM_COMPUTE_UNUSED(border_undefined);
    ARM_COMPUTE_UNUSED(border_size);

    return compute_valid_region();
}

ValidRegion AccessWindowAutoPadding::compute_valid_region() const
{
    if(_info == nullptr)
    {
        return ValidRegion();
    }

    return ValidRegion(Coordinates(), _info->tensor_shape());
}

void AccessWindowAutoPadding::set_valid_region()
{
    if(_info == nullptr)
    {
        return;
    }

    _info->set_valid_region(compute_valid_region());
}

bool AccessWindowAutoPadding::update_window_if_needed(Window &window) const
{
    ARM_COMPUTE_UNUSED(window);

    return false;
}

bool AccessWindowAutoPadding::update_padding_if_needed(const Window &window) const
{
    ARM_COMPUTE_UNUSED(window);

    // Only update the padding if the tensor allows it
    if(_info == nullptr || !_info->is_resizable())
    {
        return false;
    }

    // Update strides in tensor info
    return _info->auto_padding();
}
