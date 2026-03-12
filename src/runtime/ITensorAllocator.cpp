/*
 * Copyright (c) 2016-2021, 2026 Arm Limited.
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
#include "arm_compute/runtime/ITensorAllocator.h"

#include "arm_compute/core/Coordinates.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"

using namespace arm_compute;

void ITensorAllocator::init(const TensorInfo &input, size_t alignment)
{
    _info_owned    = input;
    _info_external = nullptr;
    _alignment     = alignment;
    _owns_info     = true;
    _is_imported   = false;
}

void ITensorAllocator::soft_init(TensorInfo &input, size_t alignment)
{
    _info_external = &input;
    _alignment     = alignment;
    _owns_info     = false;
    _is_imported   = false;
}

TensorInfo &ITensorAllocator::info()
{
    return (_info_external != nullptr) ? *_info_external : _info_owned;
}

const TensorInfo &ITensorAllocator::info() const
{
    return (_info_external != nullptr) ? *_info_external : _info_owned;
}

size_t ITensorAllocator::alignment() const
{
    return _alignment;
}

bool ITensorAllocator::owns_info() const
{
    return _owns_info;
}

bool ITensorAllocator::is_imported() const
{
    return _is_imported;
}

void ITensorAllocator::set_imported(bool imported)
{
    _is_imported = imported;
}

void ITensorAllocator::set_resizable_if_info_owned(bool is_resizable)
{
    if (_owns_info)
    {
        info().set_is_resizable(is_resizable);
    }
}
