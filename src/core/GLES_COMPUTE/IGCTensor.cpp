/*
 * Copyright (c) 2017-2018 Arm Limited.
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
#include "arm_compute/core/GLES_COMPUTE/IGCTensor.h"

using namespace arm_compute;

IGCTensor::IGCTensor()
    : _mapping(nullptr), _needs_shifting(false)
{
}

void IGCTensor::map(bool blocking)
{
    _mapping = do_map(blocking);
}

void IGCTensor::unmap()
{
    do_unmap();
    _mapping = nullptr;
}

void IGCTensor::clear()
{
    this->map();
    std::memset(static_cast<void *>(_mapping), 0, this->info()->total_size());
    this->unmap();
}

uint8_t *IGCTensor::buffer() const
{
    return _mapping;
}

bool IGCTensor::needs_shifting() const
{
    return _needs_shifting;
}

void IGCTensor::set_needs_shifting(bool needs_shifting)
{
    _needs_shifting = needs_shifting;
}
