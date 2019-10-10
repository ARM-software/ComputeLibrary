/*
 * Copyright (c) 2016-2019 ARM Limited.
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
#include "arm_compute/core/Error.h"

#include <array>
#include <cstdarg>
#include <cstdio>
#include <iostream>
#include <stdexcept>

using namespace arm_compute;

Status arm_compute::create_error(ErrorCode error_code, std::string msg)
{
    return Status(error_code, msg);
}

Status arm_compute::create_error_msg(ErrorCode error_code, const char *func, const char *file, int line, const char *msg)
{
    std::array<char, 512> out{ 0 };
    snprintf(out.data(), out.size(), "in %s %s:%d: %s", func, file, line, msg);
    return Status(error_code, std::string(out.data()));
}

void arm_compute::throw_error(Status err)
{
    ARM_COMPUTE_THROW(std::runtime_error(err.error_description()));
}
void Status::internal_throw_on_error() const
{
    ARM_COMPUTE_THROW(std::runtime_error(_error_description));
}
