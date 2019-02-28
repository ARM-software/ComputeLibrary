/*
 * Copyright (c) 2016-2018 ARM Limited.
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

#include <cstdarg>
#include <cstdio>
#include <iostream>
#include <stdexcept>

using namespace arm_compute;

Status arm_compute::create_error_va_list(ErrorCode error_code, const char *function, const char *file, const int line, const char *msg, va_list args)
{
    char out[512];
    int  offset = snprintf(out, sizeof(out), "in %s %s:%d: ", function, file, line);
    vsnprintf(out + offset, sizeof(out) - offset, msg, args);

    return Status(error_code, std::string(out));
}

Status arm_compute::create_error(ErrorCode error_code, const char *function, const char *file, const int line, const char *msg, ...)
{
    va_list args;
    va_start(args, msg);
    auto err = create_error_va_list(error_code, function, file, line, msg, args);
    va_end(args);
    return err;
}

void arm_compute::error(const char *function, const char *file, const int line, const char *msg, ...)
{
    va_list args;
    va_start(args, msg);
    auto err = create_error_va_list(ErrorCode::RUNTIME_ERROR, function, file, line, msg, args);
    va_end(args);
    ARM_COMPUTE_THROW(std::runtime_error(err.error_description()));
}
void Status::internal_throw_on_error() const
{
    ARM_COMPUTE_THROW(std::runtime_error(_error_description));
}
