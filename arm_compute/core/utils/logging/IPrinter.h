/*
 * Copyright (c) 2017-2019 ARM Limited.
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
#ifndef __ARM_COMPUTE_LOGGING_PRINTER_H__
#define __ARM_COMPUTE_LOGGING_PRINTER_H__

#include "support/Mutex.h"

namespace arm_compute
{
namespace logging
{
/** Base printer class to be inherited by other printer classes */
class Printer
{
public:
    /** Default Constructor */
    Printer() noexcept
        : _mtx()
    {
    }
    /** Prevent instances of this class from being copied */
    Printer(const Printer &) = delete;
    /** Prevent instances of this class from being copied */
    Printer &operator=(const Printer &) = delete;
    /** Prevent instances of this class from being moved */
    Printer(Printer &&) = delete;
    /** Prevent instances of this class from being moved */
    Printer &operator=(Printer &&) = delete;
    /** Defaults Destructor */
    virtual ~Printer() = default;
    /** Print message
     *
     * @param[in] msg Message to print
     */
    inline void print(const std::string &msg)
    {
        std::lock_guard<arm_compute::Mutex> lock(_mtx);
        print_internal(msg);
    }

private:
    /** Interface to be implemented by the child to print a message
     *
     * @param[in] msg Message to print
     */
    virtual void print_internal(const std::string &msg) = 0;

private:
    arm_compute::Mutex _mtx;
};
} // namespace logging
} // namespace arm_compute
#endif /* __ARM_COMPUTE_LOGGING_PRINTER_H__ */
