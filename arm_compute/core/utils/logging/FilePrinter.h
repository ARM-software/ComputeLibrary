/*
 * Copyright (c) 2017-2019 Arm Limited.
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
#ifndef ARM_COMPUTE_LOGGING_FILE_PRINTER_H
#define ARM_COMPUTE_LOGGING_FILE_PRINTER_H

#include "arm_compute/core/utils/logging/IPrinter.h"

#include "arm_compute/core/utils/io/FileHandler.h"

namespace arm_compute
{
namespace logging
{
/** File Printer */
class FilePrinter final : public Printer
{
public:
    /** Default Constructor
     *
     * @param[in] filename File name
     */
    FilePrinter(const std::string &filename);

private:
    // Inherited methods overridden:
    void print_internal(const std::string &msg) override;

private:
    io::FileHandler _handler;
};
} // namespace logging
} // namespace arm_compute
#endif /* ARM_COMPUTE_LOGGING_FILE_PRINTER_H */
