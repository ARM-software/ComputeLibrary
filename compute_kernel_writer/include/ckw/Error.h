/*
 * Copyright (c) 2023 Arm Limited.
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
#ifndef COMPUTE_KERNEL_WRITER_INCLUDE_CKW_ERROR_H
#define COMPUTE_KERNEL_WRITER_INCLUDE_CKW_ERROR_H

#include <string>
#include <stdexcept>

namespace ckw
{
/** Creates the error message
 *
 * @param[in] file       File in which the error occurred.
 * @param[in] func       Function in which the error occurred.
 * @param[in] line       Line in which the error occurred.
 * @param[in] msg        Message to display before abandoning.
 *
 * @return status containing the error
 */
std::string create_error_msg(const std::string &file, const std::string &func, const std::string &line, const std::string &msg);

/** Print the given message then throw an std::runtime_error.
 *
 * @param[in] msg Message to display.
 */
#define COMPUTE_KERNEL_WRITER_ERROR_ON_MSG(msg)     \
    do                                      \
    {                                       \
        const std::string arg0(__FILE__);   \
        const std::string arg1(__func__);   \
        const std::string arg2(std::to_string(__LINE__));   \
        const std::string arg3(msg);         \
        std::runtime_error(create_error_msg(arg0, arg1, arg2, arg3)); \
    } while(false)

} // namespace ckw

#endif /* COMPUTE_KERNEL_WRITER_INCLUDE_CKW_ERROR_H */
