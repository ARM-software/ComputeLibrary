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
#ifndef CKW_INCLUDE_CKW_ERROR_H
#define CKW_INCLUDE_CKW_ERROR_H

#include <stdexcept>
#include <string>

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
std::string
create_error_msg(const std::string &file, const std::string &func, const std::string &line, const std::string &msg);

/** Print the given message then throw an std::runtime_error.
 *
 * @param[in] msg Message to display.
 */
#define COMPUTE_KERNEL_WRITER_ERROR_ON_MSG(msg)                       \
    do                                                                \
    {                                                                 \
        const std::string arg0(__FILE__);                             \
        const std::string arg1(__func__);                             \
        const std::string arg2(std::to_string(__LINE__));             \
        const std::string arg3(msg);                                  \
        std::runtime_error(create_error_msg(arg0, arg1, arg2, arg3)); \
    } while(false)

/** Mark the variables as unused.
 *
 * @param[in] ... Variables which are unused.
 */
#define CKW_UNUSED(...) ckw::ignore_unused(__VA_ARGS__) // NOLINT

/** Mark the variables as unused.
 *
 * @param[in] ... Variables which are unused.
 */
template <typename... T>
inline void ignore_unused(T &&...)
{
}

/** Throw an std::runtime_error with the specified message.
 *
 * @param[in] msg The error message.
 */
#define CKW_THROW_MSG(msg)                                                              \
    do                                                                                  \
    {                                                                                   \
        const std::string file(__FILE__);                                               \
        const std::string func(__func__);                                               \
        const std::string line(std::to_string(__LINE__));                               \
        const std::string message(msg);                                                 \
                                                                                        \
        throw std::runtime_error(ckw::create_error_msg(file, func, line, message)); \
    } while(false)

#ifdef COMPUTE_KERNEL_WRITER_ASSERTS_ENABLED

/** If the condition is not met, throw an std::runtime_error with the specified message if assertion is enabled.
 *
 * @param[in] cond The condition that is expected to be true.
 * @param[in] msg  The error message when the condition is not met.
 */
#define CKW_ASSERT_MSG(cond, msg) \
    do                            \
    {                             \
        if(!(cond))               \
        {                         \
            CKW_THROW_MSG(msg);   \
        }                         \
    } while(false)

#else // COMPUTE_KERNEL_WRITER_ASSERTS_ENABLED

#define CKW_ASSERT_MSG(cond, msg)

#endif // COMPUTE_KERNEL_WRITER_ASSERTS_ENABLED

/** If the condition is not met, throw an std::runtime_error if assertion is enabled.
 *
 * @param[in] cond The condition that is expected to be true.
 */
#define CKW_ASSERT(cond) CKW_ASSERT_MSG(cond, #cond)

/** If the precondition is met but the condition is not met, throw an std::runtime_error if assertion is enabled.
 *
 * @param[in] precond The precondition that triggers the check.
 * @param[in] cond    The condition that is expected to be true if precondition is true.
 */
#define CKW_ASSERT_IF(precond, cond) CKW_ASSERT(!(precond) || (cond))

/** Throw an std::runtime_error with the specified message if assertion is enabled.
 *
 * @param[in] msg  The error message when the condition is not met.
 */
#define CKW_ASSERT_FAILED_MSG(msg) CKW_ASSERT_MSG(false, msg)

} // namespace ckw

#endif // CKW_INCLUDE_CKW_ERROR_H
