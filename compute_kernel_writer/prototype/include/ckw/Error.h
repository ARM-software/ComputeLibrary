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

#ifndef CKW_PROTOTYPE_INCLUDE_CKW_ERROR_H
#define CKW_PROTOTYPE_INCLUDE_CKW_ERROR_H

#include <stdexcept>
#include <string>

namespace ckw
{

/** If the condition is not met, throw an std::runtime_error with the specified message.
 *
 * @param[in] cond The condition that is expected to be true.
 * @param[in] msg  The error message when the condition is not met.
 */
#define CKW_ASSERT_MSG(cond, msg)            \
    do                                       \
    {                                        \
        if(!(cond))                          \
        {                                    \
            throw ::std::runtime_error(msg); \
        }                                    \
    } while(false)

/** If the condition is not met, throw an std::runtime_error.
 *
 * @param[in] cond The condition that is expected to be true.
 */
#define CKW_ASSERT(cond) CKW_ASSERT_MSG(cond, #cond)

/** If the precondition is met but the consequence is not met, throw an std::runtime_error.
 *
 * @param[in] precond The condition if is met requires the consequence must also be met.
 * @param[in] cond    The condition that is expected to be true if the precondition is true.
 */
#define CKW_ASSERT_IF(precond, cond) \
    CKW_ASSERT_MSG(!(precond) || ((precond) && (cond)), #precond " |-> " #cond)

/** Mark the variables as unused.
 *
 * @param[in] ... Variables which are unused.
 */
#define CKW_UNUSED(...) ::ckw::ignore_unused(__VA_ARGS__) // NOLINT

/** Mark the variables as unused.
 *
 * @param[in] ... Variables which are unused.
 */
template <typename... T>
inline void ignore_unused(T &&...)
{
}

} // namespace ckw

#endif // CKW_INCLUDE_CKW_ERROR_H
