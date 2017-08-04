/*
 * Copyright (c) 2016, 2017 ARM Limited.
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
#ifndef __ARM_COMPUTE_ERROR_H__
#define __ARM_COMPUTE_ERROR_H__

/** Print the given message then throw an std::runtime_error.
 *
 * @param[in] ... Message to display before aborting.
 */
#ifndef ARM_NO_EXCEPTIONS
#define ARM_COMPUTE_ERROR(...) ::arm_compute::error(__func__, __FILE__, __LINE__, __VA_ARGS__) // NOLINT
#else
#define ARM_COMPUTE_ERROR(...) // NOLINT
#endif // ARM_NO_EXCEPTIONS

/** Print the given message then throw an std::runtime_error.
 *
 * @param[in] func Function in which the error occurred.
 * @param[in] file File in which the error occurred.
 * @param[in] line Line in which the error occurred.
 * @param[in] ...  Message to display before aborting.
 */
#ifndef ARM_NO_EXCEPTIONS
#define ARM_COMPUTE_ERROR_LOC(func, file, line, ...) ::arm_compute::error(func, file, line, __VA_ARGS__) // NOLINT
#else
#define ARM_COMPUTE_ERROR_LOC(func, file, line, ...)
#endif // ARM_NO_EXCEPTIONS

/** To avoid unused variables warnings
 *
 * This is useful if for example a variable is only used
 * in debug builds and generates a warning in release builds.
 *
 * @param[in] var Variable which is unused
 */
#define ARM_COMPUTE_UNUSED(var) (void)(var)

#ifdef ARM_COMPUTE_DEBUG_ENABLED
/** Print the given message
 *
 * @param[in] ... Message to display
 */
#define ARM_COMPUTE_INFO(...) ::arm_compute::debug(__func__, __FILE__, __LINE__, __VA_ARGS__) // NOLINT
/** If the condition is true, the given message is printed
 *
 * @param[in] cond Condition to evaluate.
 * @param[in] ...  Message to print if cond is false.
 */
#define ARM_COMPUTE_INFO_ON_MSG(cond, ...) \
    do                                     \
    {                                      \
        if(cond)                           \
        {                                  \
            ARM_COMPUTE_INFO(__VA_ARGS__); \
        }                                  \
    } while(0)
#else /* ARM_COMPUTE_DEBUG_ENABLED */
#define ARM_COMPUTE_INFO_ON_MSG(cond, ...)
#define ARM_COMPUTE_INFO(...)
#endif /* ARM_COMPUTE_DEBUG_ENABLED */

#ifdef ARM_COMPUTE_ASSERTS_ENABLED
/** If the condition is true, the given message is printed and an exception is thrown
 *
 * @param[in] cond Condition to evaluate.
 * @param[in] ...  Message to print if cond is false.
 */
#define ARM_COMPUTE_ERROR_ON_MSG(cond, ...) \
    do                                      \
    {                                       \
        if(cond)                            \
        {                                   \
            ARM_COMPUTE_ERROR(__VA_ARGS__); \
        }                                   \
    } while(0)

/** If the condition is true, the given message is printed and an exception is thrown
 *
 * @param[in] cond Condition to evaluate.
 * @param[in] func Function in which the error occurred.
 * @param[in] file File in which the error occurred.
 * @param[in] line Line in which the error occurred.
 * @param[in] ...  Message to print if cond is false.
 */
#define ARM_COMPUTE_ERROR_ON_LOC_MSG(cond, func, file, line, ...) \
    do                                                            \
    {                                                             \
        if(cond)                                                  \
        {                                                         \
            ARM_COMPUTE_ERROR_LOC(func, file, line, __VA_ARGS__); \
        }                                                         \
    } while(0)

/** If the condition is true, the given message is printed and an exception is thrown, otherwise value is returned
 *
 * @param[in] cond Condition to evaluate.
 * @param[in] val  Value to be returned.
 * @param[in] msg  Message to print if cond is false.
 */
#define ARM_COMPUTE_CONST_ON_ERROR(cond, val, msg) (cond) ? throw std::logic_error(msg) : val;
#else /* ARM_COMPUTE_ASSERTS_ENABLED */
#define ARM_COMPUTE_ERROR_ON_MSG(cond, ...)
#define ARM_COMPUTE_ERROR_ON_LOC_MSG(cond, func, file, line, ...)
#define ARM_COMPUTE_CONST_ON_ERROR(cond, val, msg) val
#endif /* ARM_COMPUTE_ASSERTS_ENABLED */

/** If the condition is true then an error message is printed and an exception thrown
 *
 * @param[in] cond Condition to evaluate
 */
#define ARM_COMPUTE_ERROR_ON(cond) \
    ARM_COMPUTE_ERROR_ON_MSG(cond, #cond)

/** If the condition is true then an error message is printed and an exception thrown
 *
 * @param[in] cond Condition to evaluate
 * @param[in] func Function in which the error occurred.
 * @param[in] file File in which the error occurred.
 * @param[in] line Line in which the error occurred.
 */
#define ARM_COMPUTE_ERROR_ON_LOC(cond, func, file, line) \
    ARM_COMPUTE_ERROR_ON_LOC_MSG(cond, func, file, line, #cond)

namespace arm_compute
{
/** Print an error message then throw an std::runtime_error
 *
 * @param[in] function Function in which the error occurred.
 * @param[in] file     Name of the file where the error occurred.
 * @param[in] line     Line on which the error occurred.
 * @param[in] msg      Message to display before aborting.
 * @param[in] ...      Variable number of arguments of the message.
 */
[[noreturn]] void error(const char *function, const char *file, const int line, const char *msg, ...);

/** Print a debug message
 *
 * @param[in] function Function in which the error occurred.
 * @param[in] file     Name of the file where the error occurred.
 * @param[in] line     Line on which the error occurred.
 * @param[in] msg      Message to display before aborting.
 * @param[in] ...      Variable number of arguments of the message.
 */
void debug(const char *function, const char *file, const int line, const char *msg, ...);
}

#endif /* __ARM_COMPUTE_ERROR_H__ */
