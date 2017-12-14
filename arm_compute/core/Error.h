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

#include <stdarg.h>
#include <string>

namespace arm_compute
{
enum class ErrorCode
{
    OK,           /**< No error */
    RUNTIME_ERROR /**< Generic runtime error */
};

/** Status class */
class Status
{
public:
    /** Default Constructor **/
    Status()
        : _code(ErrorCode::OK), _error_description(" ")
    {
    }
    /** Default Constructor
     *
     * @param error_status      Error status.
     * @param error_description (Optional) Error description if error_status is not valid.
     */
    explicit Status(ErrorCode error_status, std::string error_description = " ")
        : _code(error_status), _error_description(error_description)
    {
    }
    /** Allow instances of this class to be copy constructed */
    Status(const Status &) = default;
    /** Allow instances of this class to be move constructed */
    Status(Status &&) = default;
    /** Allow instances of this class to be copy assigned */
    Status &operator=(const Status &) = default;
    /** Allow instances of this class to be move assigned */
    Status &operator=(Status &&) = default;
    /** Explicit bool conversion operator
     *
     * @return True if there is no error else false
     */
    explicit operator bool() const noexcept
    {
        return _code == ErrorCode::OK;
    }
    /** Gets error code
     *
     * @return Error code.
     */
    ErrorCode error_code() const
    {
        return _code;
    }
    /** Gets error description if any
     *
     * @return Error description.
     */
    std::string error_description() const
    {
        return _error_description;
    }
    /** Throws a runtime exception in case it contains a valid error status */
    void throw_if_error()
    {
        if(!bool(*this))
        {
            internal_throw_on_error();
        }
    }

private:
    /** Internal throwing function */
    [[noreturn]] void internal_throw_on_error();

private:
    ErrorCode   _code;
    std::string _error_description;
};

/** Creates an error containing the error message from variable argument list
 *
 * @param[in] error_code Error code
 * @param[in] function   Function in which the error occurred.
 * @param[in] file       Name of the file where the error occurred.
 * @param[in] line       Line on which the error occurred.
 * @param[in] msg        Message to display before aborting.
 * @param[in] args       Variable argument list of the message.
 *
 * @return status containing the error
 */
Status create_error_va_list(ErrorCode error_code, const char *function, const char *file, const int line, const char *msg, va_list args);
/** Creates an error containing the error message
 *
 * @param[in] error_code Error code
 * @param[in] function   Function in which the error occurred.
 * @param[in] file       Name of the file where the error occurred.
 * @param[in] line       Line on which the error occurred.
 * @param[in] msg        Message to display before aborting.
 * @param[in] ...        Variable number of arguments of the message.
 *
 * @return status containing the error
 */
Status create_error(ErrorCode error_code, const char *function, const char *file, const int line, const char *msg, ...);
/** Print an error message then throw an std::runtime_error
 *
 * @param[in] function Function in which the error occurred.
 * @param[in] file     Name of the file where the error occurred.
 * @param[in] line     Line on which the error occurred.
 * @param[in] msg      Message to display before aborting.
 * @param[in] ...      Variable number of arguments of the message.
 */
[[noreturn]] void error(const char *function, const char *file, const int line, const char *msg, ...);
}
/** To avoid unused variables warnings
 *
 * This is useful if for example a variable is only used
 * in debug builds and generates a warning in release builds.
 *
 * @param[in] var Variable which is unused.
 */
#define ARM_COMPUTE_UNUSED(var) (void)(var)

/** Creates an error with a given message
 *
 * @param[in] error_code Error code.
 * @param[in] ...        Message to encapsulate.
 */
#define ARM_COMPUTE_CREATE_ERROR(error_code, ...) ::arm_compute::create_error(error_code, __func__, __FILE__, __LINE__, __VA_ARGS__) // NOLINT

/** Creates an error on location with a given message
 *
 * @param[in] error_code Error code.
 * @param[in] func       Function in which the error occurred.
 * @param[in] file       File in which the error occurred.
 * @param[in] line       Line in which the error occurred.
 * @param[in] ...        Message to display before aborting.
 */
#define ARM_COMPUTE_CREATE_ERROR_LOC(error_code, func, file, line, ...) ::arm_compute::create_error(error_code, func, file, line, __VA_ARGS__) // NOLINT

/** Checks if a status contains an error and returns it
 *
 * @param[in] status Status value to check
 */
#define ARM_COMPUTE_RETURN_ON_ERROR(status) \
    do                                      \
    {                                       \
        if(!bool(status))                   \
        {                                   \
            return status;                  \
        }                                   \
    } while(false)

/** Checks if an error value is valid if not throws an exception with the error
 *
 * @param[in] error Error value to check.
 */
#define ARM_COMPUTE_THROW_ON_ERROR(error) \
    error.throw_if_error();

/** If the condition is true, an error is returned
 *
 * @param[in] cond Condition to evaluate.
 * @param[in] ...  Error description message
 */
#define ARM_COMPUTE_RETURN_ERROR_ON_MSG(cond, ...)                                               \
    do                                                                                           \
    {                                                                                            \
        if(cond)                                                                                 \
        {                                                                                        \
            return ARM_COMPUTE_CREATE_ERROR(arm_compute::ErrorCode::RUNTIME_ERROR, __VA_ARGS__); \
        }                                                                                        \
    } while(false)

/** If the condition is true, an error is thrown
 *
 * @param[in] cond Condition to evaluate.
 * @param[in] func Function in which the error occurred.
 * @param[in] file File in which the error occurred.
 * @param[in] line Line in which the error occurred.
 * @param[in] ...  Error description message.
 */
#define ARM_COMPUTE_RETURN_ERROR_ON_LOC_MSG(cond, func, file, line, ...)                                               \
    do                                                                                                                 \
    {                                                                                                                  \
        if(cond)                                                                                                       \
        {                                                                                                              \
            return ARM_COMPUTE_CREATE_ERROR_LOC(arm_compute::ErrorCode::RUNTIME_ERROR, func, file, line, __VA_ARGS__); \
        }                                                                                                              \
    } while(false)

/** If the condition is true, an error is returned
 *
 * @param[in] cond Condition to evaluate
 */
#define ARM_COMPUTE_RETURN_ERROR_ON(cond) \
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(cond, #cond)

/** If the condition is true, an error is returned
 *
 * @param[in] cond Condition to evaluate.
 * @param[in] func Function in which the error occurred.
 * @param[in] file File in which the error occurred.
 * @param[in] line Line in which the error occurred.
 */
#define ARM_COMPUTE_RETURN_ERROR_ON_LOC(cond, func, file, line) \
    ARM_COMPUTE_RETURN_ERROR_ON_LOC_MSG(cond, func, file, line, #cond)

/** Print the given message then throw an std::runtime_error.
 *
 * @param[in] ... Message to display before aborting.
 */
#define ARM_COMPUTE_ERROR(...) ::arm_compute::error(__func__, __FILE__, __LINE__, __VA_ARGS__) // NOLINT

/** Print the given message then throw an std::runtime_error.
 *
 * @param[in] func Function in which the error occurred.
 * @param[in] file File in which the error occurred.
 * @param[in] line Line in which the error occurred.
 * @param[in] ...  Message to display before aborting.
 */
#define ARM_COMPUTE_ERROR_LOC(func, file, line, ...) ::arm_compute::error(func, file, line, __VA_ARGS__) // NOLINT

#ifdef ARM_COMPUTE_ASSERTS_ENABLED
/** Checks if a status value is valid if not throws an exception with the error
 *
 * @param[in] status Status value to check.
 */
#define ARM_COMPUTE_ERROR_THROW_ON(status) \
    status.throw_if_error()

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
#define ARM_COMPUTE_ERROR_THROW_ON(status)
#define ARM_COMPUTE_ERROR_ON_MSG(cond, ...)
#define ARM_COMPUTE_ERROR_ON_LOC_MSG(cond, func, file, line, ...)
#define ARM_COMPUTE_CONST_ON_ERROR(cond, val, msg) val
#endif /* ARM_COMPUTE_ASSERTS_ENABLED */

/** If the condition is true then an error message is printed and an exception thrown
 *
 * @param[in] cond Condition to evaluate.
 */
#define ARM_COMPUTE_ERROR_ON(cond) \
    ARM_COMPUTE_ERROR_ON_MSG(cond, #cond)

/** If the condition is true then an error message is printed and an exception thrown
 *
 * @param[in] cond Condition to evaluate.
 * @param[in] func Function in which the error occurred.
 * @param[in] file File in which the error occurred.
 * @param[in] line Line in which the error occurred.
 */
#define ARM_COMPUTE_ERROR_ON_LOC(cond, func, file, line) \
    ARM_COMPUTE_ERROR_ON_LOC_MSG(cond, func, file, line, #cond)

#endif /* __ARM_COMPUTE_ERROR_H__ */
