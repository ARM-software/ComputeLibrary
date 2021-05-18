/*
 * Copyright (c) 2021 Arm Limited.
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
#ifndef SRC_COMMON_LOG_H
#define SRC_COMMON_LOG_H

#include "arm_compute/core/Error.h"
#include "arm_compute/core/utils/logging/Macros.h"

#ifdef ARM_COMPUTE_LOGGING_ENABLED
/** Create a logger
 *
 * @note It will eventually create all default loggers in don't exist
 */
#define ARM_COMPUTE_CREATE_ACL_LOGGER()                                                                                        \
    do                                                                                                                         \
    {                                                                                                                          \
        if(arm_compute::logging::LoggerRegistry::get().logger("ComputeLibrary") == nullptr)                                    \
        {                                                                                                                      \
            arm_compute::logging::LoggerRegistry::get().create_logger("ComputeLibrary", arm_compute::logging::LogLevel::INFO); \
        }                                                                                                                      \
    } while(false)
#else /* ARM_COMPUTE_LOGGING_ENABLED */
#define ARM_COMPUTE_CREATE_ACL_LOGGER()
#endif /* ARM_COMPUTE_LOGGING_ENABLED */
/** Log a message to the logger
 *
 * @param[in] log_level Logging level
 * @param[in] msg       Message to log
 */
#define ARM_COMPUTE_LOG_MSG_ACL(log_level, msg)                \
    do                                                         \
    {                                                          \
        ARM_COMPUTE_CREATE_ACL_LOGGER();                       \
        ARM_COMPUTE_LOG_MSG("ComputeLibrary", log_level, msg); \
    } while(false)
/** Log a message with format to the logger
 *
 * @param[in] log_level Logging level
 * @param[in] fmt       String format (printf style)
 * @param[in] ...       Message arguments
 */
#define ARM_COMPUTE_LOG_MSG_WITH_FORMAT_ACL(log_level, fmt, ...)                        \
    do                                                                                  \
    {                                                                                   \
        ARM_COMPUTE_CREATE_ACL_LOGGER();                                                \
        ARM_COMPUTE_LOG_MSG_WITH_FORMAT("ComputeLibrary", log_level, fmt, __VA_ARGS__); \
    } while(false)
/** Log an error message to the logger
 *
 * @param[in] msg Message to log
 */
#define ARM_COMPUTE_LOG_ERROR_ACL(msg)                                                     \
    do                                                                                     \
    {                                                                                      \
        ARM_COMPUTE_CREATE_ACL_LOGGER();                                                   \
        ARM_COMPUTE_LOG_MSG("ComputeLibrary", arm_compute::logging::LogLevel::ERROR, msg); \
    } while(false)

/** Log an error message to the logger with function name before the message
 *
 * @param[in] msg Message to log
 */
#define ARM_COMPUTE_LOG_ERROR_WITH_FUNCNAME_ACL(msg)                                                     \
    do                                                                                                   \
    {                                                                                                    \
        ARM_COMPUTE_CREATE_ACL_LOGGER();                                                                 \
        ARM_COMPUTE_LOG_MSG_WITH_FUNCNAME("ComputeLibrary", arm_compute::logging::LogLevel::ERROR, msg); \
    } while(false)

#endif /* SRC_COMMON_LOG_H */
