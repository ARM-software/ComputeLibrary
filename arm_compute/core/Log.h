/*
 * Copyright (c) 2017 ARM Limited.
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
#ifndef __ARM_COMPUTE_LOG_H__
#define __ARM_COMPUTE_LOG_H__

#include "arm_compute/core/utils/logging/Macros.h"

#ifdef ARM_COMPUTE_LOGGING_ENABLED
/** Create a default core logger
 *
 * @note It will eventually create all default loggers in don't exist
 */
#define ARM_COMPUTE_CREATE_DEFAULT_CORE_LOGGER()                                   \
    do                                                                             \
    {                                                                              \
        if(arm_compute::logging::LoggerRegistry::get().logger("CORE") == nullptr)  \
        {                                                                          \
            arm_compute::logging::LoggerRegistry::get().create_reserved_loggers(); \
        }                                                                          \
    } while(false)
#else /* ARM_COMPUTE_LOGGING_ENABLED */
#define ARM_COMPUTE_CREATE_DEFAULT_CORE_LOGGER()
#endif /* ARM_COMPUTE_LOGGING_ENABLED */

/** Log a message to the core system logger
 *
 * @param[in] log_level Logging level
 * @param[in] msg       Message to log
 */
#define ARM_COMPUTE_LOG_MSG_CORE(log_level, msg)     \
    do                                               \
    {                                                \
        ARM_COMPUTE_CREATE_DEFAULT_CORE_LOGGER();    \
        ARM_COMPUTE_LOG_MSG("CORE", log_level, msg); \
    } while(false)

/** Log a message with format to the core system logger
 *
 * @param[in] log_level Logging level
 * @param[in] fmt       String format (printf style)
 * @param[in] ...       Message arguments
 */
#define ARM_COMPUTE_LOG_MSG_WITH_FORMAT_CORE(log_level, fmt, ...)             \
    do                                                                        \
    {                                                                         \
        ARM_COMPUTE_CREATE_DEFAULT_CORE_LOGGER();                             \
        ARM_COMPUTE_LOG_MSG_WITH_FORMAT("CORE", log_level, fmt, __VA_ARGS__); \
    } while(false)

/** Log a stream to the core system logger
 *
 * @param[in] log_level Logging level
 * @param[in] ss        Stream to log
 */
#define ARM_COMPUTE_LOG_STREAM_CORE(log_level, ss)     \
    do                                                 \
    {                                                  \
        ARM_COMPUTE_CREATE_DEFAULT_CORE_LOGGER();      \
        ARM_COMPUTE_LOG_STREAM("CORE", log_level, ss); \
    } while(false)

/** Log information level message to the core system logger
 *
 * @param[in] msg Stream to log
 */
#define ARM_COMPUTE_LOG_INFO_MSG_CORE(msg)                                   \
    do                                                                       \
    {                                                                        \
        ARM_COMPUTE_CREATE_DEFAULT_CORE_LOGGER();                            \
        ARM_COMPUTE_LOG_MSG_CORE(arm_compute::logging::LogLevel::INFO, msg); \
    } while(false)

/** Log information level formatted message to the core system logger
 *
 * @param[in] fmt String format (printf style)
 * @param[in] ... Message arguments
 */
#define ARM_COMPUTE_LOG_INFO_MSG_WITH_FORMAT_CORE(fmt, ...)                                           \
    do                                                                                                \
    {                                                                                                 \
        ARM_COMPUTE_CREATE_DEFAULT_CORE_LOGGER();                                                     \
        ARM_COMPUTE_LOG_MSG_WITH_FORMAT_CORE(arm_compute::logging::LogLevel::INFO, fmt, __VA_ARGS__); \
    } while(false)

/** Log information level stream to the core system logger
 *
 * @param[in] ss Message to log
 */
#define ARM_COMPUTE_LOG_INFO_STREAM_CORE(ss)                                   \
    do                                                                         \
    {                                                                          \
        ARM_COMPUTE_CREATE_DEFAULT_CORE_LOGGER();                              \
        ARM_COMPUTE_LOG_STREAM_CORE(arm_compute::logging::LogLevel::INFO, ss); \
    } while(false)

#endif /* __ARM_COMPUTE_LOGGING_MACROS_H__ */
