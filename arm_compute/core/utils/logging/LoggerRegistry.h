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
#ifndef __ARM_COMPUTE_LOGGING_LOGGER_REGISTRY_H__
#define __ARM_COMPUTE_LOGGING_LOGGER_REGISTRY_H__

#include "arm_compute/core/utils/logging/Logger.h"
#include "arm_compute/core/utils/logging/Printers.h"
#include "arm_compute/core/utils/logging/Types.h"
#include "support/Mutex.h"

#include <memory>
#include <set>
#include <unordered_map>

namespace arm_compute
{
namespace logging
{
/** Registry class holding all the instantiated loggers */
class LoggerRegistry final
{
public:
    /** Gets registry instance
     *
     * @return Logger registry instance
     */
    static LoggerRegistry &get();
    /** Creates a logger
     *
     * @note Some names are reserved e.g. [CORE, RUNTIME, GRAPH]
     *
     * @param[in] name      Logger's name
     * @param[in] log_level Logger's log level. Defaults to @ref LogLevel::INFO
     * @param[in] printers  Printers to attach to the system loggers. Defaults with a @ref StdPrinter.
     */
    void create_logger(const std::string &name, LogLevel log_level = LogLevel::INFO,
                       std::vector<std::shared_ptr<Printer>> printers = { std::make_shared<StdPrinter>() });
    /** Remove a logger
     *
     * @param name Logger's name
     */
    void remove_logger(const std::string &name);
    /** Returns a logger instance
     *
     * @param[in] name Logger to return
     *
     * @return Logger
     */
    std::shared_ptr<Logger> logger(const std::string &name);
    /** Creates reserved library loggers
     *
     * @param[in] log_level (Optional) Logger's log level. Defaults to @ref LogLevel::INFO
     * @param[in] printers  (Optional) Printers to attach to the system loggers. Defaults with a @ref StdPrinter.
     */
    void create_reserved_loggers(LogLevel                              log_level = LogLevel::INFO,
                                 std::vector<std::shared_ptr<Printer>> printers  = { std::make_shared<StdPrinter>() });

private:
    /** Default constructor */
    LoggerRegistry();

private:
    arm_compute::Mutex _mtx;
    std::unordered_map<std::string, std::shared_ptr<Logger>> _loggers;
    static std::set<std::string> _reserved_loggers;
};
} // namespace logging
} // namespace arm_compute
#endif /* __ARM_COMPUTE_LOGGING_LOGGER_REGISTRY_H__ */
