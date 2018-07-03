/*
 * Copyright (c) 2018 ARM Limited.
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
#ifndef ARM_COMPUTE_TEST_COMMONOPTIONS
#define ARM_COMPUTE_TEST_COMMONOPTIONS

#include "../instruments/Instruments.h"

#include "utils/command_line/CommandLineOptions.h"
#include "utils/command_line/CommandLineParser.h"

#include <memory>

namespace arm_compute
{
namespace test
{
namespace framework
{
class Printer;
enum class LogFormat;
enum class LogLevel;

/** Common command line options used to configure the framework
 *
 * The options in this object get populated when "parse()" is called on the parser used to construct it.
 * The expected workflow is:
 *
 * CommandLineParser parser;
 * CommonOptions options( parser );
 * parser.parse(argc, argv);
 * if(options.log_level->value() > LogLevel::NONE) --> Use the options values
 */
class CommonOptions
{
public:
    /** Constructor
     *
     * @param[in,out] parser A parser on which "parse()" hasn't been called yet.
     */
    CommonOptions(arm_compute::utils::CommandLineParser &parser);
    /** Prevent instances of this class from being copy constructed */
    CommonOptions(const CommonOptions &) = delete;
    /** Prevent instances of this class from being copied */
    CommonOptions &operator=(const CommonOptions &) = delete;
    /** Create the printers based on parsed command line options
     *
     * @pre "parse()" has been called on the parser used to construct this object
     *
     * @return List of printers
     */
    std::vector<std::unique_ptr<Printer>> create_printers();

    arm_compute::utils::ToggleOption                           *help;           /**< Show help option */
    arm_compute::utils::EnumListOption<InstrumentsDescription> *instruments;    /**< Instruments option */
    arm_compute::utils::SimpleOption<int>                      *iterations;     /**< Number of iterations option */
    arm_compute::utils::SimpleOption<int>                      *threads;        /**< Number of threads option */
    arm_compute::utils::EnumOption<LogFormat>                  *log_format;     /**< Log format option */
    arm_compute::utils::SimpleOption<std::string>              *log_file;       /**< Log file option */
    arm_compute::utils::EnumOption<LogLevel>                   *log_level;      /**< Logging level option */
    arm_compute::utils::ToggleOption                           *throw_errors;   /**< Throw errors option */
    arm_compute::utils::ToggleOption                           *color_output;   /**< Color output option */
    arm_compute::utils::ToggleOption                           *pretty_console; /**< Pretty console option */
    arm_compute::utils::SimpleOption<std::string>              *json_file;      /**< JSON output file option */
    arm_compute::utils::SimpleOption<std::string>              *pretty_file;    /**< Pretty output file option */
    std::vector<std::shared_ptr<std::ofstream>>                 log_streams;    /**< Log streams */
};

} // namespace framework
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_COMMONOPTIONS */
