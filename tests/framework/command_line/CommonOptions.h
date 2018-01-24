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
#include "CommandLineOptions.h"
#include <memory>

namespace arm_compute
{
namespace test
{
namespace framework
{
class CommandLineParser;
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
    CommonOptions(CommandLineParser &parser);
    CommonOptions(const CommonOptions &) = delete;
    CommonOptions &operator=(const CommonOptions &) = delete;
    /** Create the printers based on parsed command line options
     *
     * @pre "parse()" has been called on the parser used to construct this object
     *
     * @return List of printers
     */
    std::vector<std::unique_ptr<Printer>> create_printers();

    ToggleOption                               *help;
    EnumListOption<InstrumentsDescription>     *instruments;
    SimpleOption<int>                          *iterations;
    SimpleOption<int>                          *threads;
    EnumOption<LogFormat>                      *log_format;
    SimpleOption<std::string>                  *log_file;
    EnumOption<LogLevel>                       *log_level;
    ToggleOption                               *throw_errors;
    ToggleOption                               *color_output;
    ToggleOption                               *pretty_console;
    SimpleOption<std::string>                  *json_file;
    SimpleOption<std::string>                  *pretty_file;
    std::vector<std::shared_ptr<std::ofstream>> log_streams;
};

} // namespace framework
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_COMMONOPTIONS */
