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
#ifndef ARM_COMPUTE_TEST_COMMANDLINEPARSER
#define ARM_COMPUTE_TEST_COMMANDLINEPARSER

#include "../Utils.h"
#include "Option.h"

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace arm_compute
{
namespace test
{
namespace framework
{
/** Class to parse command line arguments. */
class CommandLineParser final
{
public:
    /** Default constructor. */
    CommandLineParser() = default;

    /** Function to add a new option to the parser.
     *
     * @param[in] name Name of the option. Will be available under --name=VALUE.
     * @param[in] args Option specific configuration arguments.
     *
     * @return Pointer to the option. The option is owned by the parser.
     */
    template <typename T, typename... As>
    T *add_option(const std::string &name, As &&... args);

    /** Function to add a new positional argument to the parser.
     *
     * @param[in] args Option specific configuration arguments.
     *
     * @return Pointer to the option. The option is owned by the parser.
     */
    template <typename T, typename... As>
    T *add_positional_option(As &&... args);

    /** Parses the command line arguments and updates the options accordingly.
     *
     * @param[in] argc Number of arguments.
     * @param[in] argv Arguments.
     */
    void parse(int argc, char **argv);

    /** Validates the previously parsed command line arguments.
     *
     * Validation fails if not all required options are provided. Additionally
     * warnings are generated for options that have illegal values or unknown
     * options.
     *
     * @return True if all required options have been provided.
     */
    bool validate() const;

    /** Prints a help message for all configured options.
     *
     * @param[in] program_name Name of the program to be used in the help message.
     */
    void print_help(const std::string &program_name) const;

private:
    using OptionsMap              = std::map<std::string, std::unique_ptr<Option>>;
    using PositionalOptionsVector = std::vector<std::unique_ptr<Option>>;

    OptionsMap               _options{};
    PositionalOptionsVector  _positional_options{};
    std::vector<std::string> _unknown_options{};
    std::vector<std::string> _invalid_options{};
};

template <typename T, typename... As>
inline T *CommandLineParser::add_option(const std::string &name, As &&... args)
{
    auto result = _options.emplace(name, support::cpp14::make_unique<T>(name, std::forward<As>(args)...));
    return static_cast<T *>(result.first->second.get());
}

template <typename T, typename... As>
inline T *CommandLineParser::add_positional_option(As &&... args)
{
    _positional_options.emplace_back(support::cpp14::make_unique<T>(std::forward<As>(args)...));
    return static_cast<T *>(_positional_options.back().get());
}
} // namespace framework
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_COMMANDLINEPARSER */
