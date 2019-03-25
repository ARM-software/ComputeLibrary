/*
 * Copyright (c) 2017-2019 ARM Limited.
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
#ifndef ARM_COMPUTE_UTILS_COMMANDLINEPARSER
#define ARM_COMPUTE_UTILS_COMMANDLINEPARSER

#include "Option.h"
#include "arm_compute/core/utils/misc/Utility.h"
#include "support/ToolchainSupport.h"

#include <iostream>
#include <map>
#include <memory>
#include <regex>
#include <string>
#include <utility>
#include <vector>

namespace arm_compute
{
namespace utils
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

inline void CommandLineParser::parse(int argc, char **argv)
{
    const std::regex option_regex{ "--((?:no-)?)([^=]+)(?:=(.*))?" };

    const auto set_option = [&](const std::string & option, const std::string & name, const std::string & value)
    {
        if(_options.find(name) == _options.end())
        {
            _unknown_options.push_back(option);
            return;
        }

        const bool success = _options[name]->parse(value);

        if(!success)
        {
            _invalid_options.push_back(option);
        }
    };

    unsigned int positional_index = 0;

    for(int i = 1; i < argc; ++i)
    {
        std::string mixed_case_opt{ argv[i] };
        int         equal_sign = mixed_case_opt.find('=');
        int         pos        = (equal_sign == -1) ? strlen(argv[i]) : equal_sign;

        const std::string option = arm_compute::utility::tolower(mixed_case_opt.substr(0, pos)) + mixed_case_opt.substr(pos);
        std::smatch       option_matches;

        if(std::regex_match(option, option_matches, option_regex))
        {
            // Boolean option
            if(option_matches.str(3).empty())
            {
                set_option(option, option_matches.str(2), option_matches.str(1).empty() ? "true" : "false");
            }
            else
            {
                // Can't have "no-" and a value
                if(!option_matches.str(1).empty())
                {
                    _invalid_options.emplace_back(option);
                }
                else
                {
                    set_option(option, option_matches.str(2), option_matches.str(3));
                }
            }
        }
        else
        {
            if(positional_index >= _positional_options.size())
            {
                _invalid_options.push_back(mixed_case_opt);
            }
            else
            {
                _positional_options[positional_index]->parse(mixed_case_opt);
                ++positional_index;
            }
        }
    }
}

inline bool CommandLineParser::validate() const
{
    bool is_valid = true;

    for(const auto &option : _options)
    {
        if(option.second->is_required() && !option.second->is_set())
        {
            is_valid = false;
            std::cerr << "ERROR: Option '" << option.second->name() << "' is required but not given!\n";
        }
    }

    for(const auto &option : _positional_options)
    {
        if(option->is_required() && !option->is_set())
        {
            is_valid = false;
            std::cerr << "ERROR: Option '" << option->name() << "' is required but not given!\n";
        }
    }

    for(const auto &option : _unknown_options)
    {
        std::cerr << "WARNING: Skipping unknown option '" << option << "'!\n";
    }

    for(const auto &option : _invalid_options)
    {
        std::cerr << "WARNING: Skipping invalid option '" << option << "'!\n";
    }

    return is_valid;
}

inline void CommandLineParser::print_help(const std::string &program_name) const
{
    std::cout << "usage: " << program_name << " \n";

    for(const auto &option : _options)
    {
        std::cout << option.second->help() << "\n";
    }

    for(const auto &option : _positional_options)
    {
        // TODO(COMPMID-2079): Print help string as well
        std::cout << option->name() << "\n";
    }
}
} // namespace utils
} // namespace arm_compute
#endif /* ARM_COMPUTE_UTILS_COMMANDLINEPARSER */
