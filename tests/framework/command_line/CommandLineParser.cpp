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
#include "CommandLineParser.h"

#include <iostream>
#include <regex>

namespace arm_compute
{
namespace test
{
namespace framework
{
void CommandLineParser::parse(int argc, char **argv)
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

        const std::string option = tolower(mixed_case_opt.substr(0, pos)) + mixed_case_opt.substr(pos);
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

bool CommandLineParser::validate() const
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

void CommandLineParser::print_help(const std::string &program_name) const
{
    std::cout << "usage: " << program_name << " \n";

    for(const auto &option : _options)
    {
        std::cout << option.second->help() << "\n";
    }

    for(const auto &option : _positional_options)
    {
        std::cout << option->name() << "\n";
    }
}
} // namespace framework
} // namespace test
} // namespace arm_compute
