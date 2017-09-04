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
#ifndef __ARM_COMPUTE_TEST_PROGRAM_OPTIONS_H__
#define __ARM_COMPUTE_TEST_PROGRAM_OPTIONS_H__

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Weffc++"
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
#pragma GCC diagnostic ignored "-Wctor-dtor-privacy"
#include "boost/program_options.hpp"
#pragma GCC diagnostic pop

#include <random>
#include <sstream>

namespace arm_compute
{
namespace test
{
/** Defines available commandline arguments and allows to parse them. */
class ProgramOptions
{
public:
    /** Defines available options. */
    ProgramOptions();

    /** Signals if the --help flag has been passed on the commandline. */
    bool wants_help() const;

    /** Returns a string describing all available options. */
    std::string get_help() const;

    /** Parses the given arguments and makes them available via @ref get.
     *
     * @param[in] argc Number of command line arguments.
     * @param[in] argv Pointer to the command line arguments.
     */
    void parse_commandline(int argc, char **argv);

    /** Sets @p value if it has been specified on the command line.
     *
     * @note The type T has to match the type that has been specified for the
     *       command line option.
     *
     * @param[in]  name  Name of the option to query.
     * @param[out] value Variable to which the value will be assigned.
     *
     * @return True if the value is assigned, false otherwise.
     */
    template <typename T>
    bool get(const std::string &name, T &value) const;

protected:
    /** Allows subclasses to add more specific options
     *
     * @param[in] options Boost object containing options and their descriptions
     */
    void add_options(const boost::program_options::options_description &options);

private:
    boost::program_options::options_description            _hidden{};
    boost::program_options::options_description            _visible{ "Configuration options" };
    boost::program_options::positional_options_description _positional{};
    boost::program_options::variables_map                  _vm{};
};

template <typename T>
bool ProgramOptions::get(const std::string &name, T &value) const
{
    if(_vm.count(name) != 0)
    {
        value = _vm[name].as<T>();
        return true;
    }

    return false;
}
} // namespace test
} // namespace arm_compute
#endif
