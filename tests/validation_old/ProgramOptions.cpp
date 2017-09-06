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
#include "ProgramOptions.h"

#include "arm_compute/core/Types.h"
#include "tests/TypePrinter.h"
#include "tests/TypeReader.h"

#include <random>
#include <sstream>

namespace arm_compute
{
namespace test
{
ProgramOptions::ProgramOptions()
{
    boost::program_options::options_description generic("Generic options");
    generic.add_options()("help", "Print help message")("seed", boost::program_options::value<std::random_device::result_type>()->default_value(std::random_device()()), "Seed for the tensor library");

    _visible.add(generic);

    _hidden.add_options()("path", boost::program_options::value<std::string>(), "Path from where to load the asset/s");

    _positional.add("path", 1);
}

void ProgramOptions::add_options(const boost::program_options::options_description &options)
{
    _visible.add(options);
}

bool ProgramOptions::wants_help() const
{
    return (_vm.count("help") != 0);
}

std::string ProgramOptions::get_help() const
{
    std::stringstream help;
    help << _visible;

    return help.str();
}

void ProgramOptions::parse_commandline(int argc, char **argv)
{
    boost::program_options::options_description all;
    all.add(_visible).add(_hidden);

    boost::program_options::store(boost::program_options::command_line_parser(argc, argv)
                                  .options(all)
                                  .positional(_positional)
                                  .allow_unregistered()
                                  .run(),
                                  _vm);

    if(_vm.count("help") == 0 && _vm.count("path") == 0)
    {
        throw boost::program_options::required_option("PATH");
    }

    boost::program_options::notify(_vm);
}
} // namespace test
} // namespace arm_compute
