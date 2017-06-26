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
#define BOOST_TEST_ALTERNATIVE_INIT_API

#include "Globals.h"
#include "TensorLibrary.h"
#include "Utils.h"
#include "ValidationProgramOptions.h"
#include "ValidationUserConfiguration.h"

#include "arm_compute/runtime/Scheduler.h"

#include "boost_wrapper.h"

#include <iostream>
#include <memory>
#include <random>

using namespace arm_compute::test;
using namespace arm_compute::test::validation;

namespace arm_compute
{
namespace test
{
ValidationUserConfiguration    user_config;
std::unique_ptr<TensorLibrary> library;
} // namespace test
} // namespace arm_compute

struct GlobalFixture
{
    GlobalFixture()
    {
        if(user_config.seed.is_set())
        {
            library = cpp14::make_unique<TensorLibrary>(user_config.path.get(), user_config.seed);
        }
        else
        {
            library = cpp14::make_unique<TensorLibrary>(user_config.path.get());
        }

        std::cout << "Seed: " << library->seed();
    }
};

BOOST_GLOBAL_FIXTURE(GlobalFixture);

bool init_unit_test()
{
    boost::unit_test::framework::master_test_suite().p_name.value = "Compute Library Validation Tests";

    ValidationProgramOptions options;

    int   &argc = boost::unit_test::framework::master_test_suite().argc;
    char **argv = boost::unit_test::framework::master_test_suite().argv;

    try
    {
        options.parse_commandline(argc, argv);

        if(options.wants_help())
        {
            std::cout << "Usage: " << argv[0] << " [options] PATH\n";
            std::cout << options.get_help() << "\n";
            return false;
        }

        user_config = ValidationUserConfiguration(options);
    }
    catch(const boost::program_options::required_option &err)
    {
        std::cerr << "Error: " << err.what() << "\n";
        std::cout << "\nUsage: " << argv[0] << " [options] PATH\n";
        std::cout << options.get_help() << "\n";
        return false;
    }

    std::cout << "Using " << user_config.threads << " CPU " << (user_config.threads == 1 ? "thread" : "threads") << "\n";
    arm_compute::Scheduler::get().set_num_threads(user_config.threads);
    return true;
}
