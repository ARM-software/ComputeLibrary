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
#include "Globals.h"
#include "PMUCounter.h"
#include "PerformanceProgramOptions.h"
#include "PerformanceUserConfiguration.h"
#include "TensorLibrary.h"
#include "Utils.h"
#include "WallClockTimer.h"

#include "benchmark/benchmark_api.h"
#include "support/ToolchainSupport.h"

#ifdef OPENCL
#include "arm_compute/runtime/CL/CLScheduler.h"
#endif
#include "arm_compute/runtime/Scheduler.h"

#include <iostream>
#include <memory>

using namespace arm_compute::test;
using namespace arm_compute::test::performance;

namespace arm_compute
{
namespace test
{
PerformanceUserConfiguration   user_config;
std::unique_ptr<TensorLibrary> library;
} // namespace test
} // namespace arm_compute

int main(int argc, char **argv)
{
    PerformanceProgramOptions options;
    try
    {
        options.parse_commandline(argc, argv);

        if(options.wants_help())
        {
            std::cout << "Usage: " << argv[0] << " [options] PATH\n";
            std::cout << options.get_help() << "\n";
        }

        user_config = PerformanceUserConfiguration(options);
    }
    catch(const boost::program_options::required_option &err)
    {
        std::cerr << "Error: " << err.what() << "\n";
        std::cout << "\nUsage: " << argv[0] << " [options] PATH\n";
        std::cout << options.get_help() << "\n";
        return 1;
    }

    ::benchmark::Initialize(&argc, argv);

    if(user_config.seed.is_set())
    {
        library = arm_compute::support::cpp14::make_unique<TensorLibrary>(user_config.path.get(), user_config.seed);
    }
    else
    {
        library = arm_compute::support::cpp14::make_unique<TensorLibrary>(user_config.path.get());
    }

#ifdef OPENCL
    arm_compute::CLScheduler::get().default_init();
#endif

    std::cout << "Using " << user_config.threads << " CPU " << (user_config.threads == 1 ? "thread" : "threads") << "\n";
    std::cout << "Seed: " << library->seed() << "\n";
    arm_compute::Scheduler::get().set_num_threads(user_config.threads);

    ::benchmark::RunSpecifiedBenchmarks();
}
