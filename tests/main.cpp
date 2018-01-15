/*
 * Copyright (c) 2017-2018 ARM Limited.
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
#include "support/ToolchainSupport.h"
#include "tests/AssetsLibrary.h"
#include "tests/framework/DatasetModes.h"
#include "tests/framework/Exceptions.h"
#include "tests/framework/Framework.h"
#include "tests/framework/Macros.h"
#include "tests/framework/Profiler.h"
#include "tests/framework/command_line/CommandLineOptions.h"
#include "tests/framework/command_line/CommandLineParser.h"
#include "tests/framework/command_line/CommonOptions.h"
#include "tests/framework/instruments/Instruments.h"
#include "tests/framework/printers/Printers.h"

#ifdef ARM_COMPUTE_CL
#include "arm_compute/runtime/CL/CLScheduler.h"
#endif /* ARM_COMPUTE_CL */
#ifdef ARM_COMPUTE_GC
#include "arm_compute/runtime/GLES_COMPUTE/GCScheduler.h"
#endif /* ARM_COMPUTE_GC */
#include "arm_compute/runtime/Scheduler.h"

#include <fstream>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <random>
#include <utility>

using namespace arm_compute;
using namespace arm_compute::test;

namespace arm_compute
{
namespace test
{
std::unique_ptr<AssetsLibrary> library;
} // namespace test
} // namespace arm_compute

int main(int argc, char **argv)
{
#ifdef ARM_COMPUTE_CL
    CLScheduler::get().default_init();
#endif /* ARM_COMPUTE_CL */

#ifdef ARM_COMPUTE_GC
    GCScheduler::get().default_init();
#endif /* ARM_COMPUTE_GC */

    framework::Framework &framework = framework::Framework::get();

    framework::CommandLineParser parser;

    std::set<framework::DatasetMode> allowed_modes
    {
        framework::DatasetMode::PRECOMMIT,
        framework::DatasetMode::NIGHTLY,
        framework::DatasetMode::ALL
    };

    framework::CommonOptions options(parser);

    auto dataset_mode = parser.add_option<framework::EnumOption<framework::DatasetMode>>("mode", allowed_modes, framework::DatasetMode::PRECOMMIT);
    dataset_mode->set_help("For managed datasets select which group to use");
    auto filter = parser.add_option<framework::SimpleOption<std::string>>("filter", ".*");
    filter->set_help("Regular expression to select test cases");
    auto filter_id = parser.add_option<framework::SimpleOption<std::string>>("filter-id");
    filter_id->set_help("List of test ids. ... can be used to define a range.");
    auto stop_on_error = parser.add_option<framework::ToggleOption>("stop-on-error");
    stop_on_error->set_help("Abort execution after the first failed test (useful for debugging)");
    auto seed = parser.add_option<framework::SimpleOption<std::random_device::result_type>>("seed", std::random_device()());
    seed->set_help("Global seed for random number generation");
    auto list_tests = parser.add_option<framework::ToggleOption>("list-tests", false);
    list_tests->set_help("List all test names");
    auto test_instruments = parser.add_option<framework::ToggleOption>("test-instruments", false);
    test_instruments->set_help("Test if the instruments work on the platform");
    auto error_on_missing_assets = parser.add_option<framework::ToggleOption>("error-on-missing-assets", false);
    error_on_missing_assets->set_help("Mark a test as failed instead of skipping it when assets are missing");
    auto assets = parser.add_positional_option<framework::SimpleOption<std::string>>("assets");
    assets->set_help("Path to the assets directory");

    try
    {
        parser.parse(argc, argv);

        if(options.help->is_set() && options.help->value())
        {
            parser.print_help(argv[0]);
            return 0;
        }

        std::vector<std::unique_ptr<framework::Printer>> printers = options.create_printers();

        Scheduler::get().set_num_threads(options.threads->value());

        if(options.log_level->value() > framework::LogLevel::NONE)
        {
            for(auto &p : printers)
            {
                p->print_global_header();
            }
        }

        if(options.log_level->value() >= framework::LogLevel::CONFIG)
        {
            for(auto &p : printers)
            {
                p->print_entry("Version", build_information());
                p->print_entry("Seed", support::cpp11::to_string(seed->value()));
                p->print_entry("Iterations", support::cpp11::to_string(options.iterations->value()));
                p->print_entry("Threads", support::cpp11::to_string(options.threads->value()));
                {
                    using support::cpp11::to_string;
                    p->print_entry("Dataset mode", to_string(dataset_mode->value()));
                }
            }
        }

        framework.init(options.instruments->value(), options.iterations->value(), dataset_mode->value(), filter->value(), filter_id->value(), options.log_level->value());
        for(auto &p : printers)
        {
            framework.add_printer(p.get());
        }
        framework.set_throw_errors(options.throw_errors->value());
        framework.set_stop_on_error(stop_on_error->value());
        framework.set_error_on_missing_assets(error_on_missing_assets->value());

        bool success = true;

        if(list_tests->value())
        {
            for(auto &p : printers)
            {
                p->print_list_tests(framework.test_infos());
                p->print_global_footer();
            }

            return 0;
        }

        if(test_instruments->value())
        {
            framework::Profiler profiler = framework.get_profiler();
            profiler.start();
            profiler.stop();
            for(auto &p : printers)
            {
                p->print_measurements(profiler.measurements());
            }

            return 0;
        }

        library = support::cpp14::make_unique<AssetsLibrary>(assets->value(), seed->value());

        if(!parser.validate())
        {
            return 1;
        }

        success = framework.run();

        if(options.log_level->value() > framework::LogLevel::NONE)
        {
            for(auto &p : printers)
            {
                p->print_global_footer();
            }
        }

#ifdef ARM_COMPUTE_CL
        CLScheduler::get().sync();
#endif /* ARM_COMPUTE_CL */

        return (success ? 0 : 1);
    }
    catch(const std::exception &error)
    {
        std::cerr << error.what() << "\n";

        if(options.throw_errors->value())
        {
            throw;
        }

        return 1;
    }

    return 0;
}
