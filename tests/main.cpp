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
#include "support/ToolchainSupport.h"
#include "tests/AssetsLibrary.h"
#include "tests/framework/DatasetModes.h"
#include "tests/framework/Exceptions.h"
#include "tests/framework/Framework.h"
#include "tests/framework/Macros.h"
#include "tests/framework/Profiler.h"
#include "tests/framework/command_line/CommandLineOptions.h"
#include "tests/framework/command_line/CommandLineParser.h"
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

    std::set<framework::InstrumentsDescription> allowed_instruments
    {
        std::pair<framework::InstrumentType, framework::ScaleFactor>(framework::InstrumentType::ALL, framework::ScaleFactor::NONE),
        std::pair<framework::InstrumentType, framework::ScaleFactor>(framework::InstrumentType::NONE, framework::ScaleFactor::NONE),
    };

    for(const auto &type : framework.available_instruments())
    {
        allowed_instruments.insert(type);
    }

    std::set<framework::DatasetMode> allowed_modes
    {
        framework::DatasetMode::PRECOMMIT,
        framework::DatasetMode::NIGHTLY,
        framework::DatasetMode::ALL
    };

    std::set<framework::LogFormat> supported_log_formats
    {
        framework::LogFormat::NONE,
        framework::LogFormat::PRETTY,
        framework::LogFormat::JSON,
    };

    std::set<framework::LogLevel> supported_log_levels
    {
        framework::LogLevel::NONE,
        framework::LogLevel::CONFIG,
        framework::LogLevel::TESTS,
        framework::LogLevel::ERRORS,
        framework::LogLevel::DEBUG,
        framework::LogLevel::MEASUREMENTS,
        framework::LogLevel::ALL,
    };

    auto help = parser.add_option<framework::ToggleOption>("help");
    help->set_help("Show this help message");
    auto dataset_mode = parser.add_option<framework::EnumOption<framework::DatasetMode>>("mode", allowed_modes, framework::DatasetMode::PRECOMMIT);
    dataset_mode->set_help("For managed datasets select which group to use");
    auto instruments = parser.add_option<framework::EnumListOption<framework::InstrumentsDescription>>("instruments", allowed_instruments, std::initializer_list<framework::InstrumentsDescription> { std::pair<framework::InstrumentType, framework::ScaleFactor>(framework::InstrumentType::WALL_CLOCK_TIMER, framework::ScaleFactor::NONE) });
    instruments->set_help("Set the profiling instruments to use");
    auto iterations = parser.add_option<framework::SimpleOption<int>>("iterations", 1);
    iterations->set_help("Number of iterations per test case");
    auto threads = parser.add_option<framework::SimpleOption<int>>("threads", 1);
    threads->set_help("Number of threads to use");
    auto log_format = parser.add_option<framework::EnumOption<framework::LogFormat>>("log-format", supported_log_formats, framework::LogFormat::PRETTY);
    log_format->set_help("Output format for measurements and failures (affects only log-file)");
    auto filter = parser.add_option<framework::SimpleOption<std::string>>("filter", ".*");
    filter->set_help("Regular expression to select test cases");
    auto filter_id = parser.add_option<framework::SimpleOption<std::string>>("filter-id");
    filter_id->set_help("List of test ids. ... can be used to define a range.");
    auto log_file = parser.add_option<framework::SimpleOption<std::string>>("log-file");
    log_file->set_help("Write output to file instead of to the console (affected by log-format)");
    auto log_level = parser.add_option<framework::EnumOption<framework::LogLevel>>("log-level", supported_log_levels, framework::LogLevel::ALL);
    log_level->set_help("Verbosity of the output");
    auto throw_errors = parser.add_option<framework::ToggleOption>("throw-errors");
    throw_errors->set_help("Don't catch fatal errors (useful for debugging)");
    auto stop_on_error = parser.add_option<framework::ToggleOption>("stop-on-error");
    stop_on_error->set_help("Abort execution after the first failed test (useful for debugging)");
    auto seed = parser.add_option<framework::SimpleOption<std::random_device::result_type>>("seed", std::random_device()());
    seed->set_help("Global seed for random number generation");
    auto color_output = parser.add_option<framework::ToggleOption>("color-output", true);
    color_output->set_help("Produce colored output on the console");
    auto list_tests = parser.add_option<framework::ToggleOption>("list-tests", false);
    list_tests->set_help("List all test names");
    auto test_instruments = parser.add_option<framework::ToggleOption>("test-instruments", false);
    test_instruments->set_help("Test if the instruments work on the platform");
    auto error_on_missing_assets = parser.add_option<framework::ToggleOption>("error-on-missing-assets", false);
    error_on_missing_assets->set_help("Mark a test as failed instead of skipping it when assets are missing");
    auto assets = parser.add_positional_option<framework::SimpleOption<std::string>>("assets");
    assets->set_help("Path to the assets directory");
    auto pretty_console = parser.add_option<framework::ToggleOption>("pretty-console", false);
    pretty_console->set_help("Produce pretty output on the console");
    auto json_file = parser.add_option<framework::SimpleOption<std::string>>("json-file");
    json_file->set_help("Write output to a json file.");
    auto pretty_file = parser.add_option<framework::SimpleOption<std::string>>("pretty-file");
    pretty_file->set_help("Write output to a text file");

    try
    {
        parser.parse(argc, argv);

        if(help->is_set() && help->value())
        {
            parser.print_help(argv[0]);
            return 0;
        }

        std::vector<std::unique_ptr<framework::Printer>> printers;
        std::vector<std::shared_ptr<std::ofstream>>      log_streams;

        if(pretty_console->value() && (log_file->is_set() || log_format->value() != framework::LogFormat::PRETTY))
        {
            auto pretty_printer = support::cpp14::make_unique<framework::PrettyPrinter>();
            pretty_printer->set_color_output(color_output->value());
            printers.push_back(std::move(pretty_printer));
        }

        std::unique_ptr<framework::Printer> printer;
        switch(log_format->value())
        {
            case framework::LogFormat::JSON:
                printer = support::cpp14::make_unique<framework::JSONPrinter>();
                break;
            case framework::LogFormat::NONE:
                break;
            case framework::LogFormat::PRETTY:
            default:
                auto pretty_printer = support::cpp14::make_unique<framework::PrettyPrinter>();
                // Don't use colours if we print to a file:
                pretty_printer->set_color_output((!log_file->is_set()) && color_output->value());
                printer = std::move(pretty_printer);
                break;
        }

        if(log_file->is_set())
        {
            log_streams.push_back(std::make_shared<std::ofstream>(log_file->value()));
            if(printer != nullptr)
            {
                printer->set_stream(*log_streams.back().get());
            }
        }

        if(printer != nullptr)
        {
            printers.push_back(std::move(printer));
        }

        if(json_file->is_set())
        {
            printers.push_back(support::cpp14::make_unique<framework::JSONPrinter>());
            log_streams.push_back(std::make_shared<std::ofstream>(json_file->value()));
            printers.back()->set_stream(*log_streams.back().get());
        }

        if(pretty_file->is_set())
        {
            printers.push_back(support::cpp14::make_unique<framework::PrettyPrinter>());
            log_streams.push_back(std::make_shared<std::ofstream>(pretty_file->value()));
            printers.back()->set_stream(*log_streams.back().get());
        }

        Scheduler::get().set_num_threads(threads->value());

        if(log_level->value() > framework::LogLevel::NONE)
        {
            for(auto &p : printers)
            {
                p->print_global_header();
            }
        }

        if(log_level->value() >= framework::LogLevel::CONFIG)
        {
            for(auto &p : printers)
            {
                p->print_entry("Seed", support::cpp11::to_string(seed->value()));
                p->print_entry("Iterations", support::cpp11::to_string(iterations->value()));
                p->print_entry("Threads", support::cpp11::to_string(threads->value()));
                {
                    using support::cpp11::to_string;
                    p->print_entry("Dataset mode", to_string(dataset_mode->value()));
                }
            }
        }

        framework.init(instruments->value(), iterations->value(), dataset_mode->value(), filter->value(), filter_id->value(), log_level->value());
        for(auto &p : printers)
        {
            framework.add_printer(p.get());
        }
        framework.set_throw_errors(throw_errors->value());
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

        if(log_level->value() > framework::LogLevel::NONE)
        {
            for(auto &p : printers)
            {
                p->print_global_footer();
            }
        }

        return (success ? 0 : 1);
    }
    catch(const std::exception &error)
    {
        std::cerr << error.what() << "\n";

        if(throw_errors->value())
        {
            throw;
        }

        return 1;
    }

    return 0;
}
