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
#include "CommonOptions.h"

#include "../Framework.h"
#include "../printers/Printers.h"
#include "CommandLineParser.h"

namespace arm_compute
{
namespace test
{
namespace framework
{
CommonOptions::CommonOptions(CommandLineParser &parser)
    : help(parser.add_option<ToggleOption>("help")),
      instruments(),
      iterations(parser.add_option<SimpleOption<int>>("iterations", 1)),
      threads(parser.add_option<SimpleOption<int>>("threads", 1)),
      log_format(),
      log_file(parser.add_option<SimpleOption<std::string>>("log-file")),
      log_level(),
      throw_errors(parser.add_option<ToggleOption>("throw-errors")),
      color_output(parser.add_option<ToggleOption>("color-output", true)),
      pretty_console(parser.add_option<ToggleOption>("pretty-console", false)),
      json_file(parser.add_option<SimpleOption<std::string>>("json-file")),
      pretty_file(parser.add_option<SimpleOption<std::string>>("pretty-file")),
      log_streams()
{
    Framework                       &framework = Framework::get();
    std::set<InstrumentsDescription> allowed_instruments
    {
        std::pair<InstrumentType, ScaleFactor>(InstrumentType::ALL, ScaleFactor::NONE),
        std::pair<InstrumentType, ScaleFactor>(InstrumentType::NONE, ScaleFactor::NONE),
    };

    for(const auto &type : framework.available_instruments())
    {
        allowed_instruments.insert(type);
    }

    std::set<LogFormat> supported_log_formats
    {
        LogFormat::NONE,
        LogFormat::PRETTY,
        LogFormat::JSON,
    };

    std::set<LogLevel> supported_log_levels
    {
        LogLevel::NONE,
        LogLevel::CONFIG,
        LogLevel::TESTS,
        LogLevel::ERRORS,
        LogLevel::DEBUG,
        LogLevel::MEASUREMENTS,
        LogLevel::ALL,
    };

    instruments = parser.add_option<EnumListOption<InstrumentsDescription>>("instruments", allowed_instruments, std::initializer_list<InstrumentsDescription> { std::pair<InstrumentType, ScaleFactor>(InstrumentType::WALL_CLOCK_TIMER, ScaleFactor::NONE) });
    log_format  = parser.add_option<EnumOption<LogFormat>>("log-format", supported_log_formats, LogFormat::PRETTY);
    log_level   = parser.add_option<EnumOption<LogLevel>>("log-level", supported_log_levels, LogLevel::ALL);

    help->set_help("Show this help message");
    instruments->set_help("Set the profiling instruments to use");
    iterations->set_help("Number of iterations per test case");
    threads->set_help("Number of threads to use");
    log_format->set_help("Output format for measurements and failures (affects only log-file)");
    log_file->set_help("Write output to file instead of to the console (affected by log-format)");
    log_level->set_help("Verbosity of the output");
    throw_errors->set_help("Don't catch fatal errors (useful for debugging)");
    color_output->set_help("Produce colored output on the console");
    pretty_console->set_help("Produce pretty output on the console");
    json_file->set_help("Write output to a json file.");
    pretty_file->set_help("Write output to a text file");
}
std::vector<std::unique_ptr<Printer>> CommonOptions::create_printers()
{
    std::vector<std::unique_ptr<Printer>> printers;

    if(pretty_console->value() && (log_file->is_set() || log_format->value() != LogFormat::PRETTY))
    {
        auto pretty_printer = support::cpp14::make_unique<PrettyPrinter>();
        pretty_printer->set_color_output(color_output->value());
        printers.push_back(std::move(pretty_printer));
    }

    std::unique_ptr<Printer> printer;
    switch(log_format->value())
    {
        case LogFormat::JSON:
            printer = support::cpp14::make_unique<JSONPrinter>();
            break;
        case LogFormat::NONE:
            break;
        case LogFormat::PRETTY:
        default:
            auto pretty_printer = support::cpp14::make_unique<PrettyPrinter>();
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
        printers.push_back(support::cpp14::make_unique<JSONPrinter>());
        log_streams.push_back(std::make_shared<std::ofstream>(json_file->value()));
        printers.back()->set_stream(*log_streams.back().get());
    }

    if(pretty_file->is_set())
    {
        printers.push_back(support::cpp14::make_unique<PrettyPrinter>());
        log_streams.push_back(std::make_shared<std::ofstream>(pretty_file->value()));
        printers.back()->set_stream(*log_streams.back().get());
    }

    return printers;
}
} // namespace framework
} // namespace test
} // namespace arm_compute
