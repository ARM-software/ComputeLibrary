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
#include "utils/Utils.h"
//FIXME / INTERNAL_ONLY: This file should not be released!

#define BENCHMARK_EXAMPLES
#include "utils/Utils.cpp"

#include "ValidateExample.h"
#include "arm_compute/runtime/Scheduler.h"
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/framework/Framework.h"
#include "tests/framework/Macros.h"
#include "tests/framework/command_line/CommandLineParser.h"
#include "tests/framework/command_line/CommonOptions.h"
#include "tests/framework/instruments/Instruments.h"

#ifdef ARM_COMPUTE_CL
#include "arm_compute/runtime/CL/CLScheduler.h"
#endif /* ARM_COMPUTE_CL */
#ifdef ARM_COMPUTE_GC
#include "arm_compute/runtime/GLES_COMPUTE/GCScheduler.h"
#endif /* ARM_COMPUTE_GC */

#include <libgen.h>

using namespace arm_compute;
using namespace arm_compute::test;

namespace arm_compute
{
namespace test
{
std::unique_ptr<AssetsLibrary> library;
} // namespace test
namespace utils
{
static ValidateExample *g_example = nullptr;
template <bool          validate>
class ExampleTest : public arm_compute::test::framework::TestCase
{
public:
    ExampleTest() = default;
    void do_run() override
    {
        g_example->do_run();
    }
    void do_teardown() override
    {
        if(validate)
        {
            g_example->do_validate();
        }
        g_example->do_teardown();
    }
};

int run_example(int argc, char **argv, ValidateExample &example)
{
    framework::CommandLineParser parser;
    framework::CommonOptions     options(parser);
    auto                         example_args = parser.add_option<framework::ListOption<std::string>>("example_args");
    example_args->set_help("Arguments to pass to the example separated by commas (e.g: arg0,arg1,arg2)");
    auto seed = parser.add_option<framework::SimpleOption<std::random_device::result_type>>("seed", std::random_device()());
    seed->set_help("Global seed for random number generation");
    auto validate = parser.add_option<framework::SimpleOption<int>>("validate", 1);
    validate->set_help("Enable / disable output validation (0/1)");

    framework::Framework &framework = framework::Framework::get();

    parser.parse(argc, argv);

    if(options.help->is_set() && options.help->value())
    {
        parser.print_help(argv[0]);
        return 0;
    }

    std::vector<std::unique_ptr<framework::Printer>> printers = options.create_printers();
    g_example                                                 = &example;
    std::vector<char *> example_argv                          = {};
    example_argv.clear();
    example_argv.emplace_back(argv[0]);
    for(auto &arg : example_args->value())
    {
        example_argv.emplace_back(const_cast<char *>(arg.c_str())); // NOLINT
    }

    // Set number of threads in Scheduler
    Scheduler::get().set_num_threads(options.threads->value());
    library = support::cpp14::make_unique<AssetsLibrary>("." /* Only using random values */, seed->value());

    // We need to do the setup here because framework.init() will need CL / GLES to be initialised
    try
    {
        example.do_setup(example_argv.size(), &example_argv[0]);
    }
#ifdef ARM_COMPUTE_CL
    catch(cl::Error &err)
    {
        std::cerr << "!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
        std::cerr << std::endl
                  << "ERROR " << err.what() << "(" << err.err() << ")" << std::endl;
        std::cerr << "!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
        return 1;
    }
#endif /* ARM_COMPUTE_CL */
    catch(std::runtime_error &err)
    {
        std::cerr << "!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
        std::cerr << std::endl
                  << "ERROR " << err.what() << " " << (errno ? strerror(errno) : "") << std::endl;
        std::cerr << "!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
        return 1;
    }

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
#ifdef ARM_COMPUTE_CL
            if(opencl_is_available())
            {
                p->print_entry("CL_DEVICE_VERSION", CLKernelLibrary::get().get_device_version());
            }
            else
            {
                p->print_entry("CL_DEVICE_VERSION", "Unavailable");
            }
#endif /* ARM_COMPUTE_CL */
            p->print_entry("Iterations", support::cpp11::to_string(options.iterations->value()));
            p->print_entry("Threads", support::cpp11::to_string(options.threads->value()));
            example.print_parameters(*p);
        }
    }

    framework.init(options.instruments->value(), options.iterations->value(), framework::DatasetMode::ALL, "", "", options.log_level->value());
    for(auto &p : printers)
    {
        framework.add_printer(p.get());
    }

    framework.set_throw_errors(options.throw_errors->value());
    arm_compute::test::framework::detail::TestSuiteRegistrar suite{ "Examples" };
    if(validate->value() != 0)
    {
        framework.add_test_case<ExampleTest<true>>(basename(argv[0]), framework::DatasetMode::ALL, arm_compute::test::framework::TestCaseFactory::Status::ACTIVE);
    }
    else
    {
        framework.add_test_case<ExampleTest<false>>(basename(argv[0]), framework::DatasetMode::ALL, arm_compute::test::framework::TestCaseFactory::Status::ACTIVE);
    }

    //func(argc, argv);
    bool success = framework.run();
    if(options.log_level->value() > framework::LogLevel::NONE)
    {
        for(auto &p : printers)
        {
            p->print_global_footer();
        }
    }

    return (success ? 0 : 1);
}

} // namespace utils
} // namespace arm_compute
