/*
 * Copyright (c) 2018-2020 ARM Limited.
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

#define BENCHMARK_EXAMPLES
#include "utils/Utils.cpp"

#include "ValidateExample.h"
#include "arm_compute/runtime/Scheduler.h"
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/framework/Framework.h"
#include "tests/framework/Macros.h"
#include "tests/framework/command_line/CommonOptions.h"
#include "tests/framework/instruments/Instruments.h"
#include "utils/command_line/CommandLineParser.h"

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
static std::unique_ptr<ValidateExample> g_example      = nullptr;
static std::vector<char *>              g_example_argv = {};

namespace
{
std::string command_line(int argc, char **argv)
{
    std::stringstream ss;
    for(int i = 0; i < argc; i++)
    {
        ss << argv[i] << " ";
    }
    return ss.str();
}

template <bool validate>
class ExampleTest : public arm_compute::test::framework::TestCase
{
public:
    ExampleTest() = default;
    void do_setup() override
    {
        ARM_COMPUTE_ERROR_ON_NULLPTR(g_example.get());
        _is_setup = g_example->do_setup(g_example_argv.size(), &g_example_argv[0]);
    }
    void do_run() override
    {
        if(_is_setup)
        {
            g_example->do_run();
        }
    }
    void do_teardown() override
    {
        if(_is_setup)
        {
            if(validate)
            {
                g_example->do_validate();
            }
            g_example->do_teardown();
        }
        g_example = nullptr;
    }

private:
    bool _is_setup{ false };
};

} // namespace
int run_example(int argc, char **argv, std::unique_ptr<ValidateExample> example)
{
    utils::CommandLineParser parser;
    framework::CommonOptions options(parser);
    auto                     example_args = parser.add_option<utils::ListOption<std::string>>("example_args");
    example_args->set_help("Arguments to pass to the example separated by commas (e.g: arg0,arg1,arg2)");
    auto seed = parser.add_option<utils::SimpleOption<std::random_device::result_type>>("seed", std::random_device()());
    seed->set_help("Global seed for random number generation");
    auto validate = parser.add_option<utils::SimpleOption<int>>("validate", 1);
    validate->set_help("Enable / disable output validation (0/1)");

    framework::Framework &framework = framework::Framework::get();

    parser.parse(argc, argv);

    if(options.help->is_set() && options.help->value())
    {
        parser.print_help(argv[0]);
        return 0;
    }

    std::vector<std::unique_ptr<framework::Printer>> printers = options.create_printers();
    g_example                                                 = std::move(example);
    g_example_argv.clear();
    g_example_argv.emplace_back(argv[0]);
    for(auto &arg : example_args->value())
    {
        g_example_argv.emplace_back(const_cast<char *>(arg.c_str())); // NOLINT
    }

    library = support::cpp14::make_unique<AssetsLibrary>("." /* Only using random values */, seed->value());

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
            p->print_entry("CommandLine", command_line(argc, argv));
            p->print_entry("Seed", support::cpp11::to_string(seed->value()));
#ifdef ARM_COMPUTE_CL
            if(opencl_is_available())
            {
                if(!CLScheduler::get().is_initialised())
                {
                    CLScheduler::get().default_init();
                }
                p->print_entry("CL_DEVICE_VERSION", CLKernelLibrary::get().get_device_version());
            }
            else
            {
                p->print_entry("CL_DEVICE_VERSION", "Unavailable");
            }
#endif /* ARM_COMPUTE_CL */
            p->print_entry("Iterations", support::cpp11::to_string(options.iterations->value()));
            g_example->print_parameters(*p);
        }
    }

    // Initialize framework
    framework::FrameworkConfig fconfig;
    fconfig.instruments    = options.instruments->value();
    fconfig.num_iterations = options.iterations->value();
    fconfig.log_level      = options.log_level->value();
    framework.init(fconfig);

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
