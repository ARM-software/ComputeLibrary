/*
 * Copyright (c) 2017-2021 Arm Limited.
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
#include "support/StringSupport.h"
#include "tests/AssetsLibrary.h"
#include "tests/framework/DatasetModes.h"
#include "tests/framework/Exceptions.h"
#include "tests/framework/Framework.h"
#include "tests/framework/Macros.h"
#include "tests/framework/ParametersLibrary.h"
#include "tests/framework/Profiler.h"
#include "tests/framework/command_line/CommonOptions.h"
#include "tests/framework/instruments/Instruments.h"
#include "tests/framework/printers/Printers.h"
#include "tests/instruments/Helpers.h"
#include "utils/command_line/CommandLineOptions.h"
#include "utils/command_line/CommandLineParser.h"

#ifdef ARM_COMPUTE_CL
#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/runtime/CL/CLGEMMHeuristicsHandle.h"
#include "arm_compute/runtime/CL/CLHelpers.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "arm_compute/runtime/CL/CLTuner.h"
#include "utils/TypePrinter.h"
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
} // namespace
namespace arm_compute
{
namespace test
{
std::unique_ptr<AssetsLibrary> library;

static constexpr uint32_t      fixed_seed = 1;
std::unique_ptr<AssetsLibrary> fixed_library;

extern std::unique_ptr<ParametersLibrary> parameters;
} // namespace test
} // namespace arm_compute

namespace
{
#ifdef ARM_COMPUTE_CL
bool file_exists(const std::string &filename)
{
    std::ifstream file(filename);
    return file.good();
}
#endif /* ARM_COMPUTE_CL */
} //namespace

int main(int argc, char **argv)
{
    framework::Framework &framework = framework::Framework::get();

    utils::CommandLineParser parser;

    std::set<framework::DatasetMode> allowed_modes
    {
        framework::DatasetMode::DISABLED,
        framework::DatasetMode::PRECOMMIT,
        framework::DatasetMode::NIGHTLY,
        framework::DatasetMode::ALL
    };

    framework::CommonOptions options(parser);

    auto dataset_mode = parser.add_option<utils::EnumOption<framework::DatasetMode>>("mode", allowed_modes, framework::DatasetMode::PRECOMMIT);
    dataset_mode->set_help("For managed datasets select which group to use");
    auto filter = parser.add_option<utils::SimpleOption<std::string>>("filter", ".*");
    filter->set_help("Regular expression to select test cases");
    auto filter_id = parser.add_option<utils::SimpleOption<std::string>>("filter-id");
    filter_id->set_help("List of test ids. ... can be used to define a range.");
    auto stop_on_error = parser.add_option<utils::ToggleOption>("stop-on-error");
    stop_on_error->set_help("Abort execution after the first failed test (useful for debugging)");
    auto seed = parser.add_option<utils::SimpleOption<std::random_device::result_type>>("seed", std::random_device()());
    seed->set_help("Global seed for random number generation");
    auto list_tests = parser.add_option<utils::ToggleOption>("list-tests", false);
    list_tests->set_help("List all test names");
    auto test_instruments = parser.add_option<utils::ToggleOption>("test-instruments", false);
    test_instruments->set_help("Test if the instruments work on the platform");
    auto error_on_missing_assets = parser.add_option<utils::ToggleOption>("error-on-missing-assets", false);
    error_on_missing_assets->set_help("Mark a test as failed instead of skipping it when assets are missing");
    auto assets = parser.add_positional_option<utils::SimpleOption<std::string>>("assets");
    assets->set_help("Path to the assets directory");
#ifdef ARM_COMPUTE_CL
    auto enable_tuner = parser.add_option<utils::ToggleOption>("enable-tuner");
    enable_tuner->set_help("Enable OpenCL dynamic tuner");

    const std::set<CLTunerMode> supported_tuner_modes
    {
        CLTunerMode::EXHAUSTIVE,
        CLTunerMode::NORMAL,
        CLTunerMode::RAPID
    };
    auto tuner_mode = parser.add_option<utils::EnumOption<CLTunerMode>>("tuner-mode", supported_tuner_modes, CLTunerMode::NORMAL);
    tuner_mode->set_help("Configures the time taken by the tuner to tune. Slow tuner produces the most performant LWS configuration");

    auto tuner_file = parser.add_option<utils::SimpleOption<std::string>>("tuner-file", "");
    tuner_file->set_help("File to load/save CLTuner values");

    auto mlgo_file = parser.add_option<utils::SimpleOption<std::string>>("mlgo-file", "");
    mlgo_file->set_help("File to load MLGO heuristics");
#endif /* ARM_COMPUTE_CL */
    auto threads = parser.add_option<utils::SimpleOption<int>>("threads", 1);
    threads->set_help("Number of threads to use");
    auto cooldown_sec = parser.add_option<utils::SimpleOption<float>>("delay", -1.f);
    cooldown_sec->set_help("Delay to add between test executions in seconds");

    try
    {
        parser.parse(argc, argv);

        if(options.help->is_set() && options.help->value())
        {
            parser.print_help(argv[0]);
            return 0;
        }

        std::vector<std::unique_ptr<framework::Printer>> printers = options.create_printers();

        // Setup CPU Scheduler
        Scheduler::get().set_num_threads(threads->value());

        // Create CPU context
        auto cpu_ctx = std::make_unique<RuntimeContext>();
        cpu_ctx->set_scheduler(&Scheduler::get());

        // Track CPU context
        auto cpu_ctx_track = std::make_unique<ContextSchedulerUser>(cpu_ctx.get());

        // Create parameters
        parameters = std::make_unique<ParametersLibrary>();
        parameters->set_cpu_ctx(std::move(cpu_ctx));

#ifdef ARM_COMPUTE_GC
        // Setup OpenGL context
        {
            auto gles_ctx = std::make_unique<GCRuntimeContext>();
            ARM_COMPUTE_ERROR_ON(gles_ctx == nullptr);
            {
                // Legacy singletons API: This has been deprecated and the singletons will be removed in future releases
                // Setup singleton for backward compatibility
                GCScheduler::get().default_init();
            }
            parameters->set_gc_ctx(std::move(gles_ctx));
        };
#endif /* ARM_COMPUTE_GC */

#ifdef ARM_COMPUTE_CL
        CLTuner                cl_tuner(false);
        CLGEMMHeuristicsHandle gemm_heuristics;
        if(opencl_is_available())
        {
            auto ctx_dev_err = create_opencl_context_and_device();
            ARM_COMPUTE_ERROR_ON_MSG(std::get<2>(ctx_dev_err) != CL_SUCCESS, "Failed to create OpenCL context");
            gemm_heuristics.reload_from_file(mlgo_file->value());
            CLScheduler::get().default_init_with_context(std::get<1>(ctx_dev_err), std::get<0>(ctx_dev_err), &cl_tuner, &gemm_heuristics);
        }

        if(enable_tuner->is_set())
        {
            cl_tuner.set_tune_new_kernels(enable_tuner->value());

            //set tuner mode
            cl_tuner.set_tuner_mode(tuner_mode->value());

            // If that's the first run then the file won't exist yet
            if(file_exists(tuner_file->value()))
            {
                cl_tuner.load_from_file(tuner_file->value());
            }
        }
        else if(!tuner_file->value().empty())
        {
            //If we're not tuning and the file doesn't exist then we should raise an error:
            cl_tuner.load_from_file(tuner_file->value());
        }
#endif /* ARM_COMPUTE_CL */
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
                    p->print_entry("CL_DEVICE_VERSION", CLKernelLibrary::get().get_device_version());
                }
                else
                {
                    p->print_entry("CL_DEVICE_VERSION", "Unavailable");
                }
#endif /* ARM_COMPUTE_CL */
                const arm_compute::CPUInfo &cpu_info = Scheduler::get().cpu_info();
                const unsigned int          num_cpus = cpu_info.get_cpu_num();
                p->print_entry("cpu_has_fp16", support::cpp11::to_string(cpu_info.has_fp16()));
                p->print_entry("cpu_has_dotprod", support::cpp11::to_string(cpu_info.has_dotprod()));

                for(unsigned int j = 0; j < num_cpus; ++j)
                {
                    const CPUModel model = cpu_info.get_cpu_model(j);
                    p->print_entry("CPU" + support::cpp11::to_string(j), cpu_model_to_string(model));
                }
                p->print_entry("Iterations", support::cpp11::to_string(options.iterations->value()));
                p->print_entry("Threads", support::cpp11::to_string(threads->value()));
                {
                    using support::cpp11::to_string;
                    p->print_entry("Dataset mode", to_string(dataset_mode->value()));
                }
            }
        }

        // Setup instruments meta-data
        framework::InstrumentsInfo instruments_info;
        instruments_info._scheduler_users.push_back(cpu_ctx_track.get());
        framework.set_instruments_info(instruments_info);

        // Initialize framework
        framework::FrameworkConfig fconfig;
        fconfig.instruments    = options.instruments->value();
        fconfig.name_filter    = filter->value();
        fconfig.id_filter      = filter_id->value();
        fconfig.num_iterations = options.iterations->value();
        fconfig.mode           = dataset_mode->value();
        fconfig.log_level      = options.log_level->value();
        fconfig.cooldown_sec   = cooldown_sec->value();
        framework.init(fconfig);

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

        library       = std::make_unique<AssetsLibrary>(assets->value(), seed->value());
        fixed_library = std::make_unique<AssetsLibrary>(assets->value(), fixed_seed);

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
        if(opencl_is_available())
        {
            CLScheduler::get().sync();
            if(enable_tuner->is_set() && enable_tuner->value() && tuner_file->is_set())
            {
                cl_tuner.save_to_file(tuner_file->value());
            }
        }
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
