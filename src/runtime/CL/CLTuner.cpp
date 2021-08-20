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
#include "arm_compute/runtime/CL/CLTuner.h"
#include "arm_compute/runtime/CL/tuners/CLTuningParametersList.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "src/core/CL/ICLKernel.h"
#include "support/StringSupport.h"

#include <cerrno>
#include <fstream>
#include <limits>

namespace arm_compute
{
CLTuner::CLTuner(bool tune_new_kernels, CLTuningInfo tuning_info)
    : real_clEnqueueNDRangeKernel(nullptr), _tuning_params_table(), _lws_table(), _kernel_event(), _tune_new_kernels(tune_new_kernels), _tuning_info(tuning_info)
{
}

bool CLTuner::kernel_event_is_set() const
{
    return _kernel_event() != nullptr;
}
void CLTuner::set_cl_kernel_event(cl_event kernel_event)
{
    _kernel_event = kernel_event;
}

void CLTuner::set_tune_new_kernels(bool tune_new_kernels)
{
    _tune_new_kernels = tune_new_kernels;
}
bool CLTuner::tune_new_kernels() const
{
    return _tune_new_kernels;
}

void CLTuner::set_tuner_mode(CLTunerMode mode)
{
    _tuning_info.tuner_mode = mode;
}

void CLTuner::tune_kernel_static(ICLKernel &kernel)
{
    ARM_COMPUTE_UNUSED(kernel);
}

void CLTuner::tune_kernel_dynamic(ICLKernel &kernel)
{
    ITensorPack pack;
    tune_kernel_dynamic(kernel, pack);
}

void CLTuner::tune_kernel_dynamic(ICLKernel &kernel, ITensorPack &tensors)
{
    // Get the configuration ID from the kernel and append GPU target name and number of available compute units
    const std::string config_id = kernel.config_id() + "_" + string_from_target(kernel.get_target()) + "_MP" + support::cpp11::to_string(CLKernelLibrary::get().get_num_compute_units());

    // Check if we need to find the Optimal LWS. If the kernel's config_id is equal to default_config_id, the kernel does not require to be tuned
    if(kernel.config_id() != arm_compute::default_config_id)
    {
        auto p = _tuning_params_table.find(config_id);

        if(p == _tuning_params_table.end())
        {
            if(_tune_new_kernels)
            {
                // Find the optimal LWS for the kernel
                CLTuningParams opt_tuning_params = find_optimal_tuning_params(kernel, tensors);

                // Insert the optimal LWS in the table
                add_tuning_params(config_id, opt_tuning_params);

                // Set Local-Workgroup-Size
                kernel.set_lws_hint(opt_tuning_params.get_lws());
                if(_tuning_info.tune_wbsm)
                {
                    kernel.set_wbsm_hint(opt_tuning_params.get_wbsm());
                }
            }
        }
        else
        {
            // Set Local-Workgroup-Size
            kernel.set_lws_hint(p->second.get_lws());
            if(_tuning_info.tune_wbsm)
            {
                kernel.set_wbsm_hint(p->second.get_wbsm());
            }
        }
    }
}

void CLTuner::add_tuning_params(const std::string &kernel_id, CLTuningParams optimal_tuning_params)
{
    _tuning_params_table.emplace(kernel_id, optimal_tuning_params);
}

CLTuningParams CLTuner::find_optimal_tuning_params(ICLKernel &kernel, ITensorPack &tensors)
{
    // Profiling queue
    cl::CommandQueue queue_profiler;

    // Extract real OpenCL function to intercept
    if(real_clEnqueueNDRangeKernel == nullptr)
    {
        real_clEnqueueNDRangeKernel = CLSymbols::get().clEnqueueNDRangeKernel_ptr;
    }

    // Get the default queue
    cl::CommandQueue default_queue = CLScheduler::get().queue();

    // Check if we can use the OpenCL timer with the default queue
    cl_command_queue_properties props = default_queue.getInfo<CL_QUEUE_PROPERTIES>();

    if((props & CL_QUEUE_PROFILING_ENABLE) == 0)
    {
        // Set the queue for profiling
        queue_profiler = cl::CommandQueue(CLScheduler::get().context(), props | CL_QUEUE_PROFILING_ENABLE);
    }
    else
    {
        queue_profiler = default_queue;
    }

    // Start intercepting enqueues:
    auto interceptor = [this](cl_command_queue command_queue, cl_kernel kernel, cl_uint work_dim, const size_t *gwo, const size_t *gws, const size_t *lws, cl_uint num_events_in_wait_list,
                              const cl_event * event_wait_list, cl_event * event)
    {
        if(this->kernel_event_is_set())
        {
            // If the event is already set it means the kernel enqueue is sliced: given that we only time the first slice we can save time by skipping the other enqueues.
            return CL_SUCCESS;
        }
        cl_event tmp;
        cl_int   retval = this->real_clEnqueueNDRangeKernel(command_queue, kernel, work_dim, gwo, gws, lws, num_events_in_wait_list, event_wait_list, &tmp);

        // Set OpenCL event
        this->set_cl_kernel_event(tmp);

        if(event != nullptr)
        {
            //return cl_event from the intercepted call
            clRetainEvent(tmp);
            *event = tmp;
        }
        return retval;
    };
    CLSymbols::get().clEnqueueNDRangeKernel_ptr = interceptor;

    cl::NDRange gws = ICLKernel::gws_from_window(kernel.window());

    // Run the kernel with default lws to be used as baseline
    const bool inject_memory = !tensors.empty();
    inject_memory ? kernel.run_op(tensors, kernel.window(), queue_profiler) : kernel.run(kernel.window(), queue_profiler);

    queue_profiler.finish();

    const cl_ulong start         = _kernel_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    const cl_ulong end           = _kernel_event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
    cl_ulong       min_exec_time = end - start;
    _kernel_event                = nullptr;

    CLTuningParams opt_tuning_params(cl::NullRange, 0);

    // Construct the list of tuning parameters values to be tested based on the tuner mode.
    auto tuning_list = cl_tuner::get_tuning_parameters_list(_tuning_info, gws);
    for(size_t i = 0; i < tuning_list->size(); ++i)
    {
        CLTuningParams tuning_test = (*tuning_list)[i];
        // Setting the lws
        cl::NDRange lws_test    = tuning_test.get_lws();
        auto        x           = lws_test[0];
        auto        y           = lws_test[1];
        auto        z           = lws_test[2];
        const bool  invalid_lws = (x * y * z > kernel.get_max_workgroup_size()) || (x == 1 && y == 1 && z == 1);

        if(invalid_lws)
        {
            continue;
        }

        kernel.set_lws_hint(lws_test);
        if(_tuning_info.tune_wbsm && CLKernelLibrary::get().is_wbsm_supported())
        {
            cl_int wbsm_test = tuning_test.get_wbsm();
            kernel.set_wbsm_hint(wbsm_test);
        }

        // Run the kernel
        inject_memory ? kernel.run_op(tensors, kernel.window(), queue_profiler) : kernel.run(kernel.window(), queue_profiler);

        queue_profiler.finish();

        const cl_ulong start = _kernel_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
        const cl_ulong end   = _kernel_event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
        const cl_ulong diff  = end - start;
        _kernel_event        = nullptr;

        // Check the execution time
        if(diff < min_exec_time)
        {
            min_exec_time = diff;
            opt_tuning_params.set_lws(tuning_test.get_lws());
            if(_tuning_info.tune_wbsm)
            {
                opt_tuning_params.set_wbsm(tuning_test.get_wbsm());
            }
        }
    }

    // Restore real function
    CLSymbols::get().clEnqueueNDRangeKernel_ptr = real_clEnqueueNDRangeKernel;
    return opt_tuning_params;
}

const std::unordered_map<std::string, CLTuningParams> &CLTuner::tuning_params_table() const
{
    return _tuning_params_table;
}

void CLTuner::import_tuning_params(const std::unordered_map<std::string, CLTuningParams> &tuning_params_table)
{
    _tuning_params_table.clear();
    _tuning_params_table = tuning_params_table;
}

void CLTuner::load_from_file(const std::string &filename)
{
    std::ifstream fs;
    fs.exceptions(std::ifstream::badbit);
    fs.open(filename, std::ios::in);
    if(!fs.is_open())
    {
        ARM_COMPUTE_ERROR_VAR("Failed to open '%s' (%s [%d])", filename.c_str(), strerror(errno), errno);
    }
    std::string line;
    bool        header_line = true;
    while(!std::getline(fs, line).fail())
    {
        if(header_line)
        {
            header_line            = false;
            size_t pos_lws         = line.find("lws");
            size_t pos_wbsm        = line.find("wbsm");
            _tuning_info.tune_wbsm = false;
            if(pos_lws != std::string::npos || pos_wbsm != std::string::npos)
            {
                // The file has in the first line the parameters it has been tuned on
                if(pos_wbsm != std::string::npos)
                {
                    _tuning_info.tune_wbsm = true;
                }
                // Once the line with the tuning parameter is read we can
                // read the next one to start collecting the values
                if(std::getline(fs, line).fail())
                {
                    break;
                }
            }
        }

        CLTuningParams tuning_params;
        size_t         pos = line.find(";");
        if(pos == std::string::npos)
        {
            ARM_COMPUTE_ERROR_VAR("Malformed row '%s' in %s", line.c_str(), filename.c_str());
        }
        std::string kernel_id = line.substr(0, pos);
        line.erase(0, pos + 1);
        if(!tuning_params.from_string(_tuning_info, line))
        {
            ARM_COMPUTE_ERROR_VAR("Malformed row '%s' in %s", line.c_str(), filename.c_str());
        }
        add_tuning_params(kernel_id, tuning_params);
    }
    fs.close();
}

bool CLTuner::save_to_file(const std::string &filename) const
{
    if(!_tune_new_kernels || _tuning_params_table.empty() || filename.empty())
    {
        return false;
    }
    std::ofstream fs;
    fs.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    fs.open(filename, std::ios::out);
    std::string header_string = "";
    header_string += "lws";
    if(_tuning_info.tune_wbsm)
    {
        if(!header_string.empty())
        {
            header_string += " ";
        }
        header_string += "wbsm";
    }
    fs << header_string << std::endl;
    for(auto const &kernel_data : _tuning_params_table)
    {
        CLTuningParams tun_pams(kernel_data.second);
        fs << kernel_data.first << tun_pams.to_string(_tuning_info) << std::endl;
    }
    fs.close();
    return true;
}
} // namespace arm_compute
