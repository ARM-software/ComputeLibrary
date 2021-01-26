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
#include <memory>
#include <string>

namespace arm_compute
{
CLTuner::CLTuner(bool tune_new_kernels, CLTuningInfo tuning_info)
    : real_clEnqueueNDRangeKernel(nullptr), _tuning_params_table(), _lws_table(), _kernel_event(), _tune_new_kernels(tune_new_kernels), _tuning_info(tuning_info), _tuner_mode(CLTunerMode::NORMAL)
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
    _tuner_mode = mode;
}

CLTunerMode CLTuner::get_tuner_mode() const
{
    return _tuner_mode;
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
            }
        }
        else
        {
            // Set Local-Workgroup-Size
            kernel.set_lws_hint(p->second.get_lws());
        }
    }
}

void CLTuner::add_lws_to_table(const std::string &kernel_id, cl::NDRange optimal_lws)
{
    add_tuning_params(kernel_id, CLTuningParams(optimal_lws));
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

    cl::NDRange opt_lws = cl::NullRange;

    // Construct the list of tuning parameters values to be tested based on the tuner mode.
    auto lws_list = cl_tuner::get_tuning_parameters_list(_tuner_mode, gws);
    for(size_t i = 0; i < lws_list->size(); ++i)
    {
        cl::NDRange lws_test    = (*lws_list)[i].get_lws();
        auto        x           = lws_test[0];
        auto        y           = lws_test[1];
        auto        z           = lws_test[2];
        const bool  invalid_lws = (x * y * z > kernel.get_max_workgroup_size()) || (x == 1 && y == 1 && z == 1);

        if(invalid_lws)
        {
            continue;
        }

        //Set the Local-Workgroup-Size
        kernel.set_lws_hint(lws_test);

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
            opt_lws       = cl::NDRange(x, y, z);
        }
    }

    // Restore real function
    CLSymbols::get().clEnqueueNDRangeKernel_ptr = real_clEnqueueNDRangeKernel;
    return CLTuningParams(opt_lws);
}

void CLTuner::import_lws_table(const std::unordered_map<std::string, cl::NDRange> &lws_table)
{
    _tuning_params_table.clear();
    for(auto && params : lws_table)
    {
        add_tuning_params(params.first, CLTuningParams(params.second));
    }
}

const std::unordered_map<std::string, cl::NDRange> &CLTuner::lws_table()
{
    _lws_table.clear();
    for(auto && params : _tuning_params_table)
    {
        _lws_table.emplace(params.first, params.second.get_lws());
    }
    return _lws_table;
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
    while(!std::getline(fs, line).fail())
    {
        std::istringstream ss(line);
        std::string        token;
        if(std::getline(ss, token, ';').fail())
        {
            ARM_COMPUTE_ERROR_VAR("Malformed row '%s' in %s (Should be of the form 'kernel_id;lws[0];lws[1];lws[2]')", ss.str().c_str(), filename.c_str());
        }
        std::string kernel_id = token;
        cl::NDRange lws(1, 1, 1);
        for(int i = 0; i < 3; i++)
        {
            if(std::getline(ss, token, ';').fail())
            {
                ARM_COMPUTE_ERROR_VAR("Malformed row '%s' in %s (Should be of the form 'kernel_id;lws[0];lws[1];lws[2]')", ss.str().c_str(), filename.c_str());
            }
            lws.get()[i] = support::cpp11::stoi(token);
        }

        // If all dimensions are 0: reset to NullRange (i.e nullptr)
        if(lws[0] == 0 && lws[1] == 0 && lws[2] == 0)
        {
            lws = cl::NullRange;
        }
        add_tuning_params(kernel_id, lws);
    }
    fs.close();
    _tuning_info.tune_lws = true;
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
    for(auto const &kernel_data : _tuning_params_table)
    {
        const cl::NDRange lws = CLTuningParams(kernel_data.second).get_lws();
        fs << kernel_data.first << ";" << lws[0] << ";" << lws[1] << ";" << lws[2] << std::endl;
    }
    fs.close();
    return true;
}
} // namespace arm_compute
