/*
 * Copyright (c) 2016, 2017 ARM Limited.
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
#include "arm_compute/core/CL/kernels/CLMagnitudePhaseKernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

#include <set>
#include <string>

using namespace arm_compute;

CLMagnitudePhaseKernel::CLMagnitudePhaseKernel()
    : _gx(nullptr), _gy(nullptr), _magnitude(nullptr), _phase(nullptr), _run_mag(false), _run_phase(false)
{
}

void CLMagnitudePhaseKernel::configure(const ICLTensor *gx, const ICLTensor *gy, ICLTensor *magnitude, ICLTensor *phase,
                                       MagnitudeType mag_type, PhaseType phase_type)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(gx, 1, DataType::S16, DataType::S32);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(gy, 1, DataType::S16, DataType::S32);
    ARM_COMPUTE_ERROR_ON((magnitude == nullptr) && (phase == nullptr));
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(gx, gy);

    _run_mag   = (magnitude != nullptr);
    _run_phase = (phase != nullptr);
    if(_run_mag)
    {
        ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(magnitude, 1, DataType::S16, DataType::S32);
        ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(gx, magnitude);
    }
    if(_run_phase)
    {
        ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(phase, 1, DataType::U8);
    }

    if(!_run_mag && !_run_phase)
    {
        ARM_COMPUTE_ERROR("At least one output must be NOT NULL");
    }

    _gx        = gx;
    _gy        = gy;
    _magnitude = magnitude;
    _phase     = phase;

    // Construct kernel name
    std::set<std::string> build_opts = {};

    // Add magnitude type
    if(_run_mag)
    {
        switch(mag_type)
        {
            case MagnitudeType::L1NORM:
                build_opts.insert("-DMAGNITUDE=1");
                break;
            case MagnitudeType::L2NORM:
                build_opts.insert("-DMAGNITUDE=2");
                break;
            default:
                ARM_COMPUTE_ERROR("Unsupported magnitude calculation type.");
                build_opts.insert("-DMAGNITUDE=0");
                break;
        }
    }

    // Add phase type
    if(_run_phase)
    {
        switch(phase_type)
        {
            case PhaseType::UNSIGNED:
                build_opts.insert("-DPHASE=1");
                break;
            case PhaseType::SIGNED:
                build_opts.insert("-DPHASE=2");
                break;
            default:
                ARM_COMPUTE_ERROR("Unsupported phase calculation type.");
                build_opts.insert("-DPHASE=0");
                break;
        }
    }

    // Add data_type
    build_opts.insert("-DDATA_TYPE=" + get_cl_type_from_data_type(gx->info()->data_type()));

    // Create kernel
    _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel("magnitude_phase", build_opts));

    // Configure kernel window
    constexpr unsigned int num_elems_processed_per_iteration = 16;

    Window win = calculate_max_window(*gx->info(), Steps(num_elems_processed_per_iteration));

    AccessWindowHorizontal gx_access(gx->info(), 0, num_elems_processed_per_iteration);
    AccessWindowHorizontal gy_access(gy->info(), 0, num_elems_processed_per_iteration);
    AccessWindowHorizontal output_magnitude_access(magnitude == nullptr ? nullptr : magnitude->info(), 0, num_elems_processed_per_iteration);
    AccessWindowHorizontal output_phase_access(phase == nullptr ? nullptr : phase->info(), 0, num_elems_processed_per_iteration);

    update_window_and_padding(win,
                              gx_access, gy_access,
                              output_magnitude_access, output_phase_access);

    ValidRegion valid_region = intersect_valid_regions(gx->info()->valid_region(),
                                                       gy->info()->valid_region());
    output_magnitude_access.set_valid_region(win, valid_region);
    output_phase_access.set_valid_region(win, valid_region);

    ICLKernel::configure(win);
}

void CLMagnitudePhaseKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

    Window slice = window.first_slice_window_2D();
    do
    {
        unsigned int idx = 0;
        add_2D_tensor_argument(idx, _gx, slice);
        add_2D_tensor_argument(idx, _gy, slice);

        if(_run_mag)
        {
            add_2D_tensor_argument(idx, _magnitude, slice);
        }

        if(_run_phase)
        {
            add_2D_tensor_argument(idx, _phase, slice);
        }

        enqueue(queue, *this, slice);
    }
    while(window.slide_window_slice_2D(slice));
}
