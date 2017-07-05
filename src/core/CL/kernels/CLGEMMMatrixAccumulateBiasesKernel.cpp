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
#include "arm_compute/core/CL/kernels/CLGEMMMatrixAccumulateBiasesKernel.h"

#include "arm_compute/core/AccessWindowStatic.h"
#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"

using namespace arm_compute;

CLGEMMMatrixAccumulateBiasesKernel::CLGEMMMatrixAccumulateBiasesKernel()
    : _accum(nullptr), _biases(nullptr)
{
}

void CLGEMMMatrixAccumulateBiasesKernel::configure(ICLTensor *accum, const ICLTensor *biases)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(accum, 1, DataType::QS8, DataType::QS16, DataType::F16, DataType::F32);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(biases, accum);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_FIXED_POINT(biases, accum);
    ARM_COMPUTE_ERROR_ON(biases->info()->num_dimensions() != 1);

    _biases = biases;
    _accum  = accum;

    std::set<std::string> build_opts;
    build_opts.insert(("-DDATA_TYPE=" + get_cl_type_from_data_type(accum->info()->data_type())));
    if(is_data_type_fixed_point(accum->info()->data_type()))
    {
        build_opts.emplace("-DFIXED_POINT_POSITION=" + support::cpp11::to_string(accum->info()->fixed_point_position()));
    }

    // Create kernel
    _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel("gemm_accumulate_biases", build_opts));

    // Configure kernel window
    const unsigned int num_elems_processed_per_iteration = 16;

    Window win = calculate_max_window(*_accum->info(), Steps(num_elems_processed_per_iteration));

    AccessWindowStatic     biases_access(biases->info(), 0, 0, biases->info()->dimension(0), biases->info()->dimension(1));
    AccessWindowHorizontal accum_access(_accum->info(), 0, num_elems_processed_per_iteration);

    update_window_and_padding(win, biases_access, accum_access);

    ICLKernel::configure(win);
}

void CLGEMMMatrixAccumulateBiasesKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_WINDOWS(ICLKernel::window(), window);

    Window accum_slice = window.first_slice_window_2D();

    Window biases_slice(accum_slice);
    biases_slice.set(Window::DimY, Window::Dimension(0, 1, 1));

    // Run kernel
    do
    {
        // Set arguments
        unsigned int idx = 0;
        add_2D_tensor_argument(idx, _accum, accum_slice);
        add_1D_tensor_argument(idx, _biases, biases_slice);

        enqueue(queue, *this, accum_slice);
    }
    while(window.slide_window_slice_2D(accum_slice));
}
