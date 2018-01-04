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
#include "arm_compute/core/GLES_COMPUTE/kernels/GCGEMMMatrixAccumulateBiasesKernel.h"

#include "arm_compute/core/AccessWindowStatic.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/GLES_COMPUTE/GCHelpers.h"
#include "arm_compute/core/GLES_COMPUTE/GCKernelLibrary.h"
#include "arm_compute/core/GLES_COMPUTE/IGCTensor.h"
#include "arm_compute/core/GLES_COMPUTE/OpenGLES.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"

using namespace arm_compute;

GCGEMMMatrixAccumulateBiasesKernel::GCGEMMMatrixAccumulateBiasesKernel()
    : _accum(nullptr), _biases(nullptr), _lws(gles::NDRange(1U, 1U, 1U))
{
}

void GCGEMMMatrixAccumulateBiasesKernel::configure(IGCTensor *accum, const IGCTensor *biases)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(accum, 1, DataType::F16, DataType::F32);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(biases, accum);
    ARM_COMPUTE_ERROR_ON(biases->info()->num_dimensions() != 1);

    _biases = biases;
    _accum  = accum;

    std::set<std::string> build_opts;
    build_opts.emplace("#define LOCAL_SIZE_X " + support::cpp11::to_string(_lws[0]));
    build_opts.emplace("#define LOCAL_SIZE_Y " + support::cpp11::to_string(_lws[1]));
    build_opts.emplace("#define LOCAL_SIZE_Z " + support::cpp11::to_string(_lws[2]));

    // Create kernel
    build_opts.emplace("#define GEMM_ACCUMULATE_BIASES");

#define ACCUM_PROCESS_4X

#if defined(ACCUM_PROCESS_4X)
    build_opts.emplace("#define ACCUM_PROCESS_4X");
#elif defined(ACCUM_PROCESS_8X) /* ACCUM_PROCESS_4X */
    build_opts.emplace("#define ACCUM_PROCESS_8X");
#endif                          /* ACCUM_PROCESS_4X */
    std::string dt_name = (accum->info()->data_type() == DataType::F32) ? "DATA_TYPE_FP32" : "DATA_TYPE_FP16";
    build_opts.emplace(("#define " + dt_name));

    _kernel = GCKernelLibrary::get().create_kernel("gemm_accumulate_biases", build_opts);

    // Configure kernel window
    unsigned int num_elems_processed_per_iteration = 1;

    if(_accum->info()->data_type() == DataType::F32)
    {
        num_elems_processed_per_iteration = 16;
    }
    else if(_accum->info()->data_type() == DataType::F16)
    {
#if defined(ACCUM_PROCESS_4X)
        num_elems_processed_per_iteration = 4;
#elif defined(ACCUM_PROCESS_8X) /* ACCUM_PROCESS_4X */
        num_elems_processed_per_iteration = 8;
#endif                          /* ACCUM_PROCESS_4X */
    }

    const int  accum_width         = accum->info()->dimension(0);
    const int  accum_padding_right = ceil_to_multiple(accum_width, num_elems_processed_per_iteration * _lws[0]) - accum_width;
    BorderSize border              = BorderSize(0, accum_padding_right, 0, 0);

    Window win = calculate_max_enlarged_window(*_accum->info(), Steps(num_elems_processed_per_iteration), border);

    AccessWindowStatic biases_access(biases->info(), 0, 0, ceil_to_multiple(biases->info()->dimension(0), num_elems_processed_per_iteration * _lws[0]), biases->info()->dimension(1));
    AccessWindowStatic accum_access(_accum->info(), 0, 0, accum_width + accum_padding_right, _accum->info()->dimension(1));

    update_window_and_padding(win, biases_access, accum_access);

    IGCKernel::configure(win);
}

void GCGEMMMatrixAccumulateBiasesKernel::run(const Window &window)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_WINDOWS(IGCKernel::window(), window);

    _kernel.use();

    Window accum_slice = window.first_slice_window_2D();

    Window biases_slice(accum_slice);
    biases_slice.set(Window::DimY, Window::Dimension(0, 1, 1));

    // Run kernel
    do
    {
        // Set arguments
        unsigned int idx = 0;

        add_2D_tensor_argument(idx, _accum, 1, accum_slice);
        add_1D_tensor_argument(idx, _biases, 2, biases_slice);
        _kernel.update_shader_params();

        enqueue(*this, accum_slice, _lws);
    }
    while(window.slide_window_slice_2D(accum_slice));
}
