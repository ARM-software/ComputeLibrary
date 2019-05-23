/*
 * Copyright (c) 2017-2019 ARM Limited.
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
#include "arm_compute/runtime/CL/functions/CLHOGMultiDetection.h"

#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/runtime/CL/CLArray.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/Scheduler.h"
#include "support/ToolchainSupport.h"

using namespace arm_compute;

CLHOGMultiDetection::CLHOGMultiDetection(std::shared_ptr<IMemoryManager> memory_manager) // NOLINT
    : _memory_group(std::move(memory_manager)),
      _gradient_kernel(),
      _orient_bin_kernel(),
      _block_norm_kernel(),
      _hog_detect_kernel(),
      _non_maxima_kernel(),
      _hog_space(),
      _hog_norm_space(),
      _detection_windows(),
      _mag(),
      _phase(),
      _non_maxima_suppression(false),
      _num_orient_bin_kernel(0),
      _num_block_norm_kernel(0),
      _num_hog_detect_kernel(0)
{
}

void CLHOGMultiDetection::configure(ICLTensor *input, const ICLMultiHOG *multi_hog, ICLDetectionWindowArray *detection_windows, ICLSize2DArray *detection_window_strides, BorderMode border_mode,
                                    uint8_t constant_border_value, float threshold, bool non_maxima_suppression, float min_distance)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8);
    ARM_COMPUTE_ERROR_ON_INVALID_MULTI_HOG(multi_hog);
    ARM_COMPUTE_ERROR_ON(nullptr == detection_windows);
    ARM_COMPUTE_ERROR_ON(detection_window_strides->num_values() != multi_hog->num_models());

    const size_t       width      = input->info()->dimension(Window::DimX);
    const size_t       height     = input->info()->dimension(Window::DimY);
    const TensorShape &shape_img  = input->info()->tensor_shape();
    const size_t       num_models = multi_hog->num_models();
    PhaseType          phase_type = multi_hog->model(0)->info()->phase_type();

    size_t prev_num_bins     = multi_hog->model(0)->info()->num_bins();
    Size2D prev_cell_size    = multi_hog->model(0)->info()->cell_size();
    Size2D prev_block_size   = multi_hog->model(0)->info()->block_size();
    Size2D prev_block_stride = multi_hog->model(0)->info()->block_stride();

    /* Check if CLHOGOrientationBinningKernel and CLHOGBlockNormalizationKernel kernels can be skipped for a specific HOG data-object
     *
     * 1) CLHOGOrientationBinningKernel and CLHOGBlockNormalizationKernel are skipped if the cell size and the number of bins don't change.
     *        Since "multi_hog" is sorted,it is enough to check the HOG descriptors at level "ith" and level "(i-1)th
     * 2) CLHOGBlockNormalizationKernel is skipped if the cell size, the number of bins and block size do not change.
     *         Since "multi_hog" is sorted,it is enough to check the HOG descriptors at level "ith" and level "(i-1)th
     *
     * @note Since the orientation binning and block normalization kernels can be skipped, we need to keep track of the input to process for each kernel
     *       with "input_orient_bin", "input_hog_detect" and "input_block_norm"
     */
    std::vector<size_t> input_orient_bin;
    std::vector<size_t> input_hog_detect;
    std::vector<std::pair<size_t, size_t>> input_block_norm;

    input_orient_bin.push_back(0);
    input_hog_detect.push_back(0);
    input_block_norm.emplace_back(0, 0);

    for(size_t i = 1; i < num_models; ++i)
    {
        size_t cur_num_bins     = multi_hog->model(i)->info()->num_bins();
        Size2D cur_cell_size    = multi_hog->model(i)->info()->cell_size();
        Size2D cur_block_size   = multi_hog->model(i)->info()->block_size();
        Size2D cur_block_stride = multi_hog->model(i)->info()->block_stride();

        if((cur_num_bins != prev_num_bins) || (cur_cell_size.width != prev_cell_size.width) || (cur_cell_size.height != prev_cell_size.height))
        {
            prev_num_bins     = cur_num_bins;
            prev_cell_size    = cur_cell_size;
            prev_block_size   = cur_block_size;
            prev_block_stride = cur_block_stride;

            // Compute orientation binning and block normalization kernels. Update input to process
            input_orient_bin.push_back(i);
            input_block_norm.emplace_back(i, input_orient_bin.size() - 1);
        }
        else if((cur_block_size.width != prev_block_size.width) || (cur_block_size.height != prev_block_size.height) || (cur_block_stride.width != prev_block_stride.width)
                || (cur_block_stride.height != prev_block_stride.height))
        {
            prev_block_size   = cur_block_size;
            prev_block_stride = cur_block_stride;

            // Compute block normalization kernel. Update input to process
            input_block_norm.emplace_back(i, input_orient_bin.size() - 1);
        }

        // Update input to process for hog detector kernel
        input_hog_detect.push_back(input_block_norm.size() - 1);
    }

    _detection_windows      = detection_windows;
    _non_maxima_suppression = non_maxima_suppression;
    _num_orient_bin_kernel  = input_orient_bin.size(); // Number of CLHOGOrientationBinningKernel kernels to compute
    _num_block_norm_kernel  = input_block_norm.size(); // Number of CLHOGBlockNormalizationKernel kernels to compute
    _num_hog_detect_kernel  = input_hog_detect.size(); // Number of CLHOGDetector functions to compute

    _orient_bin_kernel.resize(_num_orient_bin_kernel);
    _block_norm_kernel.resize(_num_block_norm_kernel);
    _hog_detect_kernel.resize(_num_hog_detect_kernel);
    _hog_space.resize(_num_orient_bin_kernel);
    _hog_norm_space.resize(_num_block_norm_kernel);

    // Allocate tensors for magnitude and phase
    TensorInfo info_mag(shape_img, Format::S16);
    _mag.allocator()->init(info_mag);

    TensorInfo info_phase(shape_img, Format::U8);
    _phase.allocator()->init(info_phase);

    // Manage intermediate buffers
    _memory_group.manage(&_mag);
    _memory_group.manage(&_phase);

    // Initialise gradient kernel
    _gradient_kernel.configure(input, &_mag, &_phase, phase_type, border_mode, constant_border_value);

    // Configure NETensor for the HOG space and orientation binning kernel
    for(size_t i = 0; i < _num_orient_bin_kernel; ++i)
    {
        const size_t idx_multi_hog = input_orient_bin[i];

        // Get the corresponding cell size and number of bins
        const Size2D &cell     = multi_hog->model(idx_multi_hog)->info()->cell_size();
        const size_t  num_bins = multi_hog->model(idx_multi_hog)->info()->num_bins();

        // Calculate number of cells along the x and y directions for the hog_space
        const size_t num_cells_x = width / cell.width;
        const size_t num_cells_y = height / cell.height;

        // TensorShape of hog space
        TensorShape shape_hog_space = input->info()->tensor_shape();
        shape_hog_space.set(Window::DimX, num_cells_x);
        shape_hog_space.set(Window::DimY, num_cells_y);

        // Allocate HOG space
        TensorInfo info_space(shape_hog_space, num_bins, DataType::F32);
        _hog_space[i].allocator()->init(info_space);

        // Manage intermediate buffers
        _memory_group.manage(&_hog_space[i]);

        // Initialise orientation binning kernel
        _orient_bin_kernel[i].configure(&_mag, &_phase, &_hog_space[i], multi_hog->model(idx_multi_hog)->info());
    }

    // Allocate intermediate tensors
    _mag.allocator()->allocate();
    _phase.allocator()->allocate();

    // Configure CLTensor for the normalized HOG space and block normalization kernel
    for(size_t i = 0; i < _num_block_norm_kernel; ++i)
    {
        const size_t idx_multi_hog  = input_block_norm[i].first;
        const size_t idx_orient_bin = input_block_norm[i].second;

        // Allocate normalized HOG space
        TensorInfo tensor_info(*(multi_hog->model(idx_multi_hog)->info()), width, height);
        _hog_norm_space[i].allocator()->init(tensor_info);

        // Manage intermediate buffers
        _memory_group.manage(&_hog_norm_space[i]);

        // Initialize block normalization kernel
        _block_norm_kernel[i].configure(&_hog_space[idx_orient_bin], &_hog_norm_space[i], multi_hog->model(idx_multi_hog)->info());
    }

    // Allocate intermediate tensors
    for(size_t i = 0; i < _num_orient_bin_kernel; ++i)
    {
        _hog_space[i].allocator()->allocate();
    }

    detection_window_strides->map(CLScheduler::get().queue(), true);

    // Configure HOG detector kernel
    for(size_t i = 0; i < _num_hog_detect_kernel; ++i)
    {
        const size_t idx_block_norm = input_hog_detect[i];

        _hog_detect_kernel[i].configure(&_hog_norm_space[idx_block_norm], multi_hog->cl_model(i), detection_windows, detection_window_strides->at(i), threshold, i);
    }

    detection_window_strides->unmap(CLScheduler::get().queue());

    // Configure non maxima suppression kernel
    _non_maxima_kernel.configure(_detection_windows, min_distance);

    // Allocate intermediate tensors
    for(size_t i = 0; i < _num_block_norm_kernel; ++i)
    {
        _hog_norm_space[i].allocator()->allocate();
    }
}

void CLHOGMultiDetection::run()
{
    ARM_COMPUTE_ERROR_ON_MSG(_detection_windows == nullptr, "Unconfigured function");

    MemoryGroupResourceScope scope_mg(_memory_group);

    // Reset detection window
    _detection_windows->clear();

    // Run gradient
    _gradient_kernel.run();

    // Run orientation binning kernel
    for(size_t i = 0; i < _num_orient_bin_kernel; ++i)
    {
        CLScheduler::get().enqueue(_orient_bin_kernel[i], false);
    }

    // Run block normalization kernel
    for(size_t i = 0; i < _num_block_norm_kernel; ++i)
    {
        CLScheduler::get().enqueue(_block_norm_kernel[i], false);
    }

    // Run HOG detector kernel
    for(size_t i = 0; i < _num_hog_detect_kernel; ++i)
    {
        _hog_detect_kernel[i].run();
    }

    // Run non-maxima suppression kernel if enabled
    if(_non_maxima_suppression)
    {
        // Map detection windows array before computing non maxima suppression
        _detection_windows->map(CLScheduler::get().queue(), true);
        Scheduler::get().schedule(&_non_maxima_kernel, Window::DimY);
        _detection_windows->unmap(CLScheduler::get().queue());
    }
}
