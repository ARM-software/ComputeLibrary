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
#ifndef __ARM_COMPUTE_CLOPTICALFLOW_H__
#define __ARM_COMPUTE_CLOPTICALFLOW_H__

#include "arm_compute/core/CL/kernels/CLLKTrackerKernel.h"

#include "arm_compute/core/IArray.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/CLArray.h"
#include "arm_compute/runtime/CL/CLMemoryGroup.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/functions/CLScharr3x3.h"
#include "arm_compute/runtime/IFunction.h"
#include "arm_compute/runtime/IMemoryManager.h"

#include <cstddef>
#include <cstdint>
#include <memory>

namespace arm_compute
{
class CLPyramid;

/** OpenCL Array of Internal Keypoints */
using CLLKInternalKeypointArray = CLArray<CLLKInternalKeypoint>;
/** OpenCL Array of Coefficient Tables */
using CLCoefficientTableArray = CLArray<CLCoefficientTable>;
/** OpenCL Array of Old Values */
using CLOldValueArray = CLArray<CLOldValue>;

/** Basic function to execute optical flow. This function calls the following OpenCL kernels and functions:
 *
 * -# @ref CLScharr3x3
 * -# @ref CLLKTrackerInitKernel
 * -# @ref CLLKTrackerStage0Kernel
 * -# @ref CLLKTrackerStage1Kernel
 * -# @ref CLLKTrackerFinalizeKernel
 */
class CLOpticalFlow : public IFunction
{
public:
    /** Default constructor */
    CLOpticalFlow(std::shared_ptr<IMemoryManager> memory_manager = nullptr);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLOpticalFlow(const CLOpticalFlow &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLOpticalFlow &operator=(const CLOpticalFlow &) = delete;
    /** Allow instances of this class to be moved */
    CLOpticalFlow(CLOpticalFlow &&) = default;
    /** Allow instances of this class to be moved */
    CLOpticalFlow &operator=(CLOpticalFlow &&) = default;
    /**  Initialise the function input and output
     *
     * @param[in]  old_pyramid           Pointer to the pyramid for the old tensor. Data types supported U8
     * @param[in]  new_pyramid           Pointer to the pyramid for the new tensor. Data types supported U8
     * @param[in]  old_points            Pointer to the IKeyPointArray storing old key points
     * @param[in]  new_points_estimates  Pointer to the IKeyPointArray storing new estimates key points
     * @param[out] new_points            Pointer to the IKeyPointArray storing new key points
     * @param[in]  termination           The criteria to terminate the search of each keypoint.
     * @param[in]  epsilon               The error for terminating the algorithm
     * @param[in]  num_iterations        The maximum number of iterations before terminate the alogrithm
     * @param[in]  window_dimension      The size of the window on which to perform the algorithm
     * @param[in]  use_initial_estimate  The flag to indicate whether the initial estimated position should be used
     * @param[in]  border_mode           The border mode applied at scharr kernel stage
     * @param[in]  constant_border_value (Optional) Constant value to use for borders if border_mode is set to CONSTANT
     *
     */
    void configure(const CLPyramid *old_pyramid, const CLPyramid *new_pyramid,
                   const ICLKeyPointArray *old_points, const ICLKeyPointArray *new_points_estimates, ICLKeyPointArray *new_points,
                   Termination termination, float epsilon, size_t num_iterations, size_t window_dimension, bool use_initial_estimate,
                   BorderMode border_mode, uint8_t constant_border_value = 0);

    // Inherited methods overridden:
    void run() override;

private:
    CLMemoryGroup                              _memory_group;
    std::vector<CLLKTrackerInitKernel>         _tracker_init_kernel;
    std::vector<CLLKTrackerStage0Kernel>       _tracker_stage0_kernel;
    std::vector<CLLKTrackerStage1Kernel>       _tracker_stage1_kernel;
    CLLKTrackerFinalizeKernel                  _tracker_finalize_kernel;
    std::vector<CLScharr3x3>                   _func_scharr;
    std::vector<CLTensor>                      _scharr_gx;
    std::vector<CLTensor>                      _scharr_gy;
    const ICLKeyPointArray                    *_old_points;
    const ICLKeyPointArray                    *_new_points_estimates;
    ICLKeyPointArray                          *_new_points;
    std::unique_ptr<CLLKInternalKeypointArray> _old_points_internal;
    std::unique_ptr<CLLKInternalKeypointArray> _new_points_internal;
    std::unique_ptr<CLCoefficientTableArray>   _coefficient_table;
    std::unique_ptr<CLOldValueArray>           _old_values;
    size_t                                     _num_levels;
};
}
#endif /*__ARM_COMPUTE_CLOPTICALFLOW_H__ */
