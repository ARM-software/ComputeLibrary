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
#ifndef __ARM_COMPUTE_NEOPTICALFLOW_H__
#define __ARM_COMPUTE_NEOPTICALFLOW_H__

#include "arm_compute/core/IArray.h"
#include "arm_compute/core/NEON/kernels/NELKTrackerKernel.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/Array.h"
#include "arm_compute/runtime/IFunction.h"
#include "arm_compute/runtime/IMemoryManager.h"
#include "arm_compute/runtime/MemoryGroup.h"
#include "arm_compute/runtime/NEON/functions/NEScharr3x3.h"
#include "arm_compute/runtime/Tensor.h"

#include <cstddef>
#include <cstdint>
#include <memory>

namespace arm_compute
{
class Pyramid;

using LKInternalKeypointArray = Array<NELKInternalKeypoint>;
/** Basic function to execute optical flow. This function calls the following NEON kernels and functions:
 *
 * -# @ref NEScharr3x3
 * -# @ref NELKTrackerKernel
 *
 */
class NEOpticalFlow : public IFunction
{
public:
    /** Constructor */
    NEOpticalFlow(std::shared_ptr<IMemoryManager> memory_manager = nullptr);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEOpticalFlow(const NEOpticalFlow &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEOpticalFlow &operator=(const NEOpticalFlow &) = delete;
    /**  Initialise the function input and output
     *
     * @param[in]  old_pyramid           Pointer to the pyramid for the old tensor. Data type supported U8
     * @param[in]  new_pyramid           Pointer to the pyramid for the new tensor. Data type supported U8
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
    void configure(const Pyramid *old_pyramid, const Pyramid *new_pyramid, const IKeyPointArray *old_points, const IKeyPointArray *new_points_estimates,
                   IKeyPointArray *new_points, Termination termination, float epsilon, unsigned int num_iterations, size_t window_dimension,
                   bool use_initial_estimate, BorderMode border_mode, uint8_t constant_border_value = 0);

    // Inherited methods overridden:
    void run() override;

private:
    MemoryGroup                          _memory_group;
    std::unique_ptr<NEScharr3x3[]>       _func_scharr;
    std::unique_ptr<NELKTrackerKernel[]> _kernel_tracker;
    std::unique_ptr<Tensor[]>            _scharr_gx;
    std::unique_ptr<Tensor[]>            _scharr_gy;
    IKeyPointArray                      *_new_points;
    const IKeyPointArray                *_new_points_estimates;
    const IKeyPointArray                *_old_points;
    LKInternalKeypointArray              _new_points_internal;
    LKInternalKeypointArray              _old_points_internal;
    unsigned int                         _num_levels;
};
}
#endif /*__ARM_COMPUTE_NEOPTICALFLOW_H__ */
