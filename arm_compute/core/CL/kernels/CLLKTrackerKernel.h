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
#ifndef __ARM_COMPUTE_CLLKTRACKERKERNEL_H__
#define __ARM_COMPUTE_CLLKTRACKERKERNEL_H__

#include "arm_compute/core/CL/ICLArray.h"
#include "arm_compute/core/CL/ICLKernel.h"
#include "arm_compute/core/Types.h"

#include <cstddef>
#include <cstdint>

namespace arm_compute
{
class ICLTensor;

/** Internal keypoint structure for Lucas-Kanade Optical Flow */
struct CLLKInternalKeypoint
{
    float x{ 0.f };               /**< x coordinate of the keypoint */
    float y{ 0.f };               /**< y coordinate of the keypoint */
    float tracking_status{ 0.f }; /**< the tracking status of the keypoint */
    float dummy{ 0.f };           /**< Dummy field, to make sure the data structure 128-bit align, so that GPU can use vload4 */
};

/** Structure for storing Spatial Gradient Matrix and the minimum eigenvalue for each keypoint */
struct CLCoefficientTable
{
    float A11;     /**< iA11 * FLT_SCALE */
    float A12;     /**< iA11 * FLT_SCALE */
    float A22;     /**< iA11 * FLT_SCALE */
    float min_eig; /**< Minimum eigenvalue */
};

/** Structure for storing ival, ixval and iyval for each point inside the window */
struct CLOldValue
{
    int16_t ival;  /**< ival extracts from old image */
    int16_t ixval; /**< ixval extracts from scharr Gx image */
    int16_t iyval; /**< iyval extracts from scharr Gy image */
    int16_t dummy; /**< Dummy field, to make sure the data structure 128-bit align, so that GPU can use vload4 */
};

using ICLLKInternalKeypointArray = ICLArray<CLLKInternalKeypoint>;
using ICLCoefficientTableArray   = ICLArray<CLCoefficientTable>;
using ICLOldValArray             = ICLArray<CLOldValue>;

/** Interface to run the initialization step of LKTracker */
class CLLKTrackerInitKernel : public ICLKernel
{
public:
    /** Initialise the kernel input and output
     *
     * @param[in]  old_points           Pointer to the @ref ICLKeyPointArray storing old key points
     * @param[in]  new_points_estimates Pointer to the @ref ICLKeyPointArray storing new estimates key points
     * @param[out] old_points_internal  Pointer to the array of internal @ref CLLKInternalKeypoint old points
     * @param[out] new_points_internal  Pointer to the array of internal @ref CLLKInternalKeypoint new points
     * @param[in]  use_initial_estimate The flag to indicate whether the initial estimated position should be used
     * @param[in]  level                The pyramid level
     * @param[in]  num_levels           The number of pyramid levels
     * @param[in]  pyramid_scale        Scale factor used for generating the pyramid
     */
    void configure(const ICLKeyPointArray *old_points, const ICLKeyPointArray *new_points_estimates,
                   ICLLKInternalKeypointArray *old_points_internal, ICLLKInternalKeypointArray *new_points_internal,
                   bool use_initial_estimate, size_t level, size_t num_levels, float pyramid_scale);

    // Inherited methods overridden:
    void run(const Window &window, cl::CommandQueue &queue) override;
};

/** Interface to run the finalize step of LKTracker, where it truncates the coordinates stored in new_points array */
class CLLKTrackerFinalizeKernel : public ICLKernel
{
public:
    /** Initialise the kernel input and output
     *
     * @param[in]  new_points_internal Pointer to the array of internal @ref CLLKInternalKeypoint new points
     * @param[out] new_points          Pointer to the @ref ICLKeyPointArray storing new key points
     */
    void configure(ICLLKInternalKeypointArray *new_points_internal, ICLKeyPointArray *new_points);

    // Inherited methods overridden:
    void run(const Window &window, cl::CommandQueue &queue) override;
};

/** Interface to run the first stage of LKTracker, where A11, A12, A22, min_eig, ival, ixval and iyval are computed */
class CLLKTrackerStage0Kernel : public ICLKernel
{
public:
    /** Default constructor */
    CLLKTrackerStage0Kernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLLKTrackerStage0Kernel(const CLLKTrackerStage0Kernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLLKTrackerStage0Kernel &operator=(const CLLKTrackerStage0Kernel &) = delete;
    /** Allow instances of this class to be moved */
    CLLKTrackerStage0Kernel(CLLKTrackerStage0Kernel &&) = default;
    /** Allow instances of this class to be moved */
    CLLKTrackerStage0Kernel &operator=(CLLKTrackerStage0Kernel &&) = default;
    /** Initialise the kernel input and output
     *
     * @param[in]      old_input           Pointer to the input old tensor. Data types supported: U8
     * @param[in]      old_scharr_gx       Pointer to the input scharr X tensor. Data types supported: S16
     * @param[in]      old_scharr_gy       Pointer to the input scharr Y tensor. Data types supported: S16
     * @param[in]      old_points_internal Pointer to the array of CLLKInternalKeypoint old points
     * @param[in, out] new_points_internal Pointer to the array of CLLKInternalKeypoint new points
     * @param[out]     coeff_table         Pointer to the array holding the Spatial Gradient coefficients
     * @param[out]     old_ival            Pointer to the array holding internal values
     * @param[in]      window_dimension    The size of the window on which to perform the algorithm
     * @param[in]      level               The pyramid level
     */
    void configure(const ICLTensor *old_input, const ICLTensor *old_scharr_gx, const ICLTensor *old_scharr_gy,
                   ICLLKInternalKeypointArray *old_points_internal, ICLLKInternalKeypointArray *new_points_internal,
                   ICLCoefficientTableArray *coeff_table, ICLOldValArray *old_ival,
                   size_t window_dimension, size_t level);

    // Inherited methods overridden:
    void run(const Window &window, cl::CommandQueue &queue) override;

private:
    const ICLTensor *_old_input;
    const ICLTensor *_old_scharr_gx;
    const ICLTensor *_old_scharr_gy;
};

/** Interface to run the second stage of LKTracker, where the motion vectors of the given points are computed */
class CLLKTrackerStage1Kernel : public ICLKernel
{
public:
    /** Default constructor */
    CLLKTrackerStage1Kernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLLKTrackerStage1Kernel(const CLLKTrackerStage1Kernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLLKTrackerStage1Kernel &operator=(const CLLKTrackerStage1Kernel &) = delete;
    /** Allow instances of this class to be moved */
    CLLKTrackerStage1Kernel(CLLKTrackerStage1Kernel &&) = default;
    /** Allow instances of this class to be moved */
    CLLKTrackerStage1Kernel &operator=(CLLKTrackerStage1Kernel &&) = default;
    /** Initialise the kernel input and output
     *
     * @param[in]      new_input           Pointer to the input new tensor. Data types supported: U8
     * @param[in, out] new_points_internal Pointer to the array of CLLKInternalKeypoint for new points
     * @param[in]      coeff_table         Pointer to the array holding the Spatial Gradient coefficients
     * @param[in]      old_ival            Pointer to the array holding internal values
     * @param[in]      termination         The criteria to terminate the search of each keypoint.
     * @param[in]      epsilon             The error for terminating the algorithm
     * @param[in]      num_iterations      The maximum number of iterations before terminating the algorithm
     * @param[in]      window_dimension    The size of the window on which to perform the algorithm
     * @param[in]      level               The pyramid level
     */
    void configure(const ICLTensor *new_input, ICLLKInternalKeypointArray *new_points_internal, ICLCoefficientTableArray *coeff_table, ICLOldValArray *old_ival,
                   Termination termination, float epsilon, size_t num_iterations, size_t window_dimension, size_t level);

    // Inherited methods overridden:
    void run(const Window &window, cl::CommandQueue &queue) override;

private:
    const ICLTensor *_new_input;
};
} // namespace arm_compute
#endif /*__ARM_COMPUTE_CLLKTRACKERKERNEL_H__ */
