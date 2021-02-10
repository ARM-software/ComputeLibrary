/*
 * Copyright (c) 2016-2021 Arm Limited.
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
#ifndef ARM_COMPUTE_LKTRACKERKERNEL_H
#define ARM_COMPUTE_LKTRACKERKERNEL_H

#include "arm_compute/core/IArray.h"
#include "arm_compute/core/Types.h"
#include "src/core/NEON/INEKernel.h"

#include <cstddef>
#include <cstdint>
#include <tuple>
#include <utility>

namespace arm_compute
{
class ITensor;

/** Interface for Neon Array of Internal Key Points. */
using INELKInternalKeypointArray = IArray<NELKInternalKeypoint>;

/** Interface for the Lucas-Kanade tracker kernel */
class NELKTrackerKernel : public INEKernel
{
public:
    const char *name() const override
    {
        return "NELKTrackerKernel";
    }
    /** Default constructor */
    NELKTrackerKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NELKTrackerKernel(const NELKTrackerKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NELKTrackerKernel &operator=(const NELKTrackerKernel &) = delete;
    /** Allow instances of this class to be moved */
    NELKTrackerKernel(NELKTrackerKernel &&) = default;
    /** Allow instances of this class to be moved */
    NELKTrackerKernel &operator=(NELKTrackerKernel &&) = default;
    /** Default destructor */
    ~NELKTrackerKernel() = default;

    /** Initialise the kernel input and output
     *
     * @param[in]      input_old            Pointer to the input old tensor. Data type supported: U8
     * @param[in]      input_new            Pointer to the input new tensor. Data type supported. U8
     * @param[in]      old_scharr_gx        Pointer to the input scharr X tensor. Data type supported: S16
     * @param[in]      old_scharr_gy        Pointer to the input scharr Y tensor. Data type supported: S16
     * @param[in]      old_points           Pointer to the IKeyPointArray storing old key points
     * @param[in]      new_points_estimates Pointer to the IKeyPointArray storing new estimates key points
     * @param[out]     new_points           Pointer to the IKeyPointArray storing new key points
     * @param[in, out] old_points_internal  Pointer to the array of NELKInternalKeypoint for old points
     * @param[out]     new_points_internal  Pointer to the array of NELKInternalKeypoint for new points
     * @param[in]      termination          The criteria to terminate the search of each keypoint.
     * @param[in]      use_initial_estimate The flag to indicate whether the initial estimated position should be used
     * @param[in]      epsilon              The error for terminating the algorithm
     * @param[in]      num_iterations       The maximum number of iterations before terminate the algorithm
     * @param[in]      window_dimension     The size of the window on which to perform the algorithm
     * @param[in]      level                The pyramid level
     * @param[in]      num_levels           The number of pyramid levels
     * @param[in]      pyramid_scale        Scale factor used for generating the pyramid
     */
    void configure(const ITensor *input_old, const ITensor *input_new, const ITensor *old_scharr_gx, const ITensor *old_scharr_gy,
                   const IKeyPointArray *old_points, const IKeyPointArray *new_points_estimates, IKeyPointArray *new_points,
                   INELKInternalKeypointArray *old_points_internal, INELKInternalKeypointArray *new_points_internal,
                   Termination termination, bool use_initial_estimate, float epsilon, unsigned int num_iterations, size_t window_dimension,
                   size_t level, size_t num_levels, float pyramid_scale);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;
    BorderSize border_size() const override;

private:
    /** Initialise the array of keypoints in the provide range
     *
     * @param[in] start Index of first element in the keypoints array to be initialised
     * @param[in] end   Index after last elelemnt in the keypoints array to be initialised
     */
    void init_keypoints(int start, int end);
    /** Compute the structure tensor A^T * A based on the scharr gradients I_x and I_y
     *
     * @param[in]  keypoint    Keypoint for which gradients are computed
     * @param[out] bilinear_ix Intermediate interpolated data for X gradient
     * @param[out] bilinear_iy Intermediate interpolated data for Y gradient
     *
     * @return Values A11, A12, A22
     */
    std::tuple<int, int, int> compute_spatial_gradient_matrix(const NELKInternalKeypoint &keypoint, int32_t *bilinear_ix, int32_t *bilinear_iy);
    /** Compute the vector A^T * b, i.e. -sum(I_d * I_t) for d in {x,y}
     *
     * @param[in] old_keypoint Old keypoint for which gradient is computed
     * @param[in] new_keypoint New keypoint for which gradient is computed
     * @param[in] bilinear_ix  Intermediate interpolated data for X gradient
     * @param[in] bilinear_iy  Intermediate interpolated data for Y gradient
     *
     * @return Values b1, b2
     */
    std::pair<int, int> compute_image_mismatch_vector(const NELKInternalKeypoint &old_keypoint, const NELKInternalKeypoint &new_keypoint, const int32_t *bilinear_ix, const int32_t *bilinear_iy);

    const ITensor              *_input_old;
    const ITensor              *_input_new;
    const ITensor              *_old_scharr_gx;
    const ITensor              *_old_scharr_gy;
    IKeyPointArray             *_new_points;
    const IKeyPointArray       *_new_points_estimates;
    const IKeyPointArray       *_old_points;
    INELKInternalKeypointArray *_old_points_internal;
    INELKInternalKeypointArray *_new_points_internal;
    Termination                 _termination;
    bool                        _use_initial_estimate;
    float                       _pyramid_scale;
    float                       _epsilon;
    unsigned int                _num_iterations;
    int                         _window_dimension;
    unsigned int                _level;
    unsigned int                _num_levels;
    ValidRegion                 _valid_region;
};
} // namespace arm_compute
#endif /*ARM_COMPUTE_NELKTRACKERKERNEL_H */
