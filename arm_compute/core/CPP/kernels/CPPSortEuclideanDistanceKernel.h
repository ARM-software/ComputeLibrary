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
#ifndef __ARM_COMPUTE_CPPSORTEUCLIDEANDISTANCEKERNEL_H__
#define __ARM_COMPUTE_CPPSORTEUCLIDEANDISTANCEKERNEL_H__

#include "arm_compute/core/CPP/ICPPKernel.h"
#include "arm_compute/core/IArray.h"

#include <cstdint>
#include <mutex>

namespace arm_compute
{
/** CPP kernel to perform sorting and euclidean distance */
class CPPSortEuclideanDistanceKernel : public ICPPKernel
{
public:
    /** Default constructor */
    CPPSortEuclideanDistanceKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CPPSortEuclideanDistanceKernel(const CPPSortEuclideanDistanceKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CPPSortEuclideanDistanceKernel &operator=(const CPPSortEuclideanDistanceKernel &) = delete;
    /** Allow instances of this class to be moved */
    CPPSortEuclideanDistanceKernel(CPPSortEuclideanDistanceKernel &&) = default;
    /** Allow instances of this class to be moved */
    CPPSortEuclideanDistanceKernel &operator=(CPPSortEuclideanDistanceKernel &&) = default;
    /** Initialise the kernel's source, destination and border mode.
     *
     * @param[in,out] in_out                Input internal keypoints. Marked as out as the kernel writes 0 in the strength member.
     * @param[out]    output                Output keypoints.
     * @param[in]     num_corner_candidates Pointer to the number of corner candidates in the input array
     * @param[in]     min_distance          Radial Euclidean distance to use
     */
    void configure(InternalKeypoint *in_out, IKeyPointArray *output, const int32_t *num_corner_candidates, float min_distance);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;
    bool is_parallelisable() const override;

private:
    const int32_t    *_num_corner_candidates; /**< Number of corner candidates */
    float             _min_distance;          /**< Radial Euclidean distance */
    InternalKeypoint *_in_out;                /**< Source array of InternalKeypoint */
    IKeyPointArray   *_output;                /**< Destination array of IKeyPointArray */
};
} // namespace arm_compute
#endif /* __ARM_COMPUTE_CPPSORTEUCLIDEANDISTANCEKERNEL_H__ */
