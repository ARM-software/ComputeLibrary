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
#ifndef __ARM_COMPUTE_CPPCORNERCANDIDATESKERNEL_H__
#define __ARM_COMPUTE_CPPCORNERCANDIDATESKERNEL_H__

#include "arm_compute/core/IArray.h"
#include "arm_compute/core/NEON/INEKernel.h"
#include "support/Mutex.h"

#include <cstdint>

namespace arm_compute
{
class ITensor;
using IImage = ITensor;

/** CPP kernel to perform corner candidates
 */
class CPPCornerCandidatesKernel : public INEKernel
{
public:
    /** Default constructor */
    CPPCornerCandidatesKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CPPCornerCandidatesKernel(const CPPCornerCandidatesKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CPPCornerCandidatesKernel &operator=(const CPPCornerCandidatesKernel &) = delete;
    /** Allow instances of this class to be moved */
    CPPCornerCandidatesKernel(CPPCornerCandidatesKernel &&) = default;
    /** Allow instances of this class to be moved */
    CPPCornerCandidatesKernel &operator=(CPPCornerCandidatesKernel &&) = default;
    /** Default destructor */
    ~CPPCornerCandidatesKernel() = default;

    /** Setup the kernel parameters
     *
     * @param[in]  input                 Source image (harris score). Format supported F32
     * @param[out] output                Destination array of InternalKeypoint
     * @param[out] num_corner_candidates Number of corner candidates
     */
    void configure(const IImage *input, InternalKeypoint *output, int32_t *num_corner_candidates);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;

private:
    int32_t           *_num_corner_candidates;   /**< Number of corner candidates */
    arm_compute::Mutex _corner_candidates_mutex; /**< Mutex to preventing race conditions */
    const IImage      *_input;                   /**< Source image - Harris score */
    InternalKeypoint *_output;                   /**< Array of NEInternalKeypoint */
};
} // namespace arm_compute
#endif /* __ARM_COMPUTE_CPPCORNERCANDIDATESKERNEL_H__ */
