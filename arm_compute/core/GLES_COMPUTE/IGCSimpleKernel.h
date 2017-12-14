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
#ifndef __ARM_COMPUTE_IGCSIMPLEKERNEL_H__
#define __ARM_COMPUTE_IGCSIMPLEKERNEL_H__

#include "arm_compute/core/GLES_COMPUTE/IGCKernel.h"
#include "arm_compute/core/GLES_COMPUTE/IGCTensor.h"
#include "arm_compute/core/Helpers.h"

namespace arm_compute
{
/** Interface for simple OpenGL ES kernels having 1 tensor input and 1 tensor output */
class IGCSimpleKernel : public IGCKernel
{
public:
    /** Constructor. */
    IGCSimpleKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers). */
    IGCSimpleKernel(const IGCSimpleKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers). */
    IGCSimpleKernel &operator=(const IGCSimpleKernel &) = delete;
    /** Allow instances of this class to be moved. */
    IGCSimpleKernel(IGCSimpleKernel &&) = default;
    /** Allow instances of this class to be moved. */
    IGCSimpleKernel &operator=(IGCSimpleKernel &&) = default;
    /** Default destructor */
    ~IGCSimpleKernel() = default;

    /** Configure the kernel
     *
     * @param[in]  input                             Source tensor.
     * @param[out] output                            Destination tensor.
     * @param[in]  num_elems_processed_per_iteration Number of processed elements per iteration.
     * @param[in]  border_undefined                  (Optional) True if the border mode is undefined. False if it's replicate or constant.
     * @param[in]  border_size                       (Optional) Size of the border.
     */
    void configure(const IGCTensor *input, IGCTensor *output, unsigned int num_elems_processed_per_iteration, bool border_undefined = false, const BorderSize &border_size = BorderSize());

protected:
    const IGCTensor *_input;
    IGCTensor       *_output;
};
}

#endif /*__ARM_COMPUTE_IGCSIMPLEKERNEL_H__ */
