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
#ifndef __ARM_COMPUTE_CLPOOLINGLAYERKERNEL_H__
#define __ARM_COMPUTE_CLPOOLINGLAYERKERNEL_H__

#include "arm_compute/core/CL/ICLKernel.h"

namespace arm_compute
{
class ICLTensor;

/** Interface for the pooling layer kernel */
class CLPoolingLayerKernel : public ICLKernel
{
public:
    /** Default constructor */
    CLPoolingLayerKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLPoolingLayerKernel(const CLPoolingLayerKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLPoolingLayerKernel &operator=(const CLPoolingLayerKernel &) = delete;
    /** Allow instances of this class to be moved */
    CLPoolingLayerKernel(CLPoolingLayerKernel &&) = default;
    /** Allow instances of this class to be moved */
    CLPoolingLayerKernel &operator=(CLPoolingLayerKernel &&) = default;
    /** Default destructor */
    ~CLPoolingLayerKernel() = default;

    /** Set the input and output tensors.
     *
     * @param[in]  input     Source tensor. Data types supported: F16/F32.
     * @param[out] output    Destination tensor. Data types supported: Same as @p input.
     * @param[in]  pool_info Contains pooling operation information described in @ref PoolingLayerInfo.
     *                       Supported pooling sizes : 2, 3 and 7
     */
    void configure(const ICLTensor *input, ICLTensor *output, const PoolingLayerInfo &pool_info);

    // Inherited methods overridden:
    void run(const Window &window, cl::CommandQueue &queue) override;
    BorderSize border_size() const override;

private:
    const ICLTensor *_input;
    ICLTensor       *_output;
    PoolingLayerInfo _pool_info;
    BorderSize       _border_size;
    unsigned int     _num_elems_processed_per_iteration;
};
}
#endif /*__ARM_COMPUTE_CLPOOLINGLAYERKERNEL_H__ */
