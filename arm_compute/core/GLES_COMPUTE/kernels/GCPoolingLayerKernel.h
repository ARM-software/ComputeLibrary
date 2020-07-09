/*
 * Copyright (c) 2017-2020 Arm Limited.
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
#ifndef ARM_COMPUTE_GCPOOLINGLAYERKERNEL_H
#define ARM_COMPUTE_GCPOOLINGLAYERKERNEL_H

#include "arm_compute/core/GLES_COMPUTE/IGCKernel.h"

#include "arm_compute/core/Error.h"

namespace arm_compute
{
class IGCTensor;

/** Interface for the pooling layer kernel */
class GCPoolingLayerKernel : public IGCKernel
{
public:
    /** Default constructor */
    GCPoolingLayerKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    GCPoolingLayerKernel(const GCPoolingLayerKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    GCPoolingLayerKernel &operator=(const GCPoolingLayerKernel &) = delete;
    /** Allow instances of this class to be moved */
    GCPoolingLayerKernel(GCPoolingLayerKernel &&) = default;
    /** Allow instances of this class to be moved */
    GCPoolingLayerKernel &operator=(GCPoolingLayerKernel &&) = default;
    /** Default destructor */
    ~GCPoolingLayerKernel() = default;

    /** Set the input and output tensors.
     *
     * @param[in]  input     Source tensor. Data types supported: F16/F32.
     * @param[out] output    Destination tensor. Data types supported: Same as @p input.
     * @param[in]  pool_info Contains pooling operation information described in @ref PoolingLayerInfo.
     * @param[out] indices   (optional) The indices of the maximal values. Data type supported: U32.
     */
    void configure(const IGCTensor *input, IGCTensor *output, const PoolingLayerInfo &pool_info, IGCTensor *indices = nullptr);

    /** Static function to check if given info will lead to a valid configuration of @ref GCPoolingLayerKernel
     *
     * @param[in] input     Source tensor info. Data types supported: F16/F32.
     * @param[in] output    Destination tensor info. Data types supported: Same as @p input.
     * @param[in] pool_info Contains pooling operation information described in @ref PoolingLayerInfo.
     * @param[in] indices   (optional) The indices of the maximal values. Data type supported: U32.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output, const PoolingLayerInfo &pool_info, const ITensorInfo *indices = nullptr);

    // Inherited methods overridden:
    void run(const Window &window) override;
    BorderSize border_size() const override;

private:
    const IGCTensor *_input;
    IGCTensor       *_output;
    IGCTensor       *_indices;
    PoolingLayerInfo _pool_info;
    BorderSize       _border_size;
    unsigned int     _num_elems_processed_per_iteration;
};
} // namespace arm_compute
#endif /*ARM_COMPUTE_GCPOOLINGLAYERKERNEL_H */
