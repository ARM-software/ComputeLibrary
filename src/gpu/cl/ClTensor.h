/*
 * Copyright (c) 2021 Arm Limited.
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
#ifndef SRC_GPU_CLTENSOR_H
#define SRC_GPU_CLTENSOR_H

#include "src/common/ITensorV2.h"

#include "arm_compute/runtime/CL/CLTensor.h"

namespace arm_compute
{
namespace gpu
{
namespace opencl
{
/** OpenCL tensor implementation class */
class ClTensor final : public ITensorV2
{
public:
    /**  Construct a new OpenCL Tensor object
     *
     * @param[in] ctx  Context to be used
     * @param[in] desc Tensor descriptor
     */
    ClTensor(IContext *ctx, const AclTensorDescriptor &desc);
    /** Allocates tensor
     *
     * @return StatusCode A status code
     */
    StatusCode allocate();

    // Inherrited functions overriden
    void                 *map() override;
    StatusCode            unmap() override;
    arm_compute::ITensor *tensor() const override;
    StatusCode import(void *handle, ImportMemoryType type) override;

private:
    std::unique_ptr<CLTensor> _legacy_tensor;
};
} // namespace opencl
} // namespace gpu
} // namespace arm_compute

#endif /* SRC_GPU_CLTENSOR_H */