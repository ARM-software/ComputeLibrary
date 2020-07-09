/*
 * Copyright (c) 2018-2019 Arm Limited.
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
#ifndef ARM_COMPUTE_GRAPH_GCTENSORHANDLE_H
#define ARM_COMPUTE_GRAPH_GCTENSORHANDLE_H

#include "arm_compute/graph/ITensorHandle.h"

#include "arm_compute/runtime/GLES_COMPUTE/GCTensor.h"

namespace arm_compute
{
namespace graph
{
namespace backends
{
/** GLES compute tensor handle interface object **/
class GCTensorHandle final : public ITensorHandle
{
public:
    /** Default Constructor
     *
     * @param[in] info Tensor metadata
     */
    GCTensorHandle(const ITensorInfo &info);
    /** Destructor: free the tensor's memory */
    ~GCTensorHandle() = default;
    /** Allow instances of this class to be move constructed */
    GCTensorHandle(GCTensorHandle &&) = default;
    /** Allow instances of this class to be moved */
    GCTensorHandle &operator=(GCTensorHandle &&) = default;

    // Inherited overridden methods
    void allocate() override;
    void free() override;
    void manage(IMemoryGroup *mg) override;
    void map(bool blocking) override;
    void                        unmap() override;
    void                        release_if_unused() override;
    arm_compute::ITensor       &tensor() override;
    const arm_compute::ITensor &tensor() const override;
    ITensorHandle              *parent_handle() override;
    bool                        is_subtensor() const override;
    Target                      target() const override;

private:
    arm_compute::GCTensor _tensor; /**< Backend Tensor */
};
} // namespace backends
} // namespace graph
} // namespace arm_compute
#endif /* ARM_COMPUTE_GRAPH_GCTENSORHANDLE_H */
