/*
 * Copyright (c) 2018-2019 ARM Limited.
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
#ifndef ARM_COMPUTE_GRAPH_CLSUBTENSORHANDLE_H
#define ARM_COMPUTE_GRAPH_CLSUBTENSORHANDLE_H

#include "arm_compute/graph/ITensorHandle.h"

#include "arm_compute/runtime/CL/CLSubTensor.h"

namespace arm_compute
{
namespace graph
{
namespace backends
{
/** OpenCL Sub-Tensor handle interface object **/
class CLSubTensorHandle final : public ITensorHandle
{
public:
    /** Default constructor
     *
     * @param[in] parent_handle Parent tensor handle
     * @param[in] shape         Sub-Tensor shape
     * @param[in] coords        Starting coordinates
     * @param[in] extend_parent Extends parent shape if true
     */
    CLSubTensorHandle(ITensorHandle *parent_handle, const TensorShape &shape, const Coordinates &coords, bool extend_parent = false);
    /** Destructor: free the tensor's memory */
    ~CLSubTensorHandle() = default;
    /** Allow instances of this class to be move constructed */
    CLSubTensorHandle(CLSubTensorHandle &&) = default;
    /** Allow instances of this class to be moved */
    CLSubTensorHandle &operator=(CLSubTensorHandle &&) = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLSubTensorHandle(const CLSubTensorHandle &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLSubTensorHandle &operator=(const CLSubTensorHandle &) = delete;

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
    arm_compute::CLSubTensor _sub_tensor;    /**< Backend Sub-Tensor */
    ITensorHandle           *_parent_handle; /**< Parent handle */
};
} // namespace backends
} // namespace graph
} // namespace arm_compute
#endif /* ARM_COMPUTE_GRAPH_CLSUBTENSORHANDLE_H */
