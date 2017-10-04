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
#ifndef __ARM_COMPUTE_GRAPH_SUBTENSOR_H__
#define __ARM_COMPUTE_GRAPH_SUBTENSOR_H__

#include "arm_compute/graph/ITensorAccessor.h"
#include "arm_compute/graph/Tensor.h"
#include "arm_compute/graph/Types.h"
#include "support/ToolchainSupport.h"

#include <memory>

namespace arm_compute
{
namespace graph
{
/** SubTensor class */
class SubTensor final
{
public:
    /** Default Constructor */
    SubTensor();
    /** Constructor
     *
     * @param[in] parent       Parent to create sub-tensor from
     * @param[in] tensor_shape Sub-tensor shape
     * @param[in] coords       Starting coordinates of the sub-tensor in the parent tensor
     */
    SubTensor(Tensor &parent, TensorShape tensor_shape, Coordinates coords);
    /** Constructor
     *
     * @param[in] parent       Parent to create sub-tensor from
     * @param[in] tensor_shape Sub-tensor shape
     * @param[in] coords       Starting coordinates of the sub-tensor in the parent tensor
     * @param[in] target       Execution target
     */
    SubTensor(ITensor *parent, TensorShape tensor_shape, Coordinates coords, TargetHint target);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    SubTensor(const SubTensor &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    SubTensor &operator=(const SubTensor &) = delete;
    /** Allow instances of this class to be moved */
    SubTensor(SubTensor &&) = default;
    /** Allow instances of this class to be moved */
    SubTensor &operator=(SubTensor &&) = default;
    /** Default Destructor */
    ~SubTensor() = default;

    /** Sets the given TensorInfo to the tensor
     *
     * @param[in] info TensorInfo to set
     */
    void set_info(SubTensorInfo &&info);
    /** Returns tensor's TensorInfo
     *
     * @return TensorInfo of the tensor
     */
    const SubTensorInfo &info() const;
    /** Returns a pointer to the internal tensor
     *
     * @return Tensor
     */
    ITensor *tensor();
    /** Return the target that this tensor is pinned on
     *
     * @return Target of the tensor
     */
    TargetHint target() const;

private:
    /** Instantiates a sub-tensor */
    void instantiate_subtensor();

private:
    TargetHint               _target;    /**< Target that this tensor is pinned on */
    Coordinates              _coords;    /**< SubTensor Coordinates */
    SubTensorInfo            _info;      /**< SubTensor metadata */
    ITensor                 *_parent;    /**< Parent tensor */
    std::unique_ptr<ITensor> _subtensor; /**< SubTensor */
};
} // namespace graph
} // namespace arm_compute
#endif /* __ARM_COMPUTE_GRAPH_SUBTENSOR_H__ */
