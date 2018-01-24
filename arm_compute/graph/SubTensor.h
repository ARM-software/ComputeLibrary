/*
 * Copyright (c) 2017-2018 ARM Limited.
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
#include "arm_compute/graph/ITensorObject.h"
#include "arm_compute/graph/Tensor.h"
#include "arm_compute/graph/Types.h"
#include "support/ToolchainSupport.h"

#include <memory>

namespace arm_compute
{
namespace graph
{
/** SubTensor class */
class SubTensor final : public ITensorObject
{
public:
    /** Default Constructor */
    SubTensor();
    /** Constructor
     *
     * @param[in] parent        Parent to create sub-tensor from
     * @param[in] tensor_shape  Sub-tensor shape
     * @param[in] coords        Starting coordinates of the sub-tensor in the parent tensor
     * @param[in] extend_parent (Optional) Extend parent with subtensor shape if subtensor indexes out of bounds
     */
    SubTensor(Tensor &parent, TensorShape tensor_shape, Coordinates coords, bool extend_parent = false);
    /** Constructor
     *
     * @param[in] parent        Parent to create sub-tensor from
     * @param[in] tensor_shape  Sub-tensor shape
     * @param[in] coords        Starting coordinates of the sub-tensor in the parent tensor
     * @param[in] target        Execution target
     * @param[in] extend_parent (Optional) Extend parent with subtensor shape if subtensor indexes out of bounds
     */
    SubTensor(arm_compute::ITensor *parent, TensorShape tensor_shape, Coordinates coords, TargetHint target, bool extend_parent = false);
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

    // Inherited methods overriden:
    bool                  call_accessor() override;
    bool                  has_accessor() const override;
    arm_compute::ITensor *set_target(TargetHint target) override;
    arm_compute::ITensor       *tensor() override;
    const arm_compute::ITensor *tensor() const override;
    TargetHint                  target() const override;
    void                        allocate() override;

private:
    /** Instantiates a sub-tensor */
    void instantiate_subtensor();

private:
    TargetHint                            _target;        /**< Target that this tensor is pinned on */
    TensorShape                           _tensor_shape;  /**< SubTensor shape */
    Coordinates                           _coords;        /**< SubTensor Coordinates */
    arm_compute::ITensor                 *_parent;        /**< Parent tensor */
    std::unique_ptr<arm_compute::ITensor> _subtensor;     /**< SubTensor */
    bool                                  _extend_parent; /**< Parent extension flag */
};
} // namespace graph
} // namespace arm_compute
#endif /* __ARM_COMPUTE_GRAPH_SUBTENSOR_H__ */
