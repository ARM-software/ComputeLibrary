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
#ifndef __ARM_COMPUTE_GRAPH_TENSOR_H__
#define __ARM_COMPUTE_GRAPH_TENSOR_H__

#include "arm_compute/graph/ITensorAccessor.h"
#include "arm_compute/graph/Types.h"
#include "support/ToolchainSupport.h"

#include <memory>

namespace arm_compute
{
namespace graph
{
/** Tensor class */
class Tensor
{
public:
    /** Constructor
     *
     * @param[in] info Tensor info to use
     */
    Tensor(TensorInfo &&info);
    /** Constructor
     *
     * @param[in] accessor Tensor accessor
     */
    template <typename AccessorType>
    Tensor(std::unique_ptr<AccessorType> accessor)
        : _target(TargetHint::DONT_CARE), _info(), _accessor(std::move(accessor)), _tensor(nullptr)
    {
    }
    /** Constructor
     *
     * @param[in] accessor Tensor accessor
     */
    template <typename AccessorType>
    Tensor(AccessorType &&accessor)
        : _target(TargetHint::DONT_CARE), _info(), _accessor(arm_compute::support::cpp14::make_unique<AccessorType>(std::forward<AccessorType>(accessor))), _tensor(nullptr)
    {
    }
    /** Constructor
     *
     * @param[in] info     Tensor info to use
     * @param[in] accessor Tensor accessor
     */
    template <typename AccessorType>
    Tensor(TensorInfo &&info, std::unique_ptr<AccessorType> &&accessor)
        : _target(TargetHint::DONT_CARE), _info(info), _accessor(std::move(accessor)), _tensor(nullptr)
    {
    }
    /** Constructor
     *
     * @param[in] info     Tensor info to use
     * @param[in] accessor Tensor accessor
     */
    template <typename AccessorType>
    Tensor(TensorInfo &&info, AccessorType &&accessor)
        : _target(TargetHint::DONT_CARE), _info(info), _accessor(arm_compute::support::cpp14::make_unique<AccessorType>(std::forward<AccessorType>(accessor))), _tensor(nullptr)
    {
    }
    /** Default Destructor */
    ~Tensor() = default;
    /** Move Constructor
     *
     * @param[in] src Tensor to move
     */
    Tensor(Tensor &&src) noexcept;

    /** Sets the given TensorInfo to the tensor
     *
     * @param[in] info TensorInfo to set
     */
    void set_info(TensorInfo &&info);
    /** Calls accessor on tensor
     *
     * @return True if succeeds else false
     */
    bool call_accessor();
    /** Sets target of the tensor
     *
     * @param[in] target Target where the tensor should be pinned in
     *
     * @return
     */
    ITensor *set_target(TargetHint target);
    /** Returns tensor's TensorInfo
     *
     * @return TensorInfo of the tensor
     */
    const TensorInfo &info() const;
    /** Returns a pointer to the internal tensor
     *
     * @return Tensor
     */
    ITensor *tensor();
    /** Allocates and fills the tensor if needed */
    void allocate_and_fill_if_needed();
    /** Allocates the tensor */
    void allocate();
    /** Return the target that this tensor is pinned on
     *
     * @return Target of the tensor
     */
    TargetHint target() const;

private:
    TargetHint                       _target;   /**< Target that this tensor is pinned on */
    TensorInfo                       _info;     /**< Tensor metadata */
    std::unique_ptr<ITensorAccessor> _accessor; /**< Tensor Accessor */
    std::unique_ptr<ITensor>         _tensor;   /**< Tensor */
};
} // namespace graph
} // namespace arm_compute
#endif /* __ARM_COMPUTE_GRAPH_TENSOR_H__ */
