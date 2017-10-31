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
#include "arm_compute/graph/ITensorObject.h"
#include "arm_compute/graph/Types.h"
#include "support/ToolchainSupport.h"

#include <memory>

namespace arm_compute
{
namespace graph
{
/** Tensor class */
class Tensor final : public ITensorObject
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
    /** Returns tensor's TensorInfo
     *
     * @return TensorInfo of the tensor
     */
    const TensorInfo &info() const;
    /** Allocates and fills the tensor if needed */
    void allocate_and_fill_if_needed();

    // Inherited methods overriden:
    bool                  call_accessor() override;
    bool                  has_accessor() const override;
    arm_compute::ITensor *set_target(TargetHint target) override;
    arm_compute::ITensor       *tensor() override;
    const arm_compute::ITensor *tensor() const override;
    TargetHint                  target() const override;
    void                        allocate() override;

private:
    TargetHint                            _target;   /**< Target that this tensor is pinned on */
    TensorInfo                            _info;     /**< Tensor metadata */
    std::unique_ptr<ITensorAccessor>      _accessor; /**< Tensor Accessor */
    std::unique_ptr<arm_compute::ITensor> _tensor;   /**< Tensor */
};
} // namespace graph
} // namespace arm_compute
#endif /* __ARM_COMPUTE_GRAPH_TENSOR_H__ */
