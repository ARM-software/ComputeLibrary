/*
 * Copyright (c) 2022 Arm Limited.
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
#ifndef SRC_DYNAMIC_FUSION_SKETCH_GPU_GPUOPERATORGROUP
#define SRC_DYNAMIC_FUSION_SKETCH_GPU_GPUOPERATORGROUP

#include "arm_compute/core/ITensorInfo.h"
#include "src/dynamic_fusion/sketch/ArgumentPack.h"
#include "src/dynamic_fusion/sketch/gpu/GpuOperatorProperties.h"
#include "src/dynamic_fusion/sketch/utils/DependencyGraph.h"
#include <map>

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
using OperatorId = DependencyGraph::OperatorId;

/** An operator for the sole purpose of validating fusion
 */
class Operator
{
public:
    /** Default constructor */
    Operator() = default;
    /** Get Operator Id */
    OperatorId id() const;
    /** Get operator type */
    GpuOperatorType operator_type() const;
    /** Get tensor arguments */
    ArgumentPack<ITensorInfo> tensors() const;
    friend class GpuOperatorGroup;

private:
    Operator(OperatorId id, GpuOperatorType operator_type, const ArgumentPack<ITensorInfo> &tensors);
    OperatorId                _id{};
    GpuOperatorType           _operator_type{};
    ArgumentPack<ITensorInfo> _tensors{};
};

/** A linear sequence of operators to be fused in a workload
 *  For the time being, this class is only used for validating operator fusion
 * INVARIANTS:
 * @note These invariants are exactly the same as operator fusion constraints
 * 1. Fusion is limited to a linear sequence of operators
 * 2. Max number of operators that can be fused is @ref GpuOperatorGroup::max_fused_operators
 * 3. The fusion is subject to the pattern: Complex + Simple * | Simple + Simple * | Un-fusable
 * 4. All operator but unfusable, have exactly 1 dst tensor
 * 5. All fused operators share the same dst tensor shape
 * 6. All fused operators' tensors share the same @ref DataLayout
 */
class GpuOperatorGroup
{
public:
    static constexpr size_t max_fused_operators = 32;
    /** Try adding (without actually adding) an operator to the group
     *
     * @param[in] op        Operator to be added
     * @param[in] is_output Whether this operator is the output operator.
     *
     * @return true   If @p op can be added while maintaining the invariants
     * @return false  Otherwise
     */
    bool try_add_operator(const Operator &op, bool is_output = false) const;
    /** Add an operator to the group
     *
     * @param[in] op        Operator to be added
     * @param[in] is_output Whether this operator is the output operator.
     */
    void add_operator(const Operator &op, bool is_output = false);
    /** Create a new operator
     *
     * @param[in] operator_type @ref GpuOperatorType of the new operator
     * @param[in] tensors       Tensor arguments to the new operator
     *
     * @return Operator
     */
    Operator new_operator(const GpuOperatorType &operator_type, const ArgumentPack<ITensorInfo> &tensors) const;
    /** Get the "root operator" of the group, which is the first operator in a linear sequence
     * @return const Operator* Pointer to the root operator
     */
    const Operator *get_root_operator() const;

private:
    DependencyGraph _graph{};
    std::map<OperatorId, Operator> _operators{};
};
} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
#endif /* SRC_DYNAMIC_FUSION_SKETCH_GPU_GPUOPERATORGROUP */
