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
#ifdef ENABLE_EXPERIMENTAL_DYNAMIC_FUSION

#ifndef ARM_COMPUTE_EXPERIMENTAL_DYNAMICFUSION_OPERATORGRAPH
#define ARM_COMPUTE_EXPERIMENTAL_DYNAMICFUSION_OPERATORGRAPH

#include "arm_compute/core/Error.h"
#include "arm_compute/core/ITensorInfo.h"

#include <memory>

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
/** Graph of operators to execute within a Workload. This is a pure descriptive construct.
 */
class OperatorGraph final
{
public:
    struct Implementation;
    OperatorGraph();
    ~OperatorGraph();

public:
    Implementation       *impl();
    const Implementation *impl() const;

private:
    std::unique_ptr<Implementation> _impl;
};

/** Return the validity of @p op_graph, usually after performing an operation (e.g. add_tensor) on it
 *
 * @param[in,out] op_graph OperatorGraph to be validated
 *
 * @return Status
 */
Status validate(const OperatorGraph &op_graph);

/** Operator Tensor Handle
 * This can be either an argument tensor, or an intermediate tensor linking 2 @ref Operator s
 */
class OpTensor final
{
public:
    using Id = int;
    OpTensor(Id id = {});
    /** Id of the OpTensor
     * @return Id
     */
    Id id() const;

private:
    Id _id{};
};

/** Provide order of @ref OpTensor by checking if @p t0 is "lower than" @p t1
 *
 * @param[in] t0 OpTensor
 * @param[in] t1 OpTensor
 *
 * @return true   if @p t0 is lower than @p t1
 * @return false  otherwise
 */
bool operator<(const OpTensor &t0, const OpTensor &t1);

/** Associate a TensorInfo with a newly created @ref OpTensor in the @p graph.
 *
 * @note @p info needs to remain in scope and valid until the workload has finished building
 * @note Can pass in an empty TensorInfo for a destination Tensor, in which case @p info will be inferred from the source tensors
 *
 * @param[in,out] graph OperatorGraph where the tensor is added
 * @param[in]     info  TensorInfo to be associated
 *
 * @return OpTensor
 */
OpTensor add_tensor(OperatorGraph &graph, ITensorInfo &info);

/** Operator Handle
 * This can be used to further modify an existing operator
 */
class Operator final
{
public:
    using Id = int;
    Operator(Id id = {});
    /** Id of the Operator
     * @return Id
     */
    Id id() const;

private:
    Id _id{};
};

/** Provide order of @ref Operator by checking if @p op0 is "lower than" @p op1
 *
 * @param[in] op0 Operator
 * @param[in] op1 Operator
 *
 * @return true   if @p op0 is lower than @p op1
 * @return false  otherwise
 */
bool operator<(const Operator &op0, const Operator &op1);

/** Padding information for 2D operations like Conv2dDescriptor
 */
struct Padding2D
{
    Padding2D() = default;
    Padding2D(size_t left, size_t right, size_t top, size_t bottom)
        : left(left), right(right), top(top), bottom(bottom)
    {
    }
    size_t left   = { 0 }; /**<  Padding across the width dimension on the left, in elements. */
    size_t right  = { 0 }; /**<  Padding across the width dimension on the right, in elements. */
    size_t top    = { 0 }; /**<  Padding across the height dimension on the top, in elements. */
    size_t bottom = { 0 }; /**<  Padding across the height dimension on the bottom, in elements. */
};

/** Descriptor for Conv2dDescriptor operation
 */
struct Conv2dDescriptor
{
    /* TOSA compliant attribute parameters start */
    Padding2D pad{};
    Size2D    stride{ 1U, 1U };
    Size2D    dilation{ 1U, 1U };
    /* TOSA compliant attribute parameters end */
    /* Non-TOSA compliant attribute parameters start */
    /* Non-TOSA compliant attribute parameters end */
};
/** Add op Conv2d to @p graph
 *
 * @param[in,out] graph   OperatorGraph where the operator is added to
 * @param[in]     desc    Operator descriptor
 * @param[in]     input   Input OpTensor
 * @param[in]     weights Weights OpTensor
 * @param[in]     bias    (Optional) bias OpTensor
 * @param[in]     dst     Destination OpTensor
 *
 * @return Operator
 */
Operator add_op_conv2d(OperatorGraph &graph, const Conv2dDescriptor &desc, OpTensor input, OpTensor weights, OpTensor bias, OpTensor dst);
Operator add_op_conv2d(OperatorGraph &graph, const Conv2dDescriptor &desc, OpTensor input, OpTensor weights, OpTensor dst);
/** (Only for Debuging and Testing) Force a conv2d method
 *
 * @param[in,out] graph  OperatorGraph where conv2d op is located
 * @param[in]     conv2d Conv2d Op
 * @param[in]     method Forced ConvolutionMethod
 */
void force_conv2d_method(OperatorGraph &graph, Operator conv2d, ConvolutionMethod method);

/** Descriptor for Addition operation
 *
 */
struct AddDescriptor
{
    /* TOSA compliant attribute parameters start */
    /* TOSA compliant attribute parameters end */
    /* Non-TOSA compliant attribute parameters start */
    /* Non-TOSA compliant attribute parameters end */
};
/** Add op Add to @p graph, and optionally describes fusion through passing of intermediate @ref OpTensor s
 *
 * @param[in,out] graph OperatorGraph where the operator is added to
 * @param[in]     desc  Operator descriptor
 * @param[in]     lhs   Lhs OpTensor
 * @param[in]     rhs   Rhs OpTensor
 * @param[in]     dst   Destination OpTensor
 *
 * @return Operator
 */
Operator add_op_elementwise_add(OperatorGraph &graph, const AddDescriptor &desc, OpTensor lhs, OpTensor rhs, OpTensor dst);

bool operator==(const OpTensor &t0, const OpTensor &t1);
bool operator==(const Padding2D &pad0, const Padding2D &pad1);
bool operator==(const Conv2dDescriptor &conv2d0, const Conv2dDescriptor &conv2d1);
bool operator==(const AddDescriptor &, const AddDescriptor &);

} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
#endif //ARM_COMPUTE_EXPERIMENTAL_DYNAMICFUSION_OPERATORGRAPH
#endif /* ENABLE_EXPERIMENTAL_DYNAMIC_FUSION */