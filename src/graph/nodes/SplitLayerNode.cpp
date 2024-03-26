/*
 * Copyright (c) 2018-2020 Arm Limited.
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
#include "arm_compute/graph/nodes/SplitLayerNode.h"

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/graph/Graph.h"
#include "arm_compute/graph/INodeVisitor.h"

namespace arm_compute
{
namespace graph
{
SplitLayerNode::SplitLayerNode(unsigned int num_splits, int axis, std::vector<int> size_splits)
    : _num_splits(num_splits), _axis(axis), _size_splits(size_splits)
{
    _input_edges.resize(1, EmptyEdgeID);
    _outputs.resize(num_splits, NullTensorID);
}

unsigned int SplitLayerNode::num_splits() const
{
    return _num_splits;
}

unsigned int SplitLayerNode::axis() const
{
    return _axis;
}

std::pair<TensorDescriptor, Coordinates> SplitLayerNode::compute_output_descriptor(
    const TensorDescriptor &input_descriptor, unsigned int num_splits, int axis, unsigned int idx)
{
    // Handle negative axis, negative index is used to specify axis from the end (e.g. -1 for the last axis).
    int              num_dimension = static_cast<int32_t>(input_descriptor.shape.num_dimensions());
    int              tmp_axis      = wrap_around(axis, num_dimension);
    Coordinates      coords;
    TensorDescriptor output_descriptor = input_descriptor;
    int              split_size        = input_descriptor.shape[tmp_axis] / num_splits;
    if (_size_splits.empty())
    {
        output_descriptor.shape.set(tmp_axis, split_size);
        coords.set(tmp_axis, idx * split_size);
    }
    else
    {
        int split_size = _size_splits[idx];
        if (split_size == -1)
        {
            split_size = input_descriptor.shape[tmp_axis];
            for (unsigned int i = 0; i < _size_splits.size() - 1; ++i)
                split_size -= _size_splits[i];
        }
        output_descriptor.shape.set(tmp_axis, split_size);
        int coord_value = 0;
        for (unsigned int i = 0; i < idx; ++i)
            coord_value += _size_splits[i];
        coords.set(tmp_axis, coord_value);
    }

    return std::make_pair(output_descriptor, coords);
}

bool SplitLayerNode::forward_descriptors()
{
    if (input_id(0) != NullTensorID)
    {
        validate();
        for (unsigned int i = 0; i < _outputs.size(); ++i)
        {
            if (output_id(i) != NullTensorID)
            {
                Tensor *dst_i = output(i);
                ARM_COMPUTE_ERROR_ON(dst_i == nullptr);
                dst_i->desc() = configure_output(i);
            }
        }
        return true;
    }
    return false;
}

TensorDescriptor SplitLayerNode::configure_output(size_t idx) const
{
    ARM_COMPUTE_UNUSED(idx);
    ARM_COMPUTE_ERROR_ON(idx >= _outputs.size());

    const Tensor *src = input(0);
    ARM_COMPUTE_ERROR_ON(src == nullptr);

    TensorDescriptor input_descriptor  = src->desc();
    TensorDescriptor output_descriptor = input_descriptor;

    // Handle negative axis, negative index is used to specify axis from the end (e.g. -1 for the last axis).
    int num_dimension = static_cast<int32_t>(src->desc().shape.num_dimensions());
    int tmp_axis      = wrap_around(_axis, num_dimension);

    int split_size = (_size_splits.empty()) ? (input_descriptor.shape[tmp_axis] / _num_splits) : _size_splits[idx];
    if (split_size == -1)
    {
        split_size = input_descriptor.shape[tmp_axis];
        for (unsigned int i = 0; i < _size_splits.size() - 1; ++i)
            split_size -= _size_splits[i];
    }
    output_descriptor.shape.set(tmp_axis, split_size);

    return output_descriptor;
}

Status SplitLayerNode::validate() const
{
    const Tensor *src = input(0);
    ARM_COMPUTE_RETURN_ERROR_ON(src == nullptr);
    int num_dimension = static_cast<int32_t>(src->desc().shape.num_dimensions());
    ARM_COMPUTE_RETURN_ERROR_ON(_axis < (-num_dimension) || _axis >= num_dimension);

    // Handle negative axis, negative index is used to specify axis from the end (e.g. -1 for the last axis).
    int tmp_axis = wrap_around(_axis, num_dimension);

    if (_size_splits.empty())
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(src->desc().shape[tmp_axis] % _num_splits, "Split should be exact");
    }

    return Status{};
}

NodeType SplitLayerNode::type() const
{
    return NodeType::SplitLayer;
}

void SplitLayerNode::accept(INodeVisitor &v)
{
    v.visit(*this);
}
} // namespace graph
} // namespace arm_compute
