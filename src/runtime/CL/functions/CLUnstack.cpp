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
#include "arm_compute/runtime/CL/functions/CLUnstack.h"

#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"

namespace arm_compute
{
namespace
{
inline unsigned int wrap_axis(int axis, const ITensorInfo *const tensor)
{
    return wrap_around(axis, static_cast<int>(tensor->num_dimensions()));
}

inline void setup_slice_coordinates_and_mask(Coordinates &slice_start, int32_t &slice_end_mask, const unsigned int input_num_dimensions)
{
    // Setups up coordinates to slice the input tensor: start coordinates to all 0s and the unstacking axis of both Start/End to slice just one 2d tensor at a time.
    Coordinates slice_end;
    slice_start.set_num_dimensions(input_num_dimensions);
    slice_end.set_num_dimensions(input_num_dimensions);
    for(size_t k = 0; k < input_num_dimensions; ++k)
    {
        slice_start.set(k, 0);
        slice_end.set(k, -1);
    }
    slice_end_mask = arm_compute::helpers::tensor_transform::construct_slice_end_mask(slice_end);
}
} // namespace

CLUnstack::CLUnstack() // NOLINT
    : _num_slices(0),
      _strided_slice_vector()
{
}

void CLUnstack::configure(const ICLTensor *input, const std::vector<ICLTensor *> &output_vector, int axis)
{
    std::vector<ITensorInfo *> outputs_vector_info(output_vector.size());
    std::transform(output_vector.begin(), output_vector.end(), outputs_vector_info.begin(), [](ICLTensor * t)
    {
        ARM_COMPUTE_ERROR_ON_NULLPTR(t);
        return t->info();
    });

    ARM_COMPUTE_ERROR_ON_NULLPTR(input);
    ARM_COMPUTE_ERROR_THROW_ON(CLUnstack::validate(input->info(), outputs_vector_info, axis));

    // Wrap around negative values
    const unsigned int axis_u = wrap_axis(axis, input->info());
    _num_slices               = std::min(outputs_vector_info.size(), input->info()->dimension(axis_u));
    _strided_slice_vector.resize(_num_slices);

    Coordinates slice_start;
    int32_t     slice_end_mask;
    setup_slice_coordinates_and_mask(slice_start, slice_end_mask, input->info()->tensor_shape().num_dimensions());
    for(unsigned int slice = 0; slice < _num_slices; ++slice)
    {
        // Adjusts start and end coordinates to take a 2D slice at a time
        slice_start.set(axis_u, slice);
        _strided_slice_vector[slice].configure(input, output_vector[slice], slice_start, Coordinates(), BiStrides(), 0, slice_end_mask, (1 << axis_u));
    }
}

Status CLUnstack::validate(const ITensorInfo *input, const std::vector<ITensorInfo *> &output_vector, int axis)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input);
    ARM_COMPUTE_RETURN_ERROR_ON(output_vector.empty());
    ARM_COMPUTE_RETURN_ERROR_ON(axis < (-static_cast<int>(input->tensor_shape().num_dimensions())));
    ARM_COMPUTE_RETURN_ERROR_ON(axis >= static_cast<int>(input->tensor_shape().num_dimensions()));
    const unsigned int num_slices = std::min(output_vector.size(), input->dimension(wrap_axis(axis, input)));
    ARM_COMPUTE_RETURN_ERROR_ON(num_slices > input->dimension(wrap_axis(axis, input)));
    ARM_COMPUTE_RETURN_ERROR_ON(num_slices > output_vector.size());
    Coordinates slice_start;
    int32_t     slice_end_mask;
    for(size_t k = 0; k < num_slices; ++k)
    {
        slice_start.set(wrap_axis(axis, input), k);
        setup_slice_coordinates_and_mask(slice_start, slice_end_mask, input->tensor_shape().num_dimensions());
        ARM_COMPUTE_RETURN_ON_ERROR(CLStridedSlice::validate(input, output_vector[k], slice_start, Coordinates(), BiStrides(), 0, slice_end_mask, (1 << wrap_axis(axis, input))));
    }
    return Status{};
}

void CLUnstack::run()
{
    for(unsigned i = 0; i < _num_slices; ++i)
    {
        _strided_slice_vector[i].run();
    }
}

} // namespace arm_compute
