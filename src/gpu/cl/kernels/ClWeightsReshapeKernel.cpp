/*
 * Copyright (c) 2017-2021 Arm Limited.
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
#include "src/gpu/cl/kernels/ClWeightsReshapeKernel.h"

#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"

#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"
#include "support/Cast.h"
#include "support/StringSupport.h"

namespace arm_compute
{
using namespace misc::shape_calculator;
namespace opencl
{
namespace kernels
{
namespace
{
Status validate_arguments(const ITensorInfo *input,
                          const ITensorInfo *biases,
                          const ITensorInfo *output,
                          unsigned int       num_groups)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_RETURN_ERROR_ON(input->data_type() == DataType::UNKNOWN);
    ARM_COMPUTE_RETURN_ERROR_ON(num_groups == 0);
    ARM_COMPUTE_RETURN_ERROR_ON(input->data_layout() == DataLayout::NHWC && num_groups > 1);
    ARM_COMPUTE_RETURN_ERROR_ON(input->num_dimensions() > 4 && num_groups > 1);
    ARM_COMPUTE_RETURN_ERROR_ON((input->dimension(3) % num_groups) != 0);

    if (biases != nullptr)
    {
        ARM_COMPUTE_RETURN_ERROR_ON(!is_data_type_float(input->data_type()));
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, biases);
        ARM_COMPUTE_RETURN_ERROR_ON((input->num_dimensions() == 4) && (biases->num_dimensions() != 1));
        ARM_COMPUTE_RETURN_ERROR_ON((input->num_dimensions() == 5) && (biases->num_dimensions() != 2));
        ARM_COMPUTE_RETURN_ERROR_ON((input->num_dimensions() == 4) &&
                                    (biases->dimension(0) != input->tensor_shape()[3]));
        ARM_COMPUTE_RETURN_ERROR_ON(
            (input->num_dimensions() == 5) &&
            (biases->dimension(0) != input->tensor_shape()[3] || biases->dimension(1) != input->tensor_shape()[4]));
    }

    // Checks performed when output is configured
    if (output->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(
            output->tensor_shape(), compute_weights_reshaped_shape(*input, biases != nullptr, num_groups));
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_QUANTIZATION_INFO(input, output);
    }

    return Status{};
}
} // namespace

ClWeightsReshapeKernel::ClWeightsReshapeKernel()
{
    _type = CLKernelType::ELEMENTWISE;
}

void ClWeightsReshapeKernel::configure(const ClCompileContext &compile_context,
                                       const ITensorInfo      *src,
                                       const ITensorInfo      *biases,
                                       ITensorInfo            *dst,
                                       unsigned int            num_groups)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src, dst);

    // Output tensor auto inizialitation if not yet initialized
    auto_init_if_empty(
        *dst, src->clone()->set_tensor_shape(compute_weights_reshaped_shape(*src, (biases != nullptr), num_groups)));

    // Perform validation step
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(src, biases, dst, num_groups));
    auto padding_info = get_padding_info({src, biases, dst});

    const DataType data_type = src->data_type();

    // Create build options
    CLBuildOptions build_opts;
    build_opts.add_option("-DDATA_TYPE=" + get_cl_unsigned_type_from_element_size(data_size_from_type(data_type)));
    build_opts.add_option("-DNUM_GROUPS=" + support::cpp11::to_string(num_groups));
    build_opts.add_option_if(biases != nullptr, "-DHAS_BIAS");

    // Create kernel
    _kernel = create_kernel(compile_context, "reshape_to_columns", build_opts.options());

    // Configure window
    Window win = calculate_max_window(*src, Steps());
    ICLKernel::configure_internal(win);

    ARM_COMPUTE_ERROR_ON(has_padding_changed(padding_info));
}

Status ClWeightsReshapeKernel::validate(const ITensorInfo *src,
                                        const ITensorInfo *biases,
                                        const ITensorInfo *dst,
                                        unsigned int       num_groups)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(src, biases, dst, num_groups));
    return Status{};
}

void ClWeightsReshapeKernel::run_op(ITensorPack &tensors, const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_WINDOWS(ICLKernel::window(), window);

    auto src    = utils::cast::polymorphic_downcast<const ICLTensor *>(tensors.get_const_tensor(TensorType::ACL_SRC));
    auto biases = utils::cast::polymorphic_downcast<const ICLTensor *>(tensors.get_const_tensor(TensorType::ACL_BIAS));
    auto dst    = utils::cast::polymorphic_downcast<ICLTensor *>(tensors.get_tensor(TensorType::ACL_DST));

    Window out_window;
    out_window.use_tensor_dimensions(dst->info()->tensor_shape());

    Window in_slice  = window.first_slice_window_3D();
    Window out_slice = out_window.first_slice_window_2D();

    Window biases_window;
    Window biases_slice;

    unsigned int idx = num_arguments_per_3D_tensor() + num_arguments_per_2D_tensor();
    idx += (biases != nullptr) ? num_arguments_per_1D_tensor() : 0;
    _kernel.setArg<cl_uint>(idx++, src->info()->dimension(0));
    _kernel.setArg<cl_uint>(idx++, src->info()->dimension(1));
    _kernel.setArg<cl_uint>(idx++, src->info()->dimension(2));
    _kernel.setArg<cl_uint>(idx++, src->info()->dimension(3));
    _kernel.setArg<cl_uint>(idx++, dst->info()->strides_in_bytes().z());

    if (biases != nullptr)
    {
        biases_window.use_tensor_dimensions(biases->info()->tensor_shape());
        biases_slice = biases_window.first_slice_window_1D();
    }

    do
    {
        // Set arguments
        unsigned idx = 0;
        add_3D_tensor_argument(idx, src, in_slice);
        add_2D_tensor_argument(idx, dst, out_slice);
        if (biases != nullptr)
        {
            add_1D_tensor_argument(idx, biases, biases_slice);
            ARM_COMPUTE_UNUSED(biases_window.slide_window_slice_1D(biases_slice));
        }

        // Run kernel
        enqueue(queue, *this, in_slice, lws_hint());
    } while (window.slide_window_slice_4D(in_slice) && out_window.slide_window_slice_2D(out_slice));
}
} // namespace kernels
} // namespace opencl
} // namespace arm_compute
