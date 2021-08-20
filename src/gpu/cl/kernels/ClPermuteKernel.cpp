/*
 * Copyright (c) 2018-2021 Arm Limited.
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
#include "src/gpu/cl/kernels/ClPermuteKernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "src/core/CL/CLValidate.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"
#include "support/Cast.h"
#include "support/StringSupport.h"

namespace arm_compute
{
namespace opencl
{
namespace kernels
{
namespace
{
TensorShape get_dst_shape(const ITensorInfo *src, const PermutationVector &perm)
{
    TensorShape dst_shape = src->tensor_shape();
    permute(dst_shape, perm);
    return dst_shape;
}

Status validate_arguments(const ITensorInfo *src, const ITensorInfo *dst, const PermutationVector &perm)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(src, dst);
    ARM_COMPUTE_RETURN_ERROR_ON(src->data_type() == DataType::UNKNOWN);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(src->num_dimensions() < 1 || src->num_dimensions() > 4,
                                    "Permutation up to 4-D src tensor is supported");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(perm.num_dimensions() < 1 || perm.num_dimensions() > 4,
                                    "Permutation vector size should be less than or equal to 4");
    for(const auto &p : perm)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(p >= perm.num_dimensions(), "Permutation vector has invalid values");
    }

    // Validate configured dst
    if(dst->total_size() != 0)
    {
        const TensorShape dst_shape = misc::shape_calculator::compute_permutation_output_shape(*src, perm);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(dst->tensor_shape(), dst_shape);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_QUANTIZATION_INFO(src, dst);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src, dst);
    }
    return Status{};
}
} // namespace

ClPermuteKernel::ClPermuteKernel()
{
    _type = CLKernelType::ELEMENTWISE;
}

void ClPermuteKernel::configure(const CLCompileContext &compile_context, const ITensorInfo *src, ITensorInfo *dst, const PermutationVector &perm)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src, dst);
    auto              padding_info = get_padding_info({ src, dst });
    const TensorShape dst_shape    = get_dst_shape(src, perm);
    // Output auto initialization if not yet initialized
    auto_init_if_empty(*dst, src->clone()->set_tensor_shape(dst_shape));

    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(src, dst, perm));

    _perm = perm;

    // Create kernel
    CLBuildOptions build_opts;
    build_opts.add_option("-DDATA_TYPE=" + get_cl_unsigned_type_from_element_size(data_size_from_type(src->data_type())));
    build_opts.add_option("-DDEPTH_IN=" + support::cpp11::to_string(src->dimension(2)));
    // New positions of  width(W), height(H), channel(C) and batch(D) based on permutation vector
    build_opts.add_option("-DP1=" + support::cpp11::to_string((_perm.num_dimensions() >= 1) ? perm[0] : 0));
    build_opts.add_option("-DP2=" + support::cpp11::to_string((_perm.num_dimensions() >= 2) ? perm[1] : 1));
    build_opts.add_option("-DP3=" + support::cpp11::to_string((_perm.num_dimensions() >= 3) ? perm[2] : 2));
    build_opts.add_option("-DP4=" + support::cpp11::to_string((_perm.num_dimensions() >= 4) ? perm[3] : 3));

    _kernel = create_kernel(compile_context, "permute", build_opts.options());

    // Configure  kernel window
    Window win = calculate_max_window(*src, Steps());

    ICLKernel::configure_internal(win);
    ARM_COMPUTE_ERROR_ON(has_padding_changed(padding_info));
}

Status ClPermuteKernel::validate(const ITensorInfo *src, const ITensorInfo *dst, const PermutationVector &perm)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(src, dst);
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(src, dst, perm));

    return Status{};
}

void ClPermuteKernel::run_op(ITensorPack &tensors, const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_WINDOWS(ICLKernel::window(), window);

    const auto src = utils::cast::polymorphic_downcast<const ICLTensor *>(tensors.get_const_tensor(TensorType::ACL_SRC));
    auto       dst = utils::cast::polymorphic_downcast<ICLTensor *>(tensors.get_tensor(TensorType::ACL_DST));

    Window slice_in = window.first_slice_window_4D().collapse(ICLKernel::window(), 2, 4);

    // Setup dst slice
    Window slice_out(slice_in);
    slice_out.set(Window::DimX, Window::Dimension(0, 0, 0));
    slice_out.set(Window::DimY, Window::Dimension(0, 0, 0));
    slice_out.set(Window::DimZ, Window::Dimension(0, 0, 0));
    slice_out.set(3, Window::Dimension(0, 0, 0));

    do
    {
        unsigned int idx = 0;
        add_4D_tensor_argument(idx, src, slice_in);
        add_4D_tensor_argument(idx, dst, slice_out);
        enqueue(queue, *this, slice_in, lws_hint());
    }
    while(window.slide_window_slice_4D(slice_in) && window.slide_window_slice_4D(slice_out));
}
} // namespace kernels
} // namespace opencl
} // namespace arm_compute