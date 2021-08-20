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
#include "src/gpu/cl/kernels/ClConvertFullyConnectedWeightsKernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Utils.h"
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
ClConvertFullyConnectedWeightsKernel::ClConvertFullyConnectedWeightsKernel()
{
    _type = CLKernelType::ELEMENTWISE;
}

void ClConvertFullyConnectedWeightsKernel::configure(const CLCompileContext &compile_context, const ITensorInfo *src, ITensorInfo *dst, const TensorShape &original_src_shape,
                                                     DataLayout data_layout)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src, dst);

    // Output tensor auto initialisation if not yet initialized
    auto_init_if_empty(*dst, *src->clone());

    auto padding_info = get_padding_info({ src, dst });

    ARM_COMPUTE_ERROR_THROW_ON(ClConvertFullyConnectedWeightsKernel::validate(src, dst, original_src_shape, data_layout));

    const DataLayout src_data_layout = (data_layout == DataLayout::NCHW) ? DataLayout::NHWC : DataLayout::NCHW;

    const int width_idx   = get_data_layout_dimension_index(src_data_layout, DataLayoutDimension::WIDTH);
    const int height_idx  = get_data_layout_dimension_index(src_data_layout, DataLayoutDimension::HEIGHT);
    const int channel_idx = get_data_layout_dimension_index(src_data_layout, DataLayoutDimension::CHANNEL);

    const unsigned int num_elems_per_src_plane = original_src_shape[width_idx] * original_src_shape[height_idx];
    const unsigned int num_channels            = original_src_shape[channel_idx];

    const unsigned int factor_1 = (data_layout == DataLayout::NCHW) ? num_elems_per_src_plane : num_channels;
    const unsigned int factor_2 = (data_layout == DataLayout::NCHW) ? num_channels : num_elems_per_src_plane;

    // Set build options
    CLBuildOptions build_opts;
    build_opts.add_option("-DDATA_TYPE=" + get_cl_unsigned_type_from_element_size(src->element_size()));
    build_opts.add_option("-DFACTOR_1=" + support::cpp11::to_string(factor_1));
    build_opts.add_option("-DFACTOR_2=" + support::cpp11::to_string(factor_2));

    // Create kernel
    _kernel = create_kernel(compile_context, "convert_fc_weights", build_opts.options());

    // Configure kernel window
    Window win = calculate_max_window(*src, Steps());
    ICLKernel::configure_internal(win);

    ARM_COMPUTE_ERROR_ON(has_padding_changed(padding_info));
}

Status ClConvertFullyConnectedWeightsKernel::validate(const ITensorInfo *src, const ITensorInfo *dst, const TensorShape &original_src_shape,
                                                      DataLayout data_layout)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(src, dst);
    ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(src);
    ARM_COMPUTE_RETURN_ERROR_ON(src->data_type() == DataType::UNKNOWN);
    ARM_COMPUTE_RETURN_ERROR_ON(src->num_dimensions() != 2);
    ARM_COMPUTE_RETURN_ERROR_ON(src->dimension(1) != original_src_shape.total_size_lower(3));
    ARM_COMPUTE_RETURN_ERROR_ON(data_layout == DataLayout::UNKNOWN);

    // Checks performed when dst is configured
    if(dst->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src, dst);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(src, dst);
    }

    return Status{};
}

void ClConvertFullyConnectedWeightsKernel::run_op(ITensorPack &tensors, const Window &window, ::cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

    const auto src = utils::cast::polymorphic_downcast<const ICLTensor *>(tensors.get_const_tensor(TensorType::ACL_SRC));
    auto       dst = utils::cast::polymorphic_downcast<ICLTensor *>(tensors.get_tensor(TensorType::ACL_DST));
    ARM_COMPUTE_ERROR_ON_NULLPTR(src, dst);

    unsigned int idx = 0;
    add_2D_tensor_argument(idx, src, window);
    add_2D_tensor_argument(idx, dst, window);
    enqueue(queue, *this, window, lws_hint());
}
} // namespace kernels
} // namespace opencl
} // namespace arm_compute
