/*
 * Copyright (c) 2018-2022 Arm Limited.
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
#include "src/gpu/cl/kernels/ClWinogradFilterTransformKernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "src/core/CL/CLValidate.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"
#include "support/Cast.h"
#include "support/StringSupport.h"

using namespace arm_compute::misc::shape_calculator;

namespace arm_compute
{
namespace opencl
{
namespace kernels
{
namespace
{
Status validate_arguments(const ITensorInfo *input, const ITensorInfo *output, const WinogradInfo &winograd_info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F32, DataType::F16);
    ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(input);

    const Size2D kernel_size      = winograd_info.kernel_size;
    const Size2D output_tile_size = winograd_info.output_tile_size;

    const size_t idx_w = get_data_layout_dimension_index(input->data_layout(), DataLayoutDimension::WIDTH);
    const size_t idx_h = get_data_layout_dimension_index(input->data_layout(), DataLayoutDimension::HEIGHT);

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(!cl_winograd_convolution_layer_supported(output_tile_size, kernel_size, input->data_layout()), "Winograd filter transform not supported");
    ARM_COMPUTE_RETURN_ERROR_ON(input->dimension(idx_w) != kernel_size.width || input->dimension(idx_h) != kernel_size.height);
    ARM_COMPUTE_RETURN_ERROR_ON(input->num_dimensions() > 4);

    // Checks performed when output is configured
    if(output->total_size() != 0)
    {
        const TensorInfo tensor_info_output = input->clone()->set_tensor_shape(compute_winograd_filter_transform_shape(*input, winograd_info));

        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(output, &tensor_info_output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    }

    return Status{};
}

std::pair<Status, Window> validate_and_configure_window(ITensorInfo *input, ITensorInfo *output)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_UNUSED(output);

    const unsigned int num_elems_processed_per_iteration_x = input->data_layout() == DataLayout::NCHW ? input->dimension(0) : 1;
    const unsigned int num_elems_processed_per_iteration_y = input->dimension(1);
    const unsigned int num_elems_read_per_iteration_z      = input->data_layout() == DataLayout::NCHW ? 1 : input->dimension(2);

    Window win           = calculate_max_window(*input, Steps(num_elems_processed_per_iteration_x, num_elems_processed_per_iteration_y, num_elems_read_per_iteration_z));
    Window win_collapsed = win.collapse(win, Window::DimZ);
    return std::make_pair(Status{}, win_collapsed);
}
} // namespace

ClWinogradFilterTransformKernel::ClWinogradFilterTransformKernel()
{
    _type = CLKernelType::WINOGRAD;
}

void ClWinogradFilterTransformKernel::configure(const ClCompileContext &compile_context, ITensorInfo *src, ITensorInfo *dst, const WinogradInfo &winograd_info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src, dst);

    // Output auto initialization if not yet initialized
    auto_init_if_empty(*dst, src->clone()->set_tensor_shape(compute_winograd_filter_transform_shape(*src, winograd_info)));

    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(src, dst, winograd_info));
    auto padding_info = get_padding_info({ src, dst });

    // Set build options
    CLBuildOptions build_opts;

    // For NHWC layouts pass tensor dimesions at runtime
    if(src->data_layout() == DataLayout::NHWC)
    {
        _src_dim_z = src->dimension(2);
    }
    else
    {
        build_opts.add_option("-DSRC_DIM_Z=" + support::cpp11::to_string(src->dimension(2)));
    }
    build_opts.add_option("-DDATA_TYPE=" + get_cl_type_from_data_type(src->data_type()));
    build_opts.add_option_if(winograd_info.kernel_size.height == 1, "-DWINOGRAD_FILTER_TRANSFORM_HORIZONTAL");
    build_opts.add_option_if(winograd_info.kernel_size.width == 1, "-DWINOGRAD_FILTER_TRANSFORM_VERTICAL");
    const Size2D kernel_size      = winograd_info.kernel_size;
    const Size2D output_tile_size = winograd_info.output_tile_size;

    // Create kernel
    std::string kernel_name = "winograd_filter_transform_" + output_tile_size.to_string() + "_" + kernel_size.to_string() + "_" + lower_string(string_from_data_layout(src->data_layout()));

    // A macro guard to compile ONLY the kernel of interest
    build_opts.add_option("-D" + upper_string(kernel_name));
    _kernel = create_kernel(compile_context, kernel_name, build_opts.options());

    // Configure kernel window
    auto win_config = validate_and_configure_window(src, dst);
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    IClKernel::configure_internal(win_config.second);
    ARM_COMPUTE_ERROR_ON(has_padding_changed(padding_info));
}

Status ClWinogradFilterTransformKernel::validate(const ITensorInfo *src, const ITensorInfo *dst, const WinogradInfo &winograd_info)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(src, dst, winograd_info));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(src->clone().get(), dst->clone().get()).first);

    return Status{};
}

void ClWinogradFilterTransformKernel::run_op(ITensorPack &tensors, const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IClKernel::window(), window);

    auto src = utils::cast::polymorphic_downcast<const ICLTensor *>(tensors.get_const_tensor(TensorType::ACL_SRC));
    auto dst = utils::cast::polymorphic_downcast<ICLTensor *>(tensors.get_tensor(TensorType::ACL_DST));

    // Setup output window
    Window window_out;
    window_out.use_tensor_dimensions(dst->info()->tensor_shape(), 0);

    unsigned int idx = 0;
    add_4D_tensor_argument(idx, src, window);
    add_3D_tensor_argument(idx, dst, window_out);
    if(src->info()->data_layout() == DataLayout::NHWC)
    {
        _kernel.setArg<cl_uint>(idx++, _src_dim_z);
    }
    enqueue(queue, *this, window, lws_hint());
}
} // namespace kernels
} // namespace opencl
} // namespace arm_compute
