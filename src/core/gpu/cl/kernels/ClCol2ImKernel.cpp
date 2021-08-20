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
#include "src/core/gpu/cl/kernels/ClCol2ImKernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "src/core/CL/CLValidate.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"
#include "support/Cast.h"
#include "support/StringSupport.h"

#include <cmath>

namespace arm_compute
{
using namespace misc::shape_calculator;
namespace opencl
{
namespace kernels
{
namespace
{
Status validate_arguments(const ITensorInfo *src, const ITensorInfo *dst, const Size2D &convolved_dims, unsigned int num_groups)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(src, dst);
    ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(src);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(src, 1, DataType::QASYMM8, DataType::QASYMM8_SIGNED, DataType::F16, DataType::F32);

    // Checks performed when output is configured
    if(dst->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(dst->tensor_shape(), compute_col2im_shape(*src, convolved_dims, true, num_groups));
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src, dst);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_QUANTIZATION_INFO(src, dst);
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(dst->data_layout() != DataLayout::NCHW, "Col2Im output's data layout must always be NCHW");
    }

    return Status{};
}

std::pair<Status, Window> validate_and_configure_window(ITensorInfo *src, ITensorInfo *dst, const Size2D &convolved_dims, unsigned int num_groups)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src, dst);
    // Output auto inizialitation if not yet initialized
    auto_init_if_empty(*dst, src->clone()->set_tensor_shape(compute_col2im_shape(*src, convolved_dims, true, num_groups)).set_data_layout(DataLayout::NCHW));

    constexpr unsigned int num_elems_read_per_iteration = 8;

    // Configure window
    Window win = calculate_max_window(*src, Steps(num_elems_read_per_iteration));

    // Update window and padding just for the input tensor as we cannot access out-of-bounds elements in the output one
    AccessWindowHorizontal input_access(src, 0, num_elems_read_per_iteration);
    bool                   window_changed = update_window_and_padding(win, input_access);

    Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
    return std::make_pair(err, win);
}
} // namespace

ClCol2ImKernel::ClCol2ImKernel()
    : _convolved_dims()
{
    _type = CLKernelType::ELEMENTWISE;
}

void ClCol2ImKernel::configure(const CLCompileContext &compile_context, ITensorInfo *src, ITensorInfo *dst, const Size2D &convolved_dims, unsigned int num_groups)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src, dst);

    // Perform validation step
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(src, dst, convolved_dims, num_groups));

    _convolved_dims = convolved_dims;

    const DataType data_type = src->data_type();

    // Create kernel
    CLBuildOptions build_opts;
    build_opts.add_option("-DDATA_TYPE=" + get_cl_type_from_data_type(data_type));
    build_opts.add_option("-DELEMENT_SIZE=" + support::cpp11::to_string(src->element_size()));
    build_opts.add_option("-DWIDTH_INPUT=" + support::cpp11::to_string(src->dimension(0)));
    build_opts.add_option("-DWIDTH_OUTPUT=" + support::cpp11::to_string(_convolved_dims.width));
    build_opts.add_option("-DNUM_GROUPS=" + support::cpp11::to_string(num_groups));

    _kernel = create_kernel(compile_context, "col2im", build_opts.options());

    // Configure kernel window
    auto win_config = validate_and_configure_window(src, dst, _convolved_dims, num_groups);
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    IClKernel::configure_internal(win_config.second);

    // Set config_id for enabling LWS tuning
    _config_id = "col2im_";
    _config_id += lower_string(string_from_data_type(src->data_type()));
    _config_id += "_";
    _config_id += support::cpp11::to_string(num_groups);
    _config_id += "_";
    _config_id += support::cpp11::to_string(src->dimension(0));
    _config_id += "_";
    _config_id += support::cpp11::to_string(src->dimension(1));
    _config_id += "_";
    _config_id += support::cpp11::to_string(dst->dimension(0));
    _config_id += "_";
    _config_id += support::cpp11::to_string(dst->dimension(1));
}

Status ClCol2ImKernel::validate(const ITensorInfo *src, const ITensorInfo *dst, const Size2D &convolved_dims, unsigned int num_groups)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src, dst);
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(src, dst, convolved_dims, num_groups));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(src->clone().get(), dst->clone().get(), convolved_dims, num_groups).first);
    return Status{};
}

void ClCol2ImKernel::run_op(ITensorPack &tensors, const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_WINDOWS(IClKernel::window(), window);

    bool is_collapsed     = false;
    bool is_collapsed_out = false;

    auto src = utils::cast::polymorphic_downcast<const ICLTensor *>(tensors.get_const_tensor(TensorType::ACL_SRC));
    auto dst = utils::cast::polymorphic_downcast<ICLTensor *>(tensors.get_tensor(TensorType::ACL_DST));

    Window out_window;
    out_window.use_tensor_dimensions(dst->info()->tensor_shape());

    Window collapsed     = window.collapse_if_possible(ICLKernel::window(), Window::DimZ, &is_collapsed);
    Window collapsed_out = out_window.collapse_if_possible(out_window, 3, &is_collapsed_out);

    ARM_COMPUTE_ERROR_ON(is_collapsed != is_collapsed_out);

    Window slice     = collapsed.first_slice_window_3D();
    Window slice_out = collapsed_out.first_slice_window_4D();
    do
    {
        // Set inputs
        unsigned int idx = 0;
        add_3D_tensor_argument(idx, src, slice);
        add_4D_tensor_argument(idx, dst, slice_out);
        enqueue(queue, *this, slice, lws_hint());
    }
    while(collapsed.slide_window_slice_3D(slice) && collapsed_out.slide_window_slice_4D(slice_out));
}
} // namespace kernels
} // namespace opencl
} // namespace arm_compute
