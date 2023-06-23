/*
 * Copyright (c) 2016-2023 Arm Limited.
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
#include "src/gpu/cl/kernels/ClCastKernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/utils/helpers/AdjustVecSize.h"
#include "arm_compute/core/utils/StringUtils.h"
#include "arm_compute/core/Validate.h"
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
Status validate_arguments(const ITensorInfo *src, const ITensorInfo *dst, ConvertPolicy policy)
{
    ARM_COMPUTE_UNUSED(policy);
    ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(src);
    ARM_COMPUTE_RETURN_ERROR_ON(src == dst);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(src,
                                                         1,
                                                         DataType::U8, DataType::S8, DataType::QASYMM8, DataType::QASYMM8_SIGNED, DataType::QSYMM8_PER_CHANNEL, DataType::S16,
                                                         DataType::U16, DataType::U32, DataType::S32, DataType::F16,
                                                         DataType::F32, DataType::S64, DataType::U64);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(dst,
                                                         1,
                                                         DataType::U8, DataType::S8, DataType::QASYMM8, DataType::S16,
                                                         DataType::U16, DataType::U32, DataType::S32, DataType::F16,
                                                         DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(src->data_type() == dst->data_type(), "src and dst data types must be different");

    // Validate in case of configured dst
    if(dst->total_size() > 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(src, dst);
    }

    return Status{};
}
} // namespace

ClCastKernel::ClCastKernel()
{
    _type = CLKernelType::ELEMENTWISE;
}

void ClCastKernel::configure(const CLCompileContext &compile_context, const ITensorInfo *src, ITensorInfo *dst, ConvertPolicy policy)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src, dst);

    // Auto initialize dst shape if not initialized (We can only auto-configure the shape, datatype must be given)
    set_shape_if_empty(*dst, src->tensor_shape());

    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(src, dst, policy));

    auto padding_info = get_padding_info({ src, dst });

    // Get data sizes
    const size_t src_size = data_size_from_type(src->data_type());
    const size_t dst_size = data_size_from_type(dst->data_type());

    // Get number of elements to process per iterations
    const unsigned int num_elems_processed_per_iteration = adjust_vec_size(16 / src->element_size(), src->dimension(0));

    // Set build options
    CLBuildOptions build_opts;
    build_opts.add_option("-DVEC_SIZE=" + support::cpp11::to_string(num_elems_processed_per_iteration));
    build_opts.add_option("-DVEC_SIZE_LEFTOVER=" + support::cpp11::to_string(src->dimension(0) % num_elems_processed_per_iteration));
    build_opts.add_option("-DDATA_TYPE_IN=" + get_cl_type_from_data_type(src->data_type()));
    build_opts.add_option("-DDATA_TYPE_OUT=" + get_cl_type_from_data_type(dst->data_type()));
    // Conversions from float always SATURATE as out-of-bounds conversion from float->integer is implementation defined
    build_opts.add_option_if(is_data_type_float(src->data_type()) || policy == ConvertPolicy::SATURATE, "-DSATURATE");
    build_opts.add_option_if(is_data_type_float(src->data_type()) || is_data_type_float(dst->data_type()), "-DIS_DATA_TYPE_FLOAT");
    build_opts.add_option_if(is_data_type_quantized(src->data_type()), "-DIS_DATA_TYPE_QUANTIZED");

    // Create kernel
    const std::string kernel_name = (src_size >= dst_size) ? "cast_down" : "cast_up";
    _kernel                       = create_kernel(compile_context, kernel_name, build_opts.options());

    // Configure kernel
    Window win = calculate_max_window(*src, Steps(num_elems_processed_per_iteration));
    ICLKernel::configure_internal(win);

    // Collapse window
    const Window &full_window      = window();
    Window        collapsed_window = full_window.collapse_if_possible(full_window, Window::DimZ);
    ICLKernel::configure_internal(collapsed_window);

    ARM_COMPUTE_ERROR_ON(has_padding_changed(padding_info));

    // Set config_id for enabling LWS tuning
    _config_id = kernel_name;
    _config_id += "_";
    _config_id += lower_string(string_from_data_type(src->data_type()));
    _config_id += "_";
    _config_id += support::cpp11::to_string(src->dimension(0));
    _config_id += "_";
    _config_id += support::cpp11::to_string(src->dimension(1));
    _config_id += "_";
    _config_id += support::cpp11::to_string(dst->dimension(0));
    _config_id += "_";
    _config_id += support::cpp11::to_string(dst->dimension(1));
}

Status ClCastKernel::validate(const ITensorInfo *src, const ITensorInfo *dst, ConvertPolicy policy)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(src, dst, policy));
    return Status{};
}

void ClCastKernel::run_op(ITensorPack &tensors, const Window &window, ::cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

    const auto src = utils::cast::polymorphic_downcast<const ICLTensor *>(tensors.get_const_tensor(TensorType::ACL_SRC));
    auto       dst = utils::cast::polymorphic_downcast<ICLTensor *>(tensors.get_tensor(TensorType::ACL_DST));

    ARM_COMPUTE_ERROR_ON_NULLPTR(src, dst);

    Window collapsed = window.collapse_if_possible(ICLKernel::window(), Window::DimZ);
    Window slice     = collapsed.first_slice_window_3D();

    do
    {
        unsigned int idx = 0;
        add_3D_tensor_argument(idx, src, slice);
        add_3D_tensor_argument(idx, dst, slice);
        enqueue(queue, *this, slice, lws_hint());
    }
    while(collapsed.slide_window_slice_3D(slice));
}
} // namespace kernels
} // namespace opencl
} // namespace arm_compute
