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
#include "src/gpu/cl/kernels/ClIndirectConv2dAddressPrecalculationKernel.h"

#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/KernelDescriptors.h"
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
Status validate_arguments(const ITensorInfo *src, const ITensorInfo *weights, const ITensorInfo *dst,
                          const PadStrideInfo &conv_info, const DirectConvComputeKernelInfo &desc)
{
    ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(src);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(src, 1, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src, weights);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_LAYOUT_NOT_IN(src, DataLayout::NHWC);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(weights->dimension(0) != src->dimension(0), "Weights feature map dimension should match the respective src's one");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(weights->num_dimensions() > 4, "Weights can be at most 4 dimensional");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(desc.m0 <= 0 || desc.m0 > 8, "M0 can only be greater than 0 and less than or equal to 8");

    // Checks performed when dst is configured
    if(dst->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(dst->tensor_shape(),
                                                           misc::shape_calculator::compute_indirect_buffer_shape(src->tensor_shape(),
                                                                                                                 src->data_layout(),
                                                                                                                 weights->tensor_shape(),
                                                                                                                 conv_info,
                                                                                                                 desc));
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(dst, 1, DataType::S32);
    }

    return Status{};
}
} // namespace

ClIndirectConv2dAddressPrecalculationKernel::ClIndirectConv2dAddressPrecalculationKernel()
{
    _type = CLKernelType::ELEMENTWISE;
}

void ClIndirectConv2dAddressPrecalculationKernel::configure(const CLCompileContext &compile_context, ITensorInfo *src, ITensorInfo *weights, ITensorInfo *dst,
                                                            const PadStrideInfo &conv_info, const DirectConvComputeKernelInfo &desc)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src, weights, dst);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(src, weights, dst, conv_info, desc));

    constexpr unsigned int width_idx  = 1;
    constexpr unsigned int height_idx = 2;

    // Get dst shape
    TensorShape output_shape = misc::shape_calculator::compute_indirect_buffer_shape(src->tensor_shape(),
                                                                                     src->data_layout(),
                                                                                     weights->tensor_shape(),
                                                                                     conv_info,
                                                                                     desc);

    TensorShape output_conv_shape = misc::shape_calculator::compute_deep_convolution_shape(*src, *weights, conv_info);

    // Output auto inizialitation if not yet initialized
    auto_init_if_empty(*dst, output_shape, 1, DataType::S32);

    // Configure kernel window
    Window win;

    // Create window and update padding
    win = calculate_max_window(output_shape, Steps(1));

    ICLKernel::configure_internal(win);

    std::stringstream kernel_name;
    CLBuildOptions    build_options;

    kernel_name << "indirect_convolution_address_precalculation";

    const unsigned int pad_left      = conv_info.pad_left();
    const unsigned int pad_top       = conv_info.pad_top();
    const unsigned int conv_stride_x = std::get<0>(conv_info.stride());
    const unsigned int conv_stride_y = std::get<1>(conv_info.stride());
    const auto         dst_data_type = dst->data_type();

    build_options.add_option("-DSRC_CONV_WIDTH=" + support::cpp11::to_string(src->dimension(width_idx)));
    build_options.add_option("-DSRC_CONV_HEIGHT=" + support::cpp11::to_string(src->dimension(height_idx)));
    build_options.add_option("-DDST_CONV_WIDTH=" + support::cpp11::to_string(output_conv_shape[width_idx]));
    build_options.add_option("-DDST_CONV_HEIGHT=" + support::cpp11::to_string(output_conv_shape[height_idx]));
    build_options.add_option("-DDST_TENSOR_TYPE=BUFFER");
    build_options.add_option("-DDST_DATA_TYPE=" + get_cl_type_from_data_type(dst_data_type));
    build_options.add_option("-DWEI_CONV_WIDTH=" + support::cpp11::to_string(weights->dimension(width_idx)));
    build_options.add_option("-DSTRIDE_X=" + support::cpp11::to_string(conv_stride_x));
    build_options.add_option("-DSTRIDE_Y=" + support::cpp11::to_string(conv_stride_y));
    build_options.add_option("-DPAD_LEFT=" + support::cpp11::to_string(pad_left));
    build_options.add_option("-DPAD_TOP=" + support::cpp11::to_string(pad_top));
    build_options.add_option("-DM0=" + support::cpp11::to_string(desc.m0));

    // A macro guard to compile ONLY the kernel of interest
    build_options.add_option("-D" + upper_string(kernel_name.str()));

    _kernel = create_kernel(compile_context, kernel_name.str(), build_options.options());

    // Since this kernel should be called only once, we do not need to set the config_id for tuning
}

Status ClIndirectConv2dAddressPrecalculationKernel::validate(const ITensorInfo *src, const ITensorInfo *weights, const ITensorInfo *dst,
                                                             const PadStrideInfo &conv_info, const DirectConvComputeKernelInfo &desc)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(src, weights, dst, conv_info, desc));
    return Status{};
}

void ClIndirectConv2dAddressPrecalculationKernel::run_op(ITensorPack &tensors, const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);

    // Get initial windows
    const Window slice = window.first_slice_window_3D();

    auto dst = utils::cast::polymorphic_downcast<ICLTensor *>(tensors.get_tensor(TensorType::ACL_DST));

    unsigned int idx = 0;
    add_4d_tensor_nhwc_argument(idx, dst);
    enqueue(queue, *this, slice);
}
} // namespace kernels
} // namespace opencl
} // namespace arm_compute
