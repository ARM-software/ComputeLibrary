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
#include "src/gpu/cl/kernels/ClIndirectConv2dKernel.h"

#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/KernelDescriptors.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "src/core/CL/CLUtils.h"
#include "src/core/CL/CLValidate.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"
#include "src/gpu/cl/kernels/gemm/ClGemmHelpers.h"
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
Status validate_arguments(const ITensorInfo *src, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *indirect_buffer, const ITensorInfo *dst,
                          const PadStrideInfo &conv_info, const ActivationLayerInfo &act_info, const DirectConvComputeKernelInfo &desc)
{
    ARM_COMPUTE_UNUSED(act_info);
    ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(src);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(src, 1, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(indirect_buffer, 1, DataType::S32);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_LAYOUT_NOT_IN(src, DataLayout::NHWC);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src, weights);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(indirect_buffer->tensor_shape(),
                                                       misc::shape_calculator::compute_indirect_buffer_shape(src->tensor_shape(),
                                                                                                             src->data_layout(),
                                                                                                             weights->tensor_shape(),
                                                                                                             conv_info,
                                                                                                             desc));

    constexpr int channel_idx = 0;
    constexpr int batch_idx   = 3;

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(weights->dimension(channel_idx) != src->dimension(channel_idx), "Weights feature map dimension should match the respective src's one");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(weights->num_dimensions() > 4, "Weights can be at most 4 dimensional");

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(desc.m0 <= 0 || desc.m0 > 8, "M0 can only be greater than 0 and less than or equal to 8");

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(desc.n0 != 1 && desc.n0 != 2 && desc.n0 != 3 && desc.n0 != 4 && desc.n0 != 8 && desc.n0 != 16,
                                    "N0 can only be: 1, 2, 3, 4, 8, and 16");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(desc.k0 != 1 && desc.k0 != 2 && desc.k0 != 3 && desc.k0 != 4 && desc.k0 != 8 && desc.k0 != 16,
                                    "K0 can only be: 1, 2, 3, 4, 8, and 16");

    if(desc.export_weights_to_cl_image)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(desc.k0 != 4 && desc.k0 != 8 && desc.k0 != 16,
                                        "K0 can only be: 4, 8, and 16");
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(!export_to_cl_image(weights),
                                        "Export to CLImage is not supported for this weight configuration");
    }

    if(biases != nullptr)
    {
        if(is_data_type_quantized_asymmetric(src->data_type()))
        {
            ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(biases, 1, DataType::S32);
        }
        else
        {
            ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(weights, biases);
        }
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(biases->dimension(channel_idx) != weights->dimension(batch_idx),
                                        "Biases size and number of dst feature maps should match");
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(biases->num_dimensions() > 1,
                                        "Biases should be one dimensional");
    }

    // Checks performed when dst is configured
    if(dst->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(dst->tensor_shape(),
                                                           misc::shape_calculator::compute_deep_convolution_shape(*src, *weights, conv_info));
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src, dst);
    }

    return Status{};
}
} // namespace

ClIndirectConv2dKernel::ClIndirectConv2dKernel()
{
    _type = CLKernelType::DIRECT;
}

void ClIndirectConv2dKernel::configure(const CLCompileContext &compile_context, ITensorInfo *src, ITensorInfo *weights, ITensorInfo *biases, ITensorInfo *indirect_buffer, ITensorInfo *dst,
                                       const PadStrideInfo &conv_info, const ActivationLayerInfo &act_info, const DirectConvComputeKernelInfo &desc)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src, weights, indirect_buffer, dst);

    // Perform validation
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(src, weights, biases, indirect_buffer, dst, conv_info, act_info, desc));

    constexpr unsigned int channel_idx   = 0;
    constexpr unsigned int width_idx     = 1;
    constexpr unsigned int height_idx    = 2;
    const unsigned int     kernel_width  = weights->dimension(width_idx);
    const unsigned int     kernel_height = weights->dimension(height_idx);
    const DataType         data_type     = src->data_type();

    const GPUTarget gpu_target = get_target();

    // Get dst shape
    TensorShape output_shape = misc::shape_calculator::compute_deep_convolution_shape(*src, *weights, conv_info);

    // Output auto inizialitation if not yet initialized
    auto_init_if_empty(*dst, output_shape,
                       1,
                       src->data_type(),
                       src->quantization_info());

    // Configure kernel window
    Window win;
    output_shape.collapse(2U, 1U);
    const unsigned int n0 = adjust_vec_size(desc.n0, output_shape[0]);
    const unsigned int m0 = adjust_vec_size(desc.m0, output_shape[1]);
    const unsigned int k0 = adjust_vec_size(desc.k0, src->dimension(channel_idx));

    const unsigned int partial_store_n0 = dst->dimension(channel_idx) % n0;

    // Create window and update padding
    win = calculate_max_window(output_shape, Steps(n0, m0));

    ICLKernel::configure_internal(win);

    std::stringstream kernel_name;
    CLBuildOptions    build_options;

    kernel_name << "indirect_convolution_nhwc";

    _export_to_cl_image = desc.export_weights_to_cl_image;

    // Update the padding for the weights tensor if we can export to cl_image
    if(_export_to_cl_image)
    {
        gemm::update_padding_for_cl_image(weights);
    }

    // Add padding to indirect buffer to avoid out-of-bound reads
    // When M0 is 5, 6, and 7, we use vload8 to fetch the data from the buffer
    const unsigned int load_indirect_buf_size = m0 > 4 ? 8 : m0;
    const unsigned int indirect_buf_width     = indirect_buffer->tensor_shape()[0];
    const unsigned int round_up_width         = ((indirect_buf_width + load_indirect_buf_size - 1) / load_indirect_buf_size) * load_indirect_buf_size;
    const unsigned int padding                = round_up_width - indirect_buf_width;
    indirect_buffer->extend_padding(PaddingSize(0, indirect_buffer->padding().right + padding, 0, 0));

    if(biases != nullptr)
    {
        build_options.add_option(std::string("-DHAS_BIAS"));
        build_options.add_option(std::string("-DBIA_DATA_TYPE=" + get_cl_type_from_data_type(biases->data_type())));
    }

    // Conditions of -cl-fast-relaxed-math causing accuracy issues can be traced from COMPMID-5324
    const auto act_function = act_info.activation();

    if((gpu_target != GPUTarget::G71 && (gpu_target & GPUTarget::GPU_ARCH_MASK) == GPUTarget::BIFROST)
       && (act_function == ActivationLayerInfo::ActivationFunction::BOUNDED_RELU || act_function == ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU)
       && (data_type == DataType::F32 || data_type == DataType::F16))
    {
        // -cl-fast-relaxed-math also sets -cl-finite-math-only and -cl-unsafe-math-optimizations
        // to disable -cl-finite-math-only, we only include -cl-unsafe-math-optimizations
        build_options.add_option("-cl-unsafe-math-optimizations");
    }
    else
    {
        build_options.add_option("-cl-fast-relaxed-math");
    }

    build_options.add_option("-DSRC_TENSOR_TYPE=BUFFER");
    build_options.add_option("-DSRC_DATA_TYPE=" + get_cl_type_from_data_type(data_type));
    build_options.add_option("-DSRC_CHANNELS=" + support::cpp11::to_string(src->dimension(channel_idx)));
    build_options.add_option("-DOFF_TENSOR_TYPE=BUFFER");
    build_options.add_option("-DDST_WIDTH=" + support::cpp11::to_string(dst->dimension(width_idx)));
    build_options.add_option("-DDST_HEIGHT=" + support::cpp11::to_string(dst->dimension(height_idx)));
    build_options.add_option("-DDST_TENSOR_TYPE=BUFFER");
    build_options.add_option("-DDST_DATA_TYPE=" + get_cl_type_from_data_type(data_type));
    build_options.add_option_if_else(_export_to_cl_image, "-DWEI_TENSOR_TYPE=IMAGE", "-DWEI_TENSOR_TYPE=BUFFER");
    build_options.add_option("-DWEI_WIDTH=" + support::cpp11::to_string(kernel_width));
    build_options.add_option("-DWEI_HEIGHT=" + support::cpp11::to_string(kernel_height));
    build_options.add_option("-DWEI_DATA_TYPE=" + get_cl_type_from_data_type(data_type));
    build_options.add_option("-DN0=" + support::cpp11::to_string(n0));
    build_options.add_option("-DM0=" + support::cpp11::to_string(m0));
    build_options.add_option("-DK0=" + support::cpp11::to_string(k0));
    build_options.add_option("-DPARTIAL_N0=" + support::cpp11::to_string(partial_store_n0));
    build_options.add_option("-DIND_BUFF_VEC_SIZE=" + support::cpp11::to_string(load_indirect_buf_size));
    build_options.add_option_if((src->dimension(channel_idx) % k0) != 0, "-DLEFTOVER_LOOP");
    build_options.add_option("-DACTIVATION_TYPE=" + lower_string(string_from_activation_func(act_function)));
    build_options.add_option_if(act_info.enabled(), "-DA_VAL=" + float_to_string_with_full_precision(act_info.a()));
    build_options.add_option_if(act_info.enabled(), "-DB_VAL=" + float_to_string_with_full_precision(act_info.b()));

    // A macro guard to compile ONLY the kernel of interest
    build_options.add_option("-D" + upper_string(kernel_name.str()));

    if(compile_context.get_ddk_version() >= 30)
    {
        build_options.add_option("-fregister-allocation=64");
    }

    _kernel = create_kernel(compile_context, kernel_name.str(), build_options.options());

    // Set config_id for enabling LWS tuning
    _config_id = kernel_name.str();
    _config_id += "_";
    _config_id += lower_string(string_from_data_type(data_type));
    _config_id += "_";
    _config_id += support::cpp11::to_string(kernel_width);
    _config_id += "_";
    _config_id += support::cpp11::to_string(kernel_height);
    _config_id += "_";
    _config_id += support::cpp11::to_string(src->dimension(width_idx));
    _config_id += "_";
    _config_id += support::cpp11::to_string(src->dimension(height_idx));
    _config_id += "_";
    _config_id += support::cpp11::to_string(src->dimension(channel_idx));
    _config_id += "_";
    _config_id += support::cpp11::to_string(dst->dimension(width_idx));
    _config_id += "_";
    _config_id += support::cpp11::to_string(dst->dimension(height_idx));
    _config_id += "_";
    _config_id += support::cpp11::to_string(dst->dimension(channel_idx));
}

Status ClIndirectConv2dKernel::validate(const ITensorInfo *src, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *indirect_buffer, const ITensorInfo *dst,
                                        const PadStrideInfo &conv_info, const ActivationLayerInfo &act_info, const DirectConvComputeKernelInfo &desc)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(src, weights, biases, indirect_buffer, dst, conv_info, act_info, desc));
    return Status{};
}

void ClIndirectConv2dKernel::run_op(ITensorPack &tensors, const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);

    // Get initial windows
    Window slice = window.first_slice_window_3D();

    const auto src             = utils::cast::polymorphic_downcast<const ICLTensor *>(tensors.get_const_tensor(TensorType::ACL_SRC_0));
    const auto weights         = utils::cast::polymorphic_downcast<const ICLTensor *>(tensors.get_const_tensor(TensorType::ACL_SRC_1));
    const auto biases          = utils::cast::polymorphic_downcast<const ICLTensor *>(tensors.get_const_tensor(TensorType::ACL_SRC_2));
    const auto indirect_buffer = utils::cast::polymorphic_downcast<const ICLTensor *>(tensors.get_const_tensor(TensorType::ACL_SRC_3));
    auto       dst             = utils::cast::polymorphic_downcast<ICLTensor *>(tensors.get_tensor(TensorType::ACL_DST));

    cl::Image2D weights_cl_image;

    if(_export_to_cl_image)
    {
        const size_t      image_w = weights->info()->dimension(0) / 4;
        const size_t      image_h = weights->info()->dimension(1) * weights->info()->dimension(2) * weights->info()->dimension(3);
        const TensorShape shape2d(image_w, image_h);
        const size_t      image_row_pitch = weights->info()->strides_in_bytes()[1];

        // Export cl_buffer to cl_image
        weights_cl_image = create_image2d_from_buffer(CLKernelLibrary::get().context(), weights->cl_buffer(), shape2d, weights->info()->data_type(), image_row_pitch);
    }

    unsigned int idx = 0;
    add_4d_tensor_nhwc_argument(idx, src);
    add_4d_tensor_nhwc_argument(idx, indirect_buffer);
    add_4d_tensor_nhwc_argument(idx, dst);
    if(_export_to_cl_image)
    {
        _kernel.setArg(idx++, weights_cl_image);
    }
    add_4d_tensor_nhwc_argument(idx, weights);
    if(biases != nullptr)
    {
        add_1D_tensor_argument(idx, biases, slice);
    }
    enqueue(queue, *this, slice, lws_hint());
}
} // namespace kernels
} // namespace opencl
} // namespace arm_compute
