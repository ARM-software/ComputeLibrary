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
#if defined(ENABLE_EXPERIMENTAL_DYNAMIC_FUSION)

#include "src/core/experimental/dynamic_fusion/ClKernelBuildingImpl/components/ClDirectConvolutionKernelComponent.h"

#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "src/core/CL/ICLKernel.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"
#include "src/gpu/cl/kernels/gemm/ClGemmHelpers.h"

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
ComponentType ClDirectConvolutionKernelComponent::get_component_type() const
{
    return ComponentType::Complex;
}

std::set<std::string> ClDirectConvolutionKernelComponent::get_headers_list() const
{
    return std::set<std::string> { "helpers.h", "tile_helpers.h", "repeat.h" };
}

Window ClDirectConvolutionKernelComponent::get_window() const
{
    const auto src_info    = _blueprint->impl().get_kernel_argument_info(_src.arg_id);
    const auto weight_info = _blueprint->impl().get_kernel_argument_info(_weight.arg_id);
    auto       dst_info    = _blueprint->impl().get_kernel_argument_info(_blueprint->impl().get_dst_id());

    // Get dst shape
    TensorShape output_shape = misc::shape_calculator::compute_deep_convolution_shape(*src_info, *weight_info, _desc.pad_stride_info);

    // Output auto initialization if not yet initialized
    auto_init_if_empty(*dst_info, output_shape,
                       1,
                       src_info->data_type(),
                       src_info->quantization_info());

    const unsigned int vec_size = std::min(static_cast<unsigned int>(dst_info->tensor_shape()[0]), 4u);
    const unsigned int num_rows = (dst_info->tensor_shape()[0] > 16) ? ((src_info->data_type() == DataType::F32) ? 2U : 4U) : 1U;

    // Create and configure kernel window
    Window win = calculate_max_window(output_shape, Steps(vec_size, num_rows));

    const size_t dim_y_collapsed = ceil_to_multiple(output_shape[1] * output_shape[2], num_rows);
    win.set(Window::DimY, Window::Dimension(0, dim_y_collapsed, num_rows));
    win.set(Window::DimZ, Window::Dimension(0, output_shape.total_size_upper(3), 1));

    return win;
}

std::string ClDirectConvolutionKernelComponent::get_additional_macros() const
{
    return R"_()_"; // no macros
}

std::string ClDirectConvolutionKernelComponent::get_component_code() const
{
    const auto src_info  = _blueprint->impl().get_kernel_argument_info(_src.arg_id);
    const auto bias_info = _blueprint->impl().get_kernel_argument_info(_bias.arg_id);

    ARM_COMPUTE_ERROR_ON_MSG(src_info->data_layout() != DataLayout::NHWC, "Only NHWC data layout is supported by this component.");

    const auto channel_idx   = get_data_layout_dimension_index(src_info->data_layout(), DataLayoutDimension::CHANNEL);
    const auto k0            = adjust_vec_size(is_data_type_quantized(src_info->data_type()) ? 16u : 8u, src_info->dimension(channel_idx));
    const bool leftover_loop = (src_info->dimension(channel_idx) % k0) != 0;

    std::string code = R"_(
    //------------------ START KERNEL {{meta_kernel_id}} ---------------------
    // IN_0(src)            {{src}}
    // IN_1(wei)            {{weight}}
    // IN_1(bia)            {{bias}}
    // OUT(dst, accum)      {{dst}}

    const int cout = GET_SPATIAL_IDX(0, N0, PARTIAL_N0); // OFM
    const int mout = GET_SPATIAL_IDX(1, M0, 0);          // WIDTH x HEIGHT
    const int bout = GET_SPATIAL_IDX(2, 1, 0);           // BATCH SIZE IDX

    // Initialize the accumulators
    TILE({{ACC_DATA_TYPE}}, M0, N0, {{dst}});
    {
        // All the tensor dimensions are passed at compile time.
        // In case of dynamic tensor support, the following dimensions should be passed as function argument.
    #define _I{{WEI_WIDTH}} {{WEI_WIDTH}}
    #define _I{{WEI_HEIGHT}} {{WEI_HEIGHT}}
    #define _ISRC_WIDTH {{src}}_w
    #define _ISRC_HEIGHT {{src}}_h
    #define _ISRC_CHANNELS {{src}}_c
    #define _IDST_WIDTH {{dst_w}}
    #define _IDST_HEIGHT {{dst_h}}
    #define _IDST_CHANNELS {{dst_c}}
    #define _IY_MULTIPLIER (_I{{WEI_WIDTH}} * _I{{WEI_HEIGHT}})

        // .v    = access the whole vector (OpenCL vector)
        // .s[x] = access the vector element at position x (scalar access)
        TILE(int, M0, 1, xi);
        TILE(int, M0, 1, yi);

        // Convert the linear index to coordinate
        LOOP_UNROLLING(int, i, 0, 1, M0,
        {
            xi[i].v = ((mout + i) % _IDST_WIDTH) * {{STRIDE_X}};
            yi[i].v = ((mout + i) / _IDST_WIDTH) * {{STRIDE_Y}};
            xi[i].v -= {{PAD_LEFT}};
            yi[i].v -= {{PAD_TOP}};
        })

        LOOP_UNROLLING(int, i, 0, 1, M0,
        {
            {{dst}}[i].v = 0;
        })

        uint cond = (get_global_id(0) == 0) && (get_global_id(1) == 0) && (get_global_id(2) == 0);

        for(int i = 0; i < (_I{{WEI_WIDTH}} * _I{{WEI_HEIGHT}}); ++i)
        {
            int ck = 0;
            int xk = i % _I{{WEI_WIDTH}};
            int yk = i / _I{{WEI_WIDTH}};

            int k = 0;
            for(; k <= (_ISRC_CHANNELS - K0); k += K0)
            {
                TILE({{SRC_DATA_TYPE}}, M0, K0, a);
                TILE({{WEI_DATA_TYPE}}, N0, K0, b);

                LOOP_UNROLLING(int, i, 0, 1, M0,
                {
                    a[i].v = {{ZERO_VALUE}};
                })

                // Load tile from the src tensor
                T_LOAD_NHWC_INDIRECT({{SRC_DATA_TYPE}}, M0, K0, {{SRC_TENSOR_TYPE}}, {{src}}, bout, yk, xk, ck, _ISRC_WIDTH, _ISRC_HEIGHT, {{src}}_stride_y, xi, yi, a);

                // Load tile from the weights tensor
                T_LOAD({{WEI_DATA_TYPE}}, N0, K0, {{WEI_TENSOR_TYPE}}, {{weight}}, ck, cout * _IY_MULTIPLIER + i, _IY_MULTIPLIER, {{weight}}_stride_y, b);

                // Compute the matrix multiplication between two tiles
                T_MMUL({{SRC_DATA_TYPE}}, {{WEI_DATA_TYPE}}, {{ACC_DATA_TYPE}}, M0, N0, K0, NT, T, a, b, {{dst}});

                ck += K0;
            }

            // We voluntarily use SRC_CHANNELS rather than _DSRC_CHANNELS
            // This #if directive should be removed in case of dynamic tensor support
    )_";

    if(leftover_loop)
    {
        code += R"_(
            // Left-over accumulations
            for(; k < _ISRC_CHANNELS; ++k)
            {
                TILE({{SRC_DATA_TYPE}}, M0, 1, a);
                TILE({{WEI_DATA_TYPE}}, N0, 1, b);

                LOOP_UNROLLING(int, i, 0, 1, M0,
                {
                    a[i].v = {{ZERO_VALUE}};
                })

                // Load tile from the src tensor
                T_LOAD_NHWC_INDIRECT({{SRC_DATA_TYPE}}, M0, 1, {{SRC_TENSOR_TYPE}}, {{src}}, bout, yk, xk, ck, _ISRC_WIDTH, _ISRC_HEIGHT, {{src}}_stride_y, xi, yi, a);

                // Load tile from the weights tensor
                // The T_LOAD for the left-over elements can only use BUFFER because we load one element per iteration
                T_LOAD({{WEI_DATA_TYPE}}, N0, 1, BUFFER, {{weight}}, ck, cout * _IY_MULTIPLIER + i, _IY_MULTIPLIER, {{weight}}_stride_y, b);

                // Compute the matrix multiplication between two tiles
                T_MMUL({{SRC_DATA_TYPE}}, {{WEI_DATA_TYPE}}, {{ACC_DATA_TYPE}}, M0, N0, 1, NT, T, a, b, {{dst}});

                ++ck;
            }
        )_";
    }

    code += R"_(
        }
    )_";

    if(bias_info != nullptr)
    {
        code += R"_(
            TILE({{BIA_DATA_TYPE}}, 1, N0, bias0);

            T_LOAD({{BIA_DATA_TYPE}}, 1, N0, BUFFER, {{bias}}, cout, 0, 1, 0, bias0);

            // c = c + bias[broadcasted]
            T_ADD_BROADCAST_X({{ACC_DATA_TYPE}}, M0, N0, {{dst}}, bias0, {{dst}});
        )_";
    }

    code += R"_(
    #undef _I{{WEI_WIDTH}}
    #undef _I{{WEI_HEIGHT}}
    #undef _ISRC_WIDTH
    #undef _ISRC_HEIGHT
    #undef _ISRC_CHANNELS
    #undef _IDST_WIDTH
    #undef _IDST_HEIGHT
    #undef _IDST_CHANNELS
    #undef _IY_MULTIPLIER
    }

    // Workaround for the discrepancy between tiles and repeats
    VEC_DATA_TYPE({{ACC_DATA_TYPE}}, N0) {{dst}}0 = {{dst}}[0].v;
#if M0 >= 2
    VEC_DATA_TYPE({{ACC_DATA_TYPE}}, N0) {{dst}}1 = {{dst}}[1].v;
#endif // M0 >= 2
#if M0 >= 3
    VEC_DATA_TYPE({{ACC_DATA_TYPE}}, N0) {{dst}}2 = {{dst}}[2].v;
#endif // M0 >= 3
#if M0 >= 4
    VEC_DATA_TYPE({{ACC_DATA_TYPE}}, N0) {{dst}}3 = {{dst}}[3].v;
#endif // M0 >= 4
#if M0 >= 8
    VEC_DATA_TYPE({{ACC_DATA_TYPE}}, N0) {{dst}}4 = {{dst}}[4].v;
    VEC_DATA_TYPE({{ACC_DATA_TYPE}}, N0) {{dst}}5 = {{dst}}[5].v;
    VEC_DATA_TYPE({{ACC_DATA_TYPE}}, N0) {{dst}}6 = {{dst}}[6].v;
    VEC_DATA_TYPE({{ACC_DATA_TYPE}}, N0) {{dst}}7 = {{dst}}[7].v;
#endif // M0 >= 8
#if M0 == 16
    VEC_DATA_TYPE({{ACC_DATA_TYPE}}, N0) {{dst}}8 = {{dst}}[8].v;
    VEC_DATA_TYPE({{ACC_DATA_TYPE}}, N0) {{dst}}9 = {{dst}}[9].v;
    VEC_DATA_TYPE({{ACC_DATA_TYPE}}, N0) {{dst}}A = {{dst}}[10].v;
    VEC_DATA_TYPE({{ACC_DATA_TYPE}}, N0) {{dst}}B = {{dst}}[11].v;
    VEC_DATA_TYPE({{ACC_DATA_TYPE}}, N0) {{dst}}C = {{dst}}[12].v;
    VEC_DATA_TYPE({{ACC_DATA_TYPE}}, N0) {{dst}}D = {{dst}}[13].v;
    VEC_DATA_TYPE({{ACC_DATA_TYPE}}, N0) {{dst}}E = {{dst}}[14].v;
    VEC_DATA_TYPE({{ACC_DATA_TYPE}}, N0) {{dst}}F = {{dst}}[15].v;
#endif // M0 == 16
//------------------ END KERNEL {{meta_kernel_id}} ---------------------
    )_";
    return code.c_str();
}

bool export_to_cl_image_support(const ITensorInfo *tensor, GPUTarget gpu_target, DataLayout data_layout)
{
    if(tensor->tensor_shape()[0] % 4 || (data_layout != DataLayout::NHWC))
    {
        return false;
    }

    // If not floating point
    if(!is_data_type_float(tensor->data_type()))
    {
        return false;
    }

    if(gpu_target == GPUTarget::G71 || get_arch_from_target(gpu_target) == GPUTarget::MIDGARD)
    {
        return false;
    }

    // Check if the cl_khr_image2d_from_buffer extension is supported on the target platform
    if(!image2d_from_buffer_supported(CLKernelLibrary::get().get_device()))
    {
        return false;
    }

    // Check cl image pitch alignment
    if(get_cl_image_pitch_alignment(CLKernelLibrary::get().get_device()) == 0)
    {
        return false;
    }

    const size_t image_w     = tensor->tensor_shape()[0] / 4;
    const size_t image_h     = tensor->tensor_shape()[1] * tensor->tensor_shape()[2] * tensor->tensor_shape()[3];
    const size_t max_image_w = CLKernelLibrary::get().get_device().getInfo<CL_DEVICE_IMAGE2D_MAX_WIDTH>();
    const size_t max_image_h = CLKernelLibrary::get().get_device().getInfo<CL_DEVICE_IMAGE2D_MAX_HEIGHT>();

    if(image_w > max_image_w || image_h > max_image_h)
    {
        return false;
    }

    return true;
}

CLBuildOptions ClDirectConvolutionKernelComponent::generate_build_options() const
{
    const auto src_info    = _blueprint->impl().get_kernel_argument_info(_src.arg_id);
    const auto weight_info = _blueprint->impl().get_kernel_argument_info(_weight.arg_id);
    const auto dst_info    = _blueprint->impl().get_kernel_argument_info(_blueprint->impl().get_dst_id());

    const unsigned int channel_idx = get_data_layout_dimension_index(src_info->data_layout(), DataLayoutDimension::CHANNEL);
    const DataType     data_type   = src_info->data_type();
    const GPUTarget    gpu_target  = ICLKernel().get_target();

    Window win = get_window();

    const unsigned int n0                 = win.x().step();
    const unsigned int m0                 = win.y().step();
    const unsigned int k0                 = adjust_vec_size(is_data_type_quantized(data_type) ? 16u : 8u, src_info->dimension(channel_idx));
    const unsigned int partial_store_n0   = dst_info->dimension(channel_idx) % n0;
    const bool         export_to_cl_image = export_to_cl_image_support(weight_info, gpu_target, src_info->data_layout());

    // Update the padding for the weights tensor if we can export to cl_image
    if(export_to_cl_image)
    {
        arm_compute::opencl::kernels::gemm::update_padding_for_cl_image(weight_info);
    }

    CLBuildOptions build_opts{};
    build_opts.add_option("-cl-fast-relaxed-math");
    build_opts.add_option("-DIS_TILED");
    build_opts.add_option("-DN0=" + support::cpp11::to_string(n0));
    build_opts.add_option("-DM0=" + support::cpp11::to_string(m0));
    build_opts.add_option("-DK0=" + support::cpp11::to_string(k0));
    build_opts.add_option("-DPARTIAL_N0=" + support::cpp11::to_string(partial_store_n0));

    return build_opts;
}

ClDirectConvolutionKernelComponent::TagLUT ClDirectConvolutionKernelComponent::allocate_vars(SharedVarTable &vtable) const
{
    TagLUT lut{};

    const auto src_info    = _blueprint->impl().get_kernel_argument_info(_src.arg_id);
    const auto weight_info = _blueprint->impl().get_kernel_argument_info(_weight.arg_id);
    const auto bias_info   = _blueprint->impl().get_kernel_argument_info(_bias.arg_id);
    const auto dst_info    = _blueprint->impl().get_kernel_argument_info(_blueprint->impl().get_dst_id());

    const GPUTarget gpu_target         = ICLKernel().get_target();
    const bool      export_to_cl_image = export_to_cl_image_support(weight_info, gpu_target, src_info->data_layout());

    const TensorArgType weight_type = export_to_cl_image ? TensorArgType::Tensor_4D_t_Image : TensorArgType::Tensor_4D_t_Buffer;
    lut["meta_kernel_id"]           = id();
    lut["src"]                      = vtable.add(_src, ClKernelArgRuntimeDescriptor(_src.arg_id, TensorArgType::Tensor_4D_t_Buffer), "src");
    lut["weight"]                   = vtable.add(_weight, ClKernelArgRuntimeDescriptor(_weight.arg_id, weight_type), "weight");

    if(!_bias.is_empty()) // optional bias
    {
        lut["bias"]          = vtable.add(_bias, ClKernelArgRuntimeDescriptor(_bias.arg_id, TensorArgType::Vector), "bias");
        lut["BIA_DATA_TYPE"] = get_cl_type_from_data_type(bias_info->data_type());
    }
    lut["dst"] = vtable.add(_dst, ClKernelArgRuntimeDescriptor(_dst.arg_id, TensorArgType::Tensor_4D_t_Buffer), "dst");

    // Local build options
    const auto width_idx   = get_data_layout_dimension_index(src_info->data_layout(), DataLayoutDimension::WIDTH);
    const auto height_idx  = get_data_layout_dimension_index(src_info->data_layout(), DataLayoutDimension::HEIGHT);
    const auto channel_idx = get_data_layout_dimension_index(src_info->data_layout(), DataLayoutDimension::CHANNEL);

    lut["dst_w"] = dst_info->dimension(width_idx);
    lut["dst_h"] = dst_info->dimension(height_idx);
    lut["dst_c"] = dst_info->dimension(channel_idx);

    lut["ACC_DATA_TYPE"] = src_info->data_type();
    lut["SRC_DATA_TYPE"] = src_info->data_type();
    lut["WEI_DATA_TYPE"] = weight_info->data_type();

    lut["SRC_TENSOR_TYPE"] = "BUFFER";
    lut["WEI_TENSOR_TYPE"] = export_to_cl_image ? "IMAGE" : "BUFFER";

    lut["WEI_WIDTH"]  = weight_info->dimension(width_idx);
    lut["WEI_HEIGHT"] = weight_info->dimension(height_idx);

    lut["STRIDE_X"] = std::get<0>(_desc.pad_stride_info.stride());
    lut["STRIDE_Y"] = std::get<1>(_desc.pad_stride_info.stride());

    lut["PAD_LEFT"] = _desc.pad_stride_info.pad_left();
    lut["PAD_TOP"]  = _desc.pad_stride_info.pad_top();

    lut["ZERO_VALUE"] = 0;

    return lut;
}
} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute

#endif // defined(ENABLE_EXPERIMENTAL_DYNAMIC_FUSION)