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

#include "src/core/experimental/dynamic_fusion/ClKernelBuildingImpl/components/ClGemmNativeKernelComponent.h"
#include "arm_compute/core/TensorInfo.h"
#include "src/core/AccessWindowStatic.h"
#include "src/core/helpers/WindowHelpers.h"

#include "src/core/utils/helpers/float_ops.h"
#include "support/StringSupport.h"

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
ComponentType ClGemmNativeKernelComponent::get_component_type() const
{
    return ComponentType::Complex;
}

std::set<std::string> ClGemmNativeKernelComponent::get_headers_list() const
{
    return std::set<std::string> { "common/experimental/gemm_fused_post_ops/act_eltwise_op_act/fp_post_ops_act_eltwise_op_act.h", "gemm_helpers.h", "repeat.h" };
}

Window ClGemmNativeKernelComponent::get_window() const
{
    ITensorInfo *lhs_info  = _blueprint->impl().get_kernel_argument_info(_lhs.arg_id);
    ITensorInfo *rhs_info  = _blueprint->impl().get_kernel_argument_info(_rhs.arg_id);
    ITensorInfo *bias_info = _blueprint->impl().get_kernel_argument_info(_bias.arg_id);
    ITensorInfo *dst_info  = _blueprint->impl().get_kernel_argument_info(_blueprint->impl().get_dst_id());

    ARM_COMPUTE_ERROR_ON_NULLPTR(lhs_info, rhs_info, dst_info);

    bool reinterpret_input_as_3d  = _desc.reinterpret_input_as_3d;
    bool reinterpret_output_as_3d = _desc.depth_output_gemm3d != 0;

    Window win{};
    Window win_out{};
    bool   window_changed = false;

    // In case both input and dst have to be reinterpreted as 3D tensors,
    // force reinterpret_input_as_3d and reinterpret_output_as_3d to be false.
    if(reinterpret_input_as_3d == reinterpret_output_as_3d)
    {
        reinterpret_output_as_3d = false;
    }

    // activation_layer is set to dummy because it's required by GEMMKernelInfo, but it's not used in shape calculation
    GEMMKernelInfo gemm_info(_desc.m, _desc.n, _desc.k, _desc.depth_output_gemm3d, _desc.reinterpret_input_as_3d,
                             _desc.broadcast_bias, _desc.fp_mixed_precision, _desc.has_pad_y, ActivationLayerInfo(), _desc.nmult_transpose1xW_width,
                             _desc.mult_interleave4x4_height, _desc.lhs_info, _desc.rhs_info, _desc.a_offset, _desc.b_offset);

    // dst tensor auto initialization if not yet initialized
    auto_init_if_empty(*dst_info, lhs_info->clone()->set_tensor_shape(misc::shape_calculator::compute_mm_shape(*lhs_info, *rhs_info, gemm_info)));

    TensorInfo tmp_info(*dst_info);

    if(reinterpret_output_as_3d)
    {
        // Since the dst tensor has to be reinterpreted as 3D and the execute window is based on a 2D GEMM,
        // the window needs to be constructed on the 2D collapsed version of the tensor
        TensorShape tmp_shape(dst_info->tensor_shape());
        tmp_shape.collapse(2U, 1U);
        tmp_info.set_tensor_shape(tmp_shape);
    }

    win     = calculate_max_window(tmp_info, Steps(_desc.rhs_info.n0, _desc.lhs_info.m0));
    win_out = calculate_max_window(*dst_info, Steps(_desc.rhs_info.n0, _desc.lhs_info.m0));

    AccessWindowStatic src0_access(lhs_info, 0, 0,
                                   lhs_info->dimension(0),
                                   lhs_info->dimension(1));
    AccessWindowStatic src1_access(rhs_info, 0, 0,
                                   ceil_to_multiple(rhs_info->dimension(0), _desc.rhs_info.n0),
                                   rhs_info->dimension(1));
    AccessWindowStatic dst_access(dst_info, 0, 0,
                                  dst_info->dimension(0),
                                  dst_info->dimension(1));

    if(bias_info != nullptr)
    {
        const int bias_processed_per_iteration_x = _desc.rhs_info.n0;

        AccessWindowStatic src2_access(bias_info, 0, 0,
                                       ceil_to_multiple(bias_info->dimension(0), bias_processed_per_iteration_x),
                                       bias_info->dimension(1));

        window_changed = update_window_and_padding(win, src0_access, src1_access, src2_access) || // window used by the execute_window_loop
                         update_window_and_padding(win_out, dst_access);                          // window used to update the padding requirements of dst tensor
    }
    else
    {
        window_changed = update_window_and_padding(win, src0_access, src1_access) || // window used by the execute_window_loop
                         update_window_and_padding(win_out, dst_access);             // window used to update the padding requirements of dst tensor
    }

    // Collapse along the Z direction
    // This collapse needs to be here in order to tune the Z dimension of LWS
    Window             collapsed             = win;
    const unsigned int dimension_to_collapse = std::min(static_cast<unsigned int>(dst_info->num_dimensions()), 2u);
    collapsed                                = win.collapse(win, dimension_to_collapse);

    if(window_changed == true)
    {
        ARM_COMPUTE_ERROR("Insufficient Padding!");
    }

    return collapsed;
}

std::string ClGemmNativeKernelComponent::get_additional_macros() const
{
    return R"_(
#define VFMA(a, b, c) \
({                    \
    c = fma(a, b, c); \
})

#if M0 == 1
#define RHS_VFMA_M0xN0(i, a, b, c)                                    \
    ({                                                                \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##0).s##i), b, (c##0)); \
    })
#elif M0 == 2 // M0 == 2
#define RHS_VFMA_M0xN0(i, a, b, c)                                    \
    ({                                                                \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##0).s##i), b, (c##0)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##1).s##i), b, (c##1)); \
    })
#elif M0 == 3 // M0 == 3
#define RHS_VFMA_M0xN0(i, a, b, c)                                    \
    ({                                                                \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##0).s##i), b, (c##0)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##1).s##i), b, (c##1)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##2).s##i), b, (c##2)); \
    })
#elif M0 == 4 // M0 == 4
#define RHS_VFMA_M0xN0(i, a, b, c)                                    \
    ({                                                                \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##0).s##i), b, (c##0)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##1).s##i), b, (c##1)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##2).s##i), b, (c##2)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##3).s##i), b, (c##3)); \
    })
#elif M0 == 5 // M0 == 5
#define RHS_VFMA_M0xN0(i, a, b, c)                                    \
    ({                                                                \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##0).s##i), b, (c##0)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##1).s##i), b, (c##1)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##2).s##i), b, (c##2)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##3).s##i), b, (c##3)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##4).s##i), b, (c##4)); \
    })
#elif M0 == 6 // M0 == 6
#define RHS_VFMA_M0xN0(i, a, b, c)                                    \
    ({                                                                \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##0).s##i), b, (c##0)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##1).s##i), b, (c##1)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##2).s##i), b, (c##2)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##3).s##i), b, (c##3)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##4).s##i), b, (c##4)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##5).s##i), b, (c##5)); \
    })
#elif M0 == 7 // M0 == 7
#define RHS_VFMA_M0xN0(i, a, b, c)                                    \
    ({                                                                \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##0).s##i), b, (c##0)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##1).s##i), b, (c##1)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##2).s##i), b, (c##2)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##3).s##i), b, (c##3)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##4).s##i), b, (c##4)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##5).s##i), b, (c##5)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##6).s##i), b, (c##6)); \
    })
#elif M0 == 8 // M0 == 8
#define RHS_VFMA_M0xN0(i, a, b, c)                                    \
    ({                                                                \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##0).s##i), b, (c##0)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##1).s##i), b, (c##1)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##2).s##i), b, (c##2)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##3).s##i), b, (c##3)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##4).s##i), b, (c##4)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##5).s##i), b, (c##5)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##6).s##i), b, (c##6)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##7).s##i), b, (c##7)); \
    })
#else // M0 not supported
#error "M0 not supported"
#endif // M0 not supported
)_";
}

std::string ClGemmNativeKernelComponent::get_component_code() const
{
    auto t_lhs_info = _blueprint->impl().get_kernel_argument_info(_lhs.arg_id);
    auto t_rhs_info = _blueprint->impl().get_kernel_argument_info(_rhs.arg_id);

    auto has_alpha               = !(helpers::float_ops::is_one(_desc.alpha));
    auto reinterpret_input_as_3d = _desc.reinterpret_input_as_3d && _desc.depth_output_gemm3d == 0;
    auto dont_slide_b            = t_rhs_info->num_dimensions() < t_lhs_info->num_dimensions();

    std::string code = R"_(
    //------------------ START KERNEL {{meta_kernel_id}} ---------------------
    // IN_0(lhs)            {{lhs}}
    // IN_1(rhs)            {{rhs}}
    )_";

    if(!_bias.is_empty())
    {
        code += R"_(
    // IN_2(bias)           {{bias}}
    )_";
    }

    code += R"_(
    // OUT(dst, accum)      {{dst}}

    // Initialize the accumulators
    REPEAT_VAR_INIT_TO_CONST(M0, VEC_DATA_TYPE(DATA_TYPE, N0), {{dst}}, 0); //VEC_DATA_TYPE(DATA_TYPE, N0)    c0=0,c1=0,c2=0,... c(M0-1)=0;
    {
#if defined(DUMMY_WORK_ITEMS)
        if((g_x * N0 >= N) || (g_y * M0 >= M))
        {
            return;
        }
#endif // defined(DUMMY_WORK_ITEMS)

        // Compute LHS matrix address
        uint lhs_offset = {{lhs}}_offset_first_element_in_bytes + COMPUTE_M0_START_ROW(g_y, M0, PARTIAL_STORE_M0) * (uint){{lhs}}_stride_y;

        // Compute RHS matrix address
        uint rhs_offset = {{rhs}}_offset_first_element_in_bytes + g_x * N0 * sizeof(DATA_TYPE);
    )_";

    if(dont_slide_b)
    {
        code += R"_(
            // Do not slide matrix B if the matrix B has 3 dimensions and matrix A more than 3
            rhs_offset += (g_z % {{MATRIX_B_DEPTH}}) * {{rhs}}_stride_z;
        )_";
    }
    else
    {
        code += R"_(
            rhs_offset += g_z * {{rhs}}_stride_z;
        )_";
    }

    code += R"_(
        REPEAT_VAR_INIT_TO_CONST(M0, uint, zlhs, 0);
    )_";

    if(reinterpret_input_as_3d)
    {
        code += R"_(
            // The plane (zlhs) is calculated dividing M (g_y * M0) by HEIGHT_GEMM3D
            CALCULATE_Z_OFFSET(M0, uint, zlhs, COMPUTE_M0_START_ROW(g_y, M0, PARTIAL_STORE_M0), {{HEIGHT_GEMM3D}}, {{DEPTH_GEMM3D}}, {{lhs}}_cross_plane_pad, {{lhs}}_stride_y);

            // Add offset for batched GEMM. The batches will be in the fourth dimension and for this reason we
            // multiply lhs_stride_z by DEPTH_GEMM3D
            lhs_offset += g_z * {{lhs}}_stride_z * {{DEPTH_GEMM3D}};
        )_";
    }
    else
    {
        code += R"_(
            // Add offset for batched GEMM
            lhs_offset += g_z * {{lhs}}_stride_z;
        )_";
    }

    code += R"_(
        int i = 0;
#if {{K0}} > 1
        for(; i <= (K - {{K0}}); i += {{K0}})
        {
            // Supported cases (M0, K0):
            // 1,2 - 1,3 - 1,4 - 1,8 - 1,16
            // 2,2 - 2,3 - 2,4 - 2,8 - 2,16
            // 3,2 - 3,3 - 3,4 - 3,8 - 3,16
            // 4,2 - 4,3 - 4,4 - 4,8 - 4,16
            // 5,2 - 5,3 - 5,4 - 5,8 - 5,16
            // 6,2 - 6,3 - 6,4 - 6,8 - 6,16
            // 7,2 - 7,3 - 7,4 - 7,8 - 7,16
            // 8,2 - 8,3 - 8,4 - 8,8 - 8,16
            // Load values from LHS matrix
            LOAD_BLOCK(M0, {{K0}}, DATA_TYPE, a, {{lhs}}_ptr, lhs_offset, {{lhs}}_stride_y, zlhs);

            // Load values from RHS matrix
            LOAD_BLOCK({{K0}}, N0, DATA_TYPE, b, {{rhs}}_ptr, rhs_offset, {{rhs}}_stride_y, g_zero);

            RHS_VFMA_M0xN0(0, a, b0, {{dst}});
            RHS_VFMA_M0xN0(1, a, b1, {{dst}});
#if {{K0}} > 2
            RHS_VFMA_M0xN0(2, a, b2, {{dst}});
#endif // K0 > 2
#if {{K0}} > 3
            RHS_VFMA_M0xN0(3, a, b3, {{dst}});
#endif // K0 > 3
#if {{K0}} > 4
            RHS_VFMA_M0xN0(4, a, b4, {{dst}});
            RHS_VFMA_M0xN0(5, a, b5, {{dst}});
            RHS_VFMA_M0xN0(6, a, b6, {{dst}});
            RHS_VFMA_M0xN0(7, a, b7, {{dst}});
#endif // K0 > 4
#if {{K0}} > 8
            RHS_VFMA_M0xN0(8, a, b8, {{dst}});
            RHS_VFMA_M0xN0(9, a, b9, {{dst}});
            RHS_VFMA_M0xN0(A, a, bA, {{dst}});
            RHS_VFMA_M0xN0(B, a, bB, {{dst}});
            RHS_VFMA_M0xN0(C, a, bC, {{dst}});
            RHS_VFMA_M0xN0(D, a, bD, {{dst}});
            RHS_VFMA_M0xN0(E, a, bE, {{dst}});
            RHS_VFMA_M0xN0(F, a, bF, {{dst}});
#endif // K0 > 8

            lhs_offset += {{K0}} * sizeof(DATA_TYPE);
            rhs_offset += {{K0}} * {{rhs}}_stride_y;
        }
#endif // K0 > 1
        // Left-over accumulations
        for(; i < K; ++i)
        {
            // Load values from LHS matrix
            VEC_DATA_TYPE(DATA_TYPE, 2)
            a0 = *((__global DATA_TYPE *)({{lhs}}_ptr + lhs_offset + 0 * {{lhs}}_stride_y + zlhs0));
#if M0 > 1
            VEC_DATA_TYPE(DATA_TYPE, 2)
            a1 = *((__global DATA_TYPE *)({{lhs}}_ptr + lhs_offset + 1 * {{lhs}}_stride_y + zlhs1));
#endif // M0 > 1
#if M0 > 2
            VEC_DATA_TYPE(DATA_TYPE, 2)
            a2 = *((__global DATA_TYPE *)({{lhs}}_ptr + lhs_offset + 2 * {{lhs}}_stride_y + zlhs2));
#endif // M0 > 2
#if M0 > 3
            VEC_DATA_TYPE(DATA_TYPE, 2)
            a3 = *((__global DATA_TYPE *)({{lhs}}_ptr + lhs_offset + 3 * {{lhs}}_stride_y + zlhs3));
#endif // M0 > 3
#if M0 > 4
            VEC_DATA_TYPE(DATA_TYPE, 2)
            a4 = *((__global DATA_TYPE *)({{lhs}}_ptr + lhs_offset + 4 * {{lhs}}_stride_y + zlhs4));
#endif // M0 > 4
#if M0 > 5
            VEC_DATA_TYPE(DATA_TYPE, 2)
            a5 = *((__global DATA_TYPE *)({{lhs}}_ptr + lhs_offset + 5 * {{lhs}}_stride_y + zlhs5));
#endif // M0 > 5
#if M0 > 6
            VEC_DATA_TYPE(DATA_TYPE, 2)
            a6 = *((__global DATA_TYPE *)({{lhs}}_ptr + lhs_offset + 6 * {{lhs}}_stride_y + zlhs6));
#endif // M0 > 6
#if M0 > 7
            VEC_DATA_TYPE(DATA_TYPE, 2)
            a7 = *((__global DATA_TYPE *)({{lhs}}_ptr + lhs_offset + 7 * {{lhs}}_stride_y + zlhs7));
#endif // M0 > 7

            VEC_DATA_TYPE(DATA_TYPE, N0)
            b = VLOAD(N0)(0, (__global DATA_TYPE *)({{rhs}}_ptr + rhs_offset + 0 * {{rhs}}_stride_y));
            RHS_VFMA_M0xN0(0, a, b, {{dst}});

            lhs_offset += sizeof(DATA_TYPE);
            rhs_offset += {{rhs}}_stride_y;
        }

        // Multiply by the weight of matrix-matrix product and store the result
    )_";
    if(has_alpha)
    {
        code += R"_(
            SCALE_BLOCK(M0, DATA_TYPE, {{dst}}, {{ALPHA}});
        )_";
    }

    if(!_bias.is_empty())
    {
        if(_desc.broadcast_bias)
        {
            code += R"_(
                // Add beta*bias
                __global uchar *bias_addr = {{bias}}_ptr + {{bias}}_offset_first_element_in_bytes + (get_global_id(0) * (uint)N0 * sizeof(DATA_TYPE));

                LOAD_BLOCK(1, N0, DATA_TYPE, bias, bias_addr, 0, {{bias}}_stride_y, g_zero);
            )_";

            if(helpers::float_ops::is_one(_desc.beta))
            {
                code += R"_(
                    SCALE_BLOCK(1, DATA_TYPE, bias, {{BETA}});
                )_";
            }

            code += R"_(
                // c = c + bias[broadcasted]
                ADD_BLOCK_BROADCAST(M0, {{dst}}, bias0);
            )_";
        }
        else
        {
            code += R"_(
                // Add beta*bias
                __global uchar *bias_addr = {{bias}}_ptr + {{bias}}_offset_first_element_in_bytes + (g_x * (uint)N0 * sizeof(DATA_TYPE)) + (COMPUTE_M0_START_ROW(g_y, M0,
                                            PARTIAL_STORE_M0)
                                            * {{bias}}_stride_y)
                                            + g_z * {{bias}}_stride_z;

                LOAD_BLOCK(M0, N0, DATA_TYPE, bias, bias_addr, 0, {{bias}}_stride_y, g_zero);
            )_";

            if(helpers::float_ops::is_one(_desc.beta))
            {
                code += R"_(
                    SCALE_BLOCK(M0, DATA_TYPE, bias, {{BETA}});
                )_";
            }

            code += R"_(
                // c = c + bias
                ADD_BLOCK(M0, {{dst}}, bias);
            )_";
        }
    }

    code += R"_(
    }
    //------------------ END KERNEL {{meta_kernel_id}} ---------------------
    )_";
    return code.c_str();
}

CLBuildOptions ClGemmNativeKernelComponent::generate_build_options() const
{
    auto t_dst_info = _blueprint->impl().get_kernel_argument_info(_blueprint->impl().get_dst_id());
    auto tile_info  = _blueprint->impl().get_tile_info();

    CLBuildOptions build_opts{};

    build_opts.add_option("-DDATA_TYPE=" + get_cl_type_from_data_type(t_dst_info->data_type()));
    build_opts.add_option("-DM=" + support::cpp11::to_string(tile_info.boundaries.y()));
    build_opts.add_option("-DN=" + support::cpp11::to_string(tile_info.boundaries.x()));
    build_opts.add_option("-DK=" + support::cpp11::to_string(_desc.k));
    build_opts.add_option("-DM0=" + support::cpp11::to_string(tile_info.tile_dims.y()));
    build_opts.add_option("-DN0=" + support::cpp11::to_string(tile_info.tile_dims.x()));
    build_opts.add_option("-DPARTIAL_STORE_M0=" + support::cpp11::to_string(tile_info.boundaries.y() % tile_info.tile_dims.y()));
    build_opts.add_option("-DPARTIAL_STORE_N0=" + support::cpp11::to_string(tile_info.boundaries.x() % tile_info.tile_dims.x()));

    return build_opts;
}

std::string ClGemmNativeKernelComponent::generate_config_id() const
{
    auto        t_dst_info = _blueprint->impl().get_kernel_argument_info(_blueprint->impl().get_dst_id());
    std::string config_id{};
    config_id += (_bias.is_empty() ? "add_bias_" : "");
    config_id += (_desc.broadcast_bias ? "broadcast_bias_" : "");
    config_id += (_desc.reinterpret_input_as_3d ? "3di_" : "");
    config_id += (_desc.depth_output_gemm3d > 0 ? "3do_" : "");
    config_id += lower_string(string_from_data_type(t_dst_info->data_type()));
    config_id += "_";
    config_id += support::cpp11::to_string(t_dst_info->dimension(1));
    config_id += "_";
    config_id += support::cpp11::to_string(t_dst_info->dimension(0));
    config_id += "_";
    config_id += support::cpp11::to_string(_desc.k);
    config_id += "_";
    config_id += support::cpp11::to_string(t_dst_info->dimension(2));
    config_id += "_";
    config_id += support::cpp11::to_string(_desc.lhs_info.m0);
    config_id += "_";
    config_id += support::cpp11::to_string(_desc.rhs_info.n0);
    config_id += "_";
    config_id += support::cpp11::to_string(_desc.rhs_info.k0);
    return config_id;
}

ClGemmNativeKernelComponent::TagLUT ClGemmNativeKernelComponent::allocate_vars(SharedVarTable &vtable) const
{
    TagLUT lut{};

    lut["meta_kernel_id"] = id();
    lut["lhs"]            = vtable.add(_lhs, ClKernelArgRuntimeDescriptor(_lhs.arg_id, TensorArgType::Image_3D), "lhs");
    lut["rhs"]            = vtable.add(_rhs, ClKernelArgRuntimeDescriptor(_rhs.arg_id, TensorArgType::Image_3D), "rhs");
    if(!_bias.is_empty()) // optional bias
    {
        lut["bias"] = vtable.add(_bias, ClKernelArgRuntimeDescriptor(_bias.arg_id, TensorArgType::Image_3D), "bias");
    }
    lut["dst"] = vtable.add(_dst, ClKernelArgRuntimeDescriptor(_dst.arg_id, TensorArgType::Image_3D), "dst");

    // Local build options
    auto t_lhs_info = _blueprint->impl().get_kernel_argument_info(_lhs.arg_id);
    auto t_rhs_info = _blueprint->impl().get_kernel_argument_info(_rhs.arg_id);
    auto t_dst_info = _blueprint->impl().get_kernel_argument_info(_blueprint->impl().get_dst_id());

    auto has_alpha                = !(helpers::float_ops::is_one(_desc.alpha));
    auto has_beta                 = _blueprint->impl().get_kernel_argument_info(_bias.arg_id) != nullptr;
    auto reinterpret_input_as_3d  = _desc.reinterpret_input_as_3d && _desc.depth_output_gemm3d == 0;
    auto reinterpret_output_as_3d = !_desc.reinterpret_input_as_3d && _desc.depth_output_gemm3d != 0;
    auto dont_slide_b             = t_rhs_info->num_dimensions() < t_lhs_info->num_dimensions();

    lut["K0"] = support::cpp11::to_string(_desc.rhs_info.k0);

    if(has_alpha)
    {
        lut["ALPHA"] = float_to_string_with_full_precision(_desc.alpha);
    }
    if(has_beta)
    {
        lut["BETA"] = float_to_string_with_full_precision(_desc.beta);
    }
    if(dont_slide_b)
    {
        lut["MATRIX_B_DEPTH"] = support::cpp11::to_string(t_rhs_info->dimension(2));
    }

    if(reinterpret_output_as_3d)
    {
        lut["HEIGHT_GEMM3D"] = support::cpp11::to_string(t_dst_info->dimension(1));
        lut["DEPTH_GEMM3D"]  = support::cpp11::to_string(t_dst_info->dimension(2));
    }
    else if(reinterpret_input_as_3d)
    {
        lut["HEIGHT_GEMM3D"] = support::cpp11::to_string(t_lhs_info->dimension(1));
        lut["DEPTH_GEMM3D"]  = support::cpp11::to_string(t_lhs_info->dimension(2));
    }

    return lut;
}
} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute

#endif // defined(ENABLE_EXPERIMENTAL_DYNAMIC_FUSION)