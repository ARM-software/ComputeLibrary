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
    return std::set<std::string> { "./common/experimental/gemm_fused_post_ops/act_eltwise_op_act/fp_post_ops_act_eltwise_op_act.h", "gemm_helpers.h", "repeat.h" };
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

#if defined(MATRIX_B_DEPTH)
        // Do not slide matrix B if the matrix B has 3 dimensions and matrix A more than 3
        rhs_offset += (g_z % MATRIX_B_DEPTH) * {{rhs}}_stride_z;
#else  // defined(MATRIX_B_DEPTH)
        rhs_offset += g_z * {{rhs}}_stride_z;
#endif // defined(MATRIX_B_DEPTH)

        REPEAT_VAR_INIT_TO_CONST(M0, uint, zlhs, 0);

#if defined(REINTERPRET_INPUT_AS_3D)
        // The plane (zlhs) is calculated dividing M (g_y * M0) by HEIGHT_GEMM3D
        CALCULATE_Z_OFFSET(M0, uint, zlhs, COMPUTE_M0_START_ROW(g_y, M0, PARTIAL_STORE_M0), HEIGHT_GEMM3D, DEPTH_GEMM3D, {{lhs}}_cross_plane_pad, {{lhs}}_stride_y);

        // Add offset for batched GEMM. The batches will be in the fourth dimension and for this reason we
        // multiply lhs_stride_z by DEPTH_GEMM3D
        lhs_offset += g_z * {{lhs}}_stride_z * DEPTH_GEMM3D;

#else // defined(REINTERPRET_INPUT_AS_3D)

        // Add offset for batched GEMM
        lhs_offset += g_z * {{lhs}}_stride_z;

#endif // defined(REINTERPRET_INPUT_AS_3D)

        int i = 0;
#if K0 > 1
        for(; i <= (K - K0); i += K0)
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
            LOAD_BLOCK(M0, K0, DATA_TYPE, a, {{lhs}}_ptr, lhs_offset, {{lhs}}_stride_y, zlhs);

            // Load values from RHS matrix
            LOAD_BLOCK(K0, N0, DATA_TYPE, b, {{rhs}}_ptr, rhs_offset, {{rhs}}_stride_y, g_zero);

            RHS_VFMA_M0xN0(0, a, b0, {{dst}});
            RHS_VFMA_M0xN0(1, a, b1, {{dst}});
#if K0 > 2
            RHS_VFMA_M0xN0(2, a, b2, {{dst}});
#endif // K0 > 2
#if K0 > 3
            RHS_VFMA_M0xN0(3, a, b3, {{dst}});
#endif // K0 > 3
#if K0 > 4
            RHS_VFMA_M0xN0(4, a, b4, {{dst}});
            RHS_VFMA_M0xN0(5, a, b5, {{dst}});
            RHS_VFMA_M0xN0(6, a, b6, {{dst}});
            RHS_VFMA_M0xN0(7, a, b7, {{dst}});
#endif // K0 > 4
#if K0 > 8
            RHS_VFMA_M0xN0(8, a, b8, {{dst}});
            RHS_VFMA_M0xN0(9, a, b9, {{dst}});
            RHS_VFMA_M0xN0(A, a, bA, {{dst}});
            RHS_VFMA_M0xN0(B, a, bB, {{dst}});
            RHS_VFMA_M0xN0(C, a, bC, {{dst}});
            RHS_VFMA_M0xN0(D, a, bD, {{dst}});
            RHS_VFMA_M0xN0(E, a, bE, {{dst}});
            RHS_VFMA_M0xN0(F, a, bF, {{dst}});
#endif // K0 > 8

            lhs_offset += K0 * sizeof(DATA_TYPE);
            rhs_offset += K0 * {{rhs}}_stride_y;
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
#if defined(ALPHA)
        SCALE_BLOCK(M0, DATA_TYPE, {{dst}}, ALPHA);
#endif // defined(ALPHA)
    )_";

    if(!_bias.is_empty())
    {
        code += R"_(
        // Add beta*bias
#if defined(BROADCAST_BIAS)
        __global uchar *bias_addr = {{bias}}_ptr + {{bias}}_offset_first_element_in_bytes + (get_global_id(0) * (uint)N0 * sizeof(DATA_TYPE));

        LOAD_BLOCK(1, N0, DATA_TYPE, bias, bias_addr, 0, {{bias}}_stride_y, g_zero);

#ifndef UNIT_BETA
        SCALE_BLOCK(1, DATA_TYPE, bias, BETA);
#endif // UNIT_BIAS

        // c = c + bias[broadcasted]
        ADD_BLOCK_BROADCAST(M0, {{dst}}, bias0);

#else // defined(BROADCAST_BIAS)
        __global uchar *bias_addr = {{bias}}_ptr + {{bias}}_offset_first_element_in_bytes + (g_x * (uint)N0 * sizeof(DATA_TYPE)) + (COMPUTE_M0_START_ROW(g_y, M0,
                                    PARTIAL_STORE_M0)
                                    * {{bias}}_stride_y)
                                    + g_z * {{bias}}_stride_z;

        LOAD_BLOCK(M0, N0, DATA_TYPE, bias, bias_addr, 0, {{bias}}_stride_y, g_zero);

#ifndef UNIT_BETA
        SCALE_BLOCK(M0, DATA_TYPE, bias, BETA);
#endif // UNIT_BIAS

        // c = c + bias
        ADD_BLOCK(M0, {{dst}}, bias);

#endif // defined(BROADCAST_BIAS)
    )_";
    }

    code += R"_(
    }
    //------------------ END KERNEL {{meta_kernel_id}} ---------------------
    )_";
    return code.c_str();
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
    return lut;
}
} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute

#endif // defined(ENABLE_EXPERIMENTAL_DYNAMIC_FUSION)