/*
 * Copyright (c) 2021-2023 Arm Limited.
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
#ifndef ACL_SRC_CORE_CL_CL_KERNELS_TILE_HELPERS
#define ACL_SRC_CORE_CL_CL_KERNELS_TILE_HELPERS

// *INDENT-OFF*
// clang-format off

#define TILE_VECTOR_SIZE1 1
#define TILE_VECTOR_SIZE2 2
#define TILE_VECTOR_SIZE3 3
#define TILE_VECTOR_SIZE4 4
#define TILE_VECTOR_SIZE5 8
#define TILE_VECTOR_SIZE6 8
#define TILE_VECTOR_SIZE7 8
#define TILE_VECTOR_SIZE8 8
#define TILE_VECTOR_SIZE9 16
#define TILE_VECTOR_SIZE10 16
#define TILE_VECTOR_SIZE11 16
#define TILE_VECTOR_SIZE12 16
#define TILE_VECTOR_SIZE13 16
#define TILE_VECTOR_SIZE14 16
#define TILE_VECTOR_SIZE15 16
#define TILE_VECTOR_SIZE16 16

#define TILE_VECTOR_TYPE1(DATA_TYPE) DATA_TYPE##1
#define TILE_VECTOR_TYPE2(DATA_TYPE) DATA_TYPE##2
#define TILE_VECTOR_TYPE3(DATA_TYPE) DATA_TYPE##3
#define TILE_VECTOR_TYPE4(DATA_TYPE) DATA_TYPE##4
#define TILE_VECTOR_TYPE5(DATA_TYPE) DATA_TYPE##8
#define TILE_VECTOR_TYPE6(DATA_TYPE) DATA_TYPE##8
#define TILE_VECTOR_TYPE7(DATA_TYPE) DATA_TYPE##8
#define TILE_VECTOR_TYPE8(DATA_TYPE) DATA_TYPE##8
#define TILE_VECTOR_TYPE9(DATA_TYPE) DATA_TYPE##16
#define TILE_VECTOR_TYPE10(DATA_TYPE) DATA_TYPE##16
#define TILE_VECTOR_TYPE11(DATA_TYPE) DATA_TYPE##16
#define TILE_VECTOR_TYPE12(DATA_TYPE) DATA_TYPE##16
#define TILE_VECTOR_TYPE13(DATA_TYPE) DATA_TYPE##16
#define TILE_VECTOR_TYPE14(DATA_TYPE) DATA_TYPE##16
#define TILE_VECTOR_TYPE15(DATA_TYPE) DATA_TYPE##16
#define TILE_VECTOR_TYPE16(DATA_TYPE) DATA_TYPE##16

/** Tile object
 *  A tile object is a 2D memory block and can be accessed using the following syntax:
 *  -# a[m0].v    = access the the vector at row "m0" (OpenCL vector)
 *  -# dst[m0].s[n0] = access the scalar element at row "m0" and column "n0" (scalar access)
 *
 * @param[in] DATA_TYPE Data type of the tile
 * @param[in] H         Number of tile rows
 * @param[in] W         Number of tile colums
 * @param[in] BASENAME  Tile's name
 */
#define TILE(DATA_TYPE, H, W, BASENAME) TILE_STR(DATA_TYPE, H, W, BASENAME)
#define TILE_STR(DATA_TYPE, H, W, BASENAME) \
    union {                                 \
        DATA_TYPE                      s[TILE_VECTOR_SIZE##W];                  \
        TILE_VECTOR_TYPE##W(DATA_TYPE) v;                     \
    } BASENAME[H]

#define TENSOR4D_IMAGE(name)          \
    __read_only image2d_t name##_img, \
    __global uchar *name##_ptr,       \
    uint            name##_stride_x,  \
    uint            name##_step_x,    \
    uint            name##_stride_y,  \
    uint            name##_step_y,    \
    uint            name##_stride_z,  \
    uint            name##_step_z,    \
    uint            name##_stride_w,  \
    uint            name##_step_w,    \
    uint            name##_offset_first_element_in_bytes

#define TENSOR4D_BUFFER(name)    \
    __global uchar *name##_ptr,  \
    uint        name##_stride_x, \
    uint        name##_step_x,   \
    uint        name##_stride_y, \
    uint        name##_step_y,   \
    uint        name##_stride_z, \
    uint        name##_step_z,   \
    uint        name##_stride_w, \
    uint        name##_step_w,   \
    uint        name##_offset_first_element_in_bytes

#define TENSOR4D_STR(name, type) TENSOR4D_##type(name)
#define TENSOR4D(name, type) TENSOR4D_STR(name, type)

#define TENSOR4D_T_IMAGE(name)          \
    __read_only image2d_t name##_img, \
    __global uchar *name##_ptr,       \
    uint        name##_stride_y, \
    uint        name##_stride_z, \
    uint        name##_stride_w, \
    uint        name##_c,   \
    uint        name##_w,   \
    uint        name##_h,   \
    uint        name##_n,   \
    uint        name##_offset_first_element_in_bytes

#define TENSOR4D_T_BUFFER(name)    \
    __global uchar *name##_ptr,  \
    uint        name##_stride_y, \
    uint        name##_stride_z, \
    uint        name##_stride_w, \
    uint        name##_c,   \
    uint        name##_w,   \
    uint        name##_h,   \
    uint        name##_n,   \
    uint        name##_offset_first_element_in_bytes

#define TENSOR4D_T_STR(name, type) TENSOR4D_T_##type(name)

/** Legacy tensor 4D arguments
 *
 * @param[in] name Tensor name. The tensor name is the prefix of the tensor components
 * @param[in] type Tensor type (BUFFER or IMAGE)
 */
#define TENSOR4D_T(name, type) TENSOR4D_T_STR(name, type)

#define TENSOR4D_RO_T_IMAGE(name)          \
    __read_only image2d_t name##_img, \
    TENSOR4D_T_BUFFER(name)

#define TENSOR4D_RO_T_BUFFER(name) TENSOR4D_T_BUFFER(name)

#define TENSOR4D_RO_T_STR(name, type) TENSOR4D_RO_T_##type(name)

/** Read-Only (RO) tensor 4D.
 *
 * @param[in] name Tensor name. The tensor name is the prefix of the tensor components
 * @param[in] type Tensor type (BUFFER or IMAGE)
 */
#define TENSOR4D_RO_T(name, type) TENSOR4D_RO_T_STR(name, type)

#define TENSOR4D_WO_T_IMAGE(name)          \
    __write_only image2d_t name##_img, \
    TENSOR4D_T_BUFFER(name)

#define TENSOR4D_WO_T_BUFFER(name) TENSOR4D_T_BUFFER(name)

#define TENSOR4D_WO_T_STR(name, type) TENSOR4D_WO_T_##type(name)

/** Write-Only (WO) tensor 4D.
 *
 * @param[in] name Tensor name. The tensor name is the prefix of the tensor components
 * @param[in] type Tensor type (BUFFER or IMAGE)
 */
#define TENSOR4D_WO_T(name, type) TENSOR4D_WO_T_STR(name, type)

#define TENSOR3D_T_IMAGE(name)          \
    __read_only image2d_t name##_img, \
    __global uchar *name##_ptr,       \
    uint        name##_stride_y, \
    uint        name##_stride_z, \
    uint        name##_w,   \
    uint        name##_h,   \
    uint        name##_n,   \
    uint        name##_offset_first_element_in_bytes

#define TENSOR3D_T_BUFFER(name)    \
    __global uchar *name##_ptr,  \
    uint        name##_stride_y, \
    uint        name##_stride_z, \
    uint        name##_w,   \
    uint        name##_h,   \
    uint        name##_n,   \
    uint        name##_offset_first_element_in_bytes

#define TENSOR3D_T_STR(name, type) TENSOR3D_T_##type(name)
#define TENSOR3D_T(name, type) TENSOR3D_T_STR(name, type)

#if !defined(UNROLL_WITH_PRAGMA)
#define UNROLL_INCR(idx, step, macro) idx += (step); (macro)

#define LOOP_UNROLLING_1(idx, step, macro) (macro)
#define LOOP_UNROLLING_2(idx, step, macro) LOOP_UNROLLING_1(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_3(idx, step, macro) LOOP_UNROLLING_2(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_4(idx, step, macro) LOOP_UNROLLING_3(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_5(idx, step, macro) LOOP_UNROLLING_4(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_6(idx, step, macro) LOOP_UNROLLING_5(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_7(idx, step, macro) LOOP_UNROLLING_6(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_8(idx, step, macro) LOOP_UNROLLING_7(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_9(idx, step, macro) LOOP_UNROLLING_8(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_10(idx, step, macro) LOOP_UNROLLING_9(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_11(idx, step, macro) LOOP_UNROLLING_10(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_12(idx, step, macro) LOOP_UNROLLING_11(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_13(idx, step, macro) LOOP_UNROLLING_12(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_14(idx, step, macro) LOOP_UNROLLING_13(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_15(idx, step, macro) LOOP_UNROLLING_14(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_16(idx, step, macro) LOOP_UNROLLING_15(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_17(idx, step, macro) LOOP_UNROLLING_16(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_18(idx, step, macro) LOOP_UNROLLING_17(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_19(idx, step, macro) LOOP_UNROLLING_18(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_20(idx, step, macro) LOOP_UNROLLING_19(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_21(idx, step, macro) LOOP_UNROLLING_20(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_22(idx, step, macro) LOOP_UNROLLING_21(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_23(idx, step, macro) LOOP_UNROLLING_22(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_24(idx, step, macro) LOOP_UNROLLING_23(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_25(idx, step, macro) LOOP_UNROLLING_24(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_26(idx, step, macro) LOOP_UNROLLING_25(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_27(idx, step, macro) LOOP_UNROLLING_26(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_28(idx, step, macro) LOOP_UNROLLING_27(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_29(idx, step, macro) LOOP_UNROLLING_28(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_30(idx, step, macro) LOOP_UNROLLING_29(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_31(idx, step, macro) LOOP_UNROLLING_30(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_32(idx, step, macro) LOOP_UNROLLING_31(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_33(idx, step, macro) LOOP_UNROLLING_32(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_34(idx, step, macro) LOOP_UNROLLING_33(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_35(idx, step, macro) LOOP_UNROLLING_34(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_36(idx, step, macro) LOOP_UNROLLING_35(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_37(idx, step, macro) LOOP_UNROLLING_36(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_38(idx, step, macro) LOOP_UNROLLING_37(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_39(idx, step, macro) LOOP_UNROLLING_38(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_40(idx, step, macro) LOOP_UNROLLING_39(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_41(idx, step, macro) LOOP_UNROLLING_40(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_42(idx, step, macro) LOOP_UNROLLING_41(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_43(idx, step, macro) LOOP_UNROLLING_42(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_44(idx, step, macro) LOOP_UNROLLING_43(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_45(idx, step, macro) LOOP_UNROLLING_44(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_46(idx, step, macro) LOOP_UNROLLING_45(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_47(idx, step, macro) LOOP_UNROLLING_46(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_48(idx, step, macro) LOOP_UNROLLING_47(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_49(idx, step, macro) LOOP_UNROLLING_48(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_50(idx, step, macro) LOOP_UNROLLING_49(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_51(idx, step, macro) LOOP_UNROLLING_50(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_52(idx, step, macro) LOOP_UNROLLING_51(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_53(idx, step, macro) LOOP_UNROLLING_52(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_54(idx, step, macro) LOOP_UNROLLING_53(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_55(idx, step, macro) LOOP_UNROLLING_54(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_56(idx, step, macro) LOOP_UNROLLING_55(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_57(idx, step, macro) LOOP_UNROLLING_56(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_58(idx, step, macro) LOOP_UNROLLING_57(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_59(idx, step, macro) LOOP_UNROLLING_58(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_60(idx, step, macro) LOOP_UNROLLING_59(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_61(idx, step, macro) LOOP_UNROLLING_60(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_62(idx, step, macro) LOOP_UNROLLING_61(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_63(idx, step, macro) LOOP_UNROLLING_62(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_64(idx, step, macro) LOOP_UNROLLING_63(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_65(idx, step, macro) LOOP_UNROLLING_64(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_66(idx, step, macro) LOOP_UNROLLING_65(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_67(idx, step, macro) LOOP_UNROLLING_66(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_68(idx, step, macro) LOOP_UNROLLING_67(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_69(idx, step, macro) LOOP_UNROLLING_68(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_70(idx, step, macro) LOOP_UNROLLING_69(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_71(idx, step, macro) LOOP_UNROLLING_70(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_72(idx, step, macro) LOOP_UNROLLING_71(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_73(idx, step, macro) LOOP_UNROLLING_72(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_74(idx, step, macro) LOOP_UNROLLING_73(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_75(idx, step, macro) LOOP_UNROLLING_74(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_76(idx, step, macro) LOOP_UNROLLING_75(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_77(idx, step, macro) LOOP_UNROLLING_76(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_78(idx, step, macro) LOOP_UNROLLING_77(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_79(idx, step, macro) LOOP_UNROLLING_78(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_80(idx, step, macro) LOOP_UNROLLING_79(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_81(idx, step, macro) LOOP_UNROLLING_80(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_82(idx, step, macro) LOOP_UNROLLING_81(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_83(idx, step, macro) LOOP_UNROLLING_82(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_84(idx, step, macro) LOOP_UNROLLING_83(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_85(idx, step, macro) LOOP_UNROLLING_84(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_86(idx, step, macro) LOOP_UNROLLING_85(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_87(idx, step, macro) LOOP_UNROLLING_86(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_88(idx, step, macro) LOOP_UNROLLING_87(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_89(idx, step, macro) LOOP_UNROLLING_88(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_90(idx, step, macro) LOOP_UNROLLING_89(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_91(idx, step, macro) LOOP_UNROLLING_90(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_92(idx, step, macro) LOOP_UNROLLING_91(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_93(idx, step, macro) LOOP_UNROLLING_92(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_94(idx, step, macro) LOOP_UNROLLING_93(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_95(idx, step, macro) LOOP_UNROLLING_94(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_96(idx, step, macro) LOOP_UNROLLING_95(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_97(idx, step, macro) LOOP_UNROLLING_96(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_98(idx, step, macro) LOOP_UNROLLING_97(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_99(idx, step, macro) LOOP_UNROLLING_98(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_100(idx, step, macro) LOOP_UNROLLING_99(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_101(idx, step, macro) LOOP_UNROLLING_100(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_102(idx, step, macro) LOOP_UNROLLING_101(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_103(idx, step, macro) LOOP_UNROLLING_102(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_104(idx, step, macro) LOOP_UNROLLING_103(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_105(idx, step, macro) LOOP_UNROLLING_104(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_106(idx, step, macro) LOOP_UNROLLING_105(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_107(idx, step, macro) LOOP_UNROLLING_106(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_108(idx, step, macro) LOOP_UNROLLING_107(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_109(idx, step, macro) LOOP_UNROLLING_108(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_110(idx, step, macro) LOOP_UNROLLING_109(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_111(idx, step, macro) LOOP_UNROLLING_110(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_112(idx, step, macro) LOOP_UNROLLING_111(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_113(idx, step, macro) LOOP_UNROLLING_112(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_114(idx, step, macro) LOOP_UNROLLING_113(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_115(idx, step, macro) LOOP_UNROLLING_114(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_116(idx, step, macro) LOOP_UNROLLING_115(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_117(idx, step, macro) LOOP_UNROLLING_116(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_118(idx, step, macro) LOOP_UNROLLING_117(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_119(idx, step, macro) LOOP_UNROLLING_118(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_120(idx, step, macro) LOOP_UNROLLING_119(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_121(idx, step, macro) LOOP_UNROLLING_120(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_122(idx, step, macro) LOOP_UNROLLING_121(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_123(idx, step, macro) LOOP_UNROLLING_122(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_124(idx, step, macro) LOOP_UNROLLING_123(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_125(idx, step, macro) LOOP_UNROLLING_124(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_126(idx, step, macro) LOOP_UNROLLING_125(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_127(idx, step, macro) LOOP_UNROLLING_126(idx, step, macro); UNROLL_INCR(idx, step, macro)
#define LOOP_UNROLLING_128(idx, step, macro) LOOP_UNROLLING_127(idx, step, macro); UNROLL_INCR(idx, step, macro)

#define LOOP_UNROLLING_STR(type, idx, start, step, num, macro) \
    {                                                          \
        type idx = start;                                      \
        LOOP_UNROLLING_##num(idx, step, macro);                \
    }
#else // !defined(UNROLL_WITH_PRAGMA)
#define LOOP_UNROLLING_STR(type, idx, start, step, num, macro) \
    {                                                          \
        _Pragma("unroll")                                      \
        for(type idx = start; idx < (num * step); idx += step) \
        {                                                      \
            (macro);                                           \
        }                                                      \
    }
#endif // !defined(UNROLL_WITH_PRAGMA)
#define LOOP_UNROLLING(type, idx, start, step, num, macro) LOOP_UNROLLING_STR(type, idx, start, step, num, macro)

/** Get the get_global_id with partial N0. This function is useful when the dimension is not multiple of N0 and we need to use a partial N0
 *  to avoid out-of-bound read/write
 *
 * @note PARTIAL_N0 is used for get_global_id(n) = 0.
 *
 * @param[in] IDX        get_global_id index (0,1 and 2 only)
 * @param[in] N0         Number of elements read/written on the IDX direction
 * @param[in] PARTIAL_N0 Number of elements read/written on the IDX direction for get_global_id(IDX) = 0. If zero,
 *                        the Number of elements read/written on the IDX direction for get_global_id(IDX) = 0 is N0
 */
#define GET_SPATIAL_IDX(IDX, N0, PARTIAL_N0) (max((int)(get_global_id(IDX) * N0 - (N0 - PARTIAL_N0) % N0), 0))

/** Dot product integet 8bit function
 *
 *  @note Performs: c += dot(a, b)
 *
 * @param[in] A_DATA_TYPE A (lhs) data type
 * @param[in] B_DATA_TYPE B (rhs) data type
 * @param[in] C_DATA_TYPE C (accumulator) data type
 * @param[in] K0          Number of accumulations
 * @param[in] a           OpenCL vector a
 * @param[in] b           OpenCL vector b
 * @param[in] c           Scalar variable c
 */
#define DOT_PRODUCT_INTEGER8(A_DATA_TYPE, B_DATA_TYPE, C_DATA_TYPE, K0, a, b, c) DOT_PRODUCT_INTEGER8_STR(A_DATA_TYPE, B_DATA_TYPE, C_DATA_TYPE, K0, a, b, c)
#define DOT_PRODUCT_INTEGER8_STR(A_DATA_TYPE, B_DATA_TYPE, C_DATA_TYPE, K0, a, b, c) DOT_PRODUCT##K0##_INTEGER8(A_DATA_TYPE, B_DATA_TYPE, C_DATA_TYPE, a, b, c)
#define DOT_PRODUCT1_INTEGER8(A_DATA_TYPE, B_DATA_TYPE, C_DATA_TYPE, a, b, c) \
    ({                                                \
        c += (C_DATA_TYPE)(a) * (C_DATA_TYPE)(b);     \
    })
#if defined(ARM_COMPUTE_OPENCL_DOT8_ENABLED) && defined(cl_khr_integer_dot_product)
#define DOT_PRODUCT2_INTEGER8(A_DATA_TYPE, B_DATA_TYPE, C_DATA_TYPE, a, b, c) c += dot((A_DATA_TYPE##4)((a).s01, (A_DATA_TYPE##2)(0)), (B_DATA_TYPE##4)(((b).s01), (B_DATA_TYPE##2)(0)));
#define DOT_PRODUCT3_INTEGER8(A_DATA_TYPE, B_DATA_TYPE, C_DATA_TYPE, a, b, c) c += dot((A_DATA_TYPE##4)((a).s012, (A_DATA_TYPE)0), (B_DATA_TYPE##4)(((b).s012), (B_DATA_TYPE)0));
#define DOT_PRODUCT4_INTEGER8(A_DATA_TYPE, B_DATA_TYPE, C_DATA_TYPE, a, b, c) c += dot((a), (b));
#elif defined(ARM_COMPUTE_OPENCL_DOT8_ACC_ENABLED) && defined(cl_arm_integer_dot_product_accumulate_int8) //  defined(ARM_COMPUTE_OPENCL_DOT8_ENABLED) && defined(cl_khr_integer_dot_product)
#define DOT_PRODUCT2_INTEGER8(A_DATA_TYPE, B_DATA_TYPE, C_DATA_TYPE, a, b, c) c = arm_dot_acc((A_DATA_TYPE##4)((a).s01, (A_DATA_TYPE##2)(0)), (B_DATA_TYPE##4)(((b).s01), (B_DATA_TYPE##2)(0)), (c));
#define DOT_PRODUCT3_INTEGER8(A_DATA_TYPE, B_DATA_TYPE, C_DATA_TYPE, a, b, c) c = arm_dot_acc((A_DATA_TYPE##4)((a).s012, (A_DATA_TYPE)0), (B_DATA_TYPE##4)(((b).s012), (B_DATA_TYPE)0), (c));
#define DOT_PRODUCT4_INTEGER8(A_DATA_TYPE, B_DATA_TYPE, C_DATA_TYPE, a, b, c) c = arm_dot_acc((a), (b), (c));
#elif defined(ARM_COMPUTE_OPENCL_DOT8_ENABLED) && defined(cl_arm_integer_dot_product_int8) // defined(ARM_COMPUTE_OPENCL_DOT8_ENABLED) && defined(cl_arm_integer_dot_product_int8)
#define DOT_PRODUCT2_INTEGER8(A_DATA_TYPE, B_DATA_TYPE, C_DATA_TYPE, a, b, c) c += arm_dot((A_DATA_TYPE##4)((a).s01, (A_DATA_TYPE##2)(0)), (B_DATA_TYPE##4)(((b).s01), (B_DATA_TYPE##2)(0)));
#define DOT_PRODUCT3_INTEGER8(A_DATA_TYPE, B_DATA_TYPE, C_DATA_TYPE, a, b, c) c += arm_dot((A_DATA_TYPE##4)((a).s012, (A_DATA_TYPE)0), (B_DATA_TYPE##4)(((b).s012), (B_DATA_TYPE)0));
#define DOT_PRODUCT4_INTEGER8(A_DATA_TYPE, B_DATA_TYPE, C_DATA_TYPE, a, b, c) c += arm_dot((a), (b));
#else // defined(ARM_COMPUTE_OPENCL_DOT8_ACC_ENABLED) && defined(cl_arm_integer_dot_product_accumulate_int8)
#define DOT_PRODUCT2_INTEGER8(A_DATA_TYPE, B_DATA_TYPE, C_DATA_TYPE, a, b, c)   \
    ({                                                  \
        c += (C_DATA_TYPE)(a).s0 * (C_DATA_TYPE)(b).s0; \
        c += (C_DATA_TYPE)(a).s1 * (C_DATA_TYPE)(b).s1; \
    })
#define DOT_PRODUCT3_INTEGER8(A_DATA_TYPE, B_DATA_TYPE, C_DATA_TYPE, a, b, c)   \
    ({                                                  \
        DOT_PRODUCT2_INTEGER8(A_DATA_TYPE, B_DATA_TYPE, C_DATA_TYPE, a, b, c);  \
        c += (C_DATA_TYPE)(a).s2 * (C_DATA_TYPE)(b).s2; \
    })
#define DOT_PRODUCT4_INTEGER8(A_DATA_TYPE, B_DATA_TYPE, C_DATA_TYPE, x, y, val)   \
    ({                                                    \
        val += (C_DATA_TYPE)(x).s0 * (C_DATA_TYPE)(y).s0; \
        val += (C_DATA_TYPE)(x).s1 * (C_DATA_TYPE)(y).s1; \
        val += (C_DATA_TYPE)(x).s2 * (C_DATA_TYPE)(y).s2; \
        val += (C_DATA_TYPE)(x).s3 * (C_DATA_TYPE)(y).s3; \
    })
#endif // defined(ARM_COMPUTE_OPENCL_DOT8_ACC_ENABLED) && defined(cl_arm_integer_dot_product_accumulate_int8)
#define DOT_PRODUCT5_INTEGER8(A_DATA_TYPE, B_DATA_TYPE, C_DATA_TYPE, a, b, c) \
    ({                                                \
        DOT_PRODUCT4_INTEGER8(A_DATA_TYPE, B_DATA_TYPE, C_DATA_TYPE, ((a).s0123), ((b).s0123), c);     \
        DOT_PRODUCT1_INTEGER8(A_DATA_TYPE, B_DATA_TYPE, C_DATA_TYPE, ((a).s4), ((b).s4), c);     \
    })
#define DOT_PRODUCT6_INTEGER8(A_DATA_TYPE, B_DATA_TYPE, C_DATA_TYPE, a, b, c) \
    ({                                                \
        DOT_PRODUCT4_INTEGER8(A_DATA_TYPE, B_DATA_TYPE, C_DATA_TYPE, ((a).s0123), ((b).s0123), c);     \
        DOT_PRODUCT2_INTEGER8(A_DATA_TYPE, B_DATA_TYPE, C_DATA_TYPE, ((a).s45), ((b).s45), c);     \
    })
#define DOT_PRODUCT7_INTEGER8(A_DATA_TYPE, B_DATA_TYPE, C_DATA_TYPE, a, b, c) \
    ({                                                \
        DOT_PRODUCT4_INTEGER8(A_DATA_TYPE, B_DATA_TYPE, C_DATA_TYPE, ((a).s0123), ((b).s0123), c);     \
        DOT_PRODUCT3_INTEGER8(A_DATA_TYPE, B_DATA_TYPE, C_DATA_TYPE, ((a).s456), ((b).s456), c);     \
    })
#define DOT_PRODUCT8_INTEGER8(A_DATA_TYPE, B_DATA_TYPE, C_DATA_TYPE, a, b, c) \
    ({                                                \
        DOT_PRODUCT4_INTEGER8(A_DATA_TYPE, B_DATA_TYPE, C_DATA_TYPE, ((a).lo), ((b).lo), c);     \
        DOT_PRODUCT4_INTEGER8(A_DATA_TYPE, B_DATA_TYPE, C_DATA_TYPE, ((a).hi), ((b).hi), c);     \
    })
#define DOT_PRODUCT9_INTEGER8(A_DATA_TYPE, B_DATA_TYPE, C_DATA_TYPE, a, b, c) \
    ({                                                \
        DOT_PRODUCT8_INTEGER8(A_DATA_TYPE, B_DATA_TYPE, C_DATA_TYPE, ((a).s01234567), ((b).s01234567), c);     \
        DOT_PRODUCT1_INTEGER8(A_DATA_TYPE, B_DATA_TYPE, C_DATA_TYPE, ((a).s8), ((b).s8), c);     \
    })
#define DOT_PRODUCT10_INTEGER8(A_DATA_TYPE, B_DATA_TYPE, C_DATA_TYPE, a, b, c) \
    ({                                                \
        DOT_PRODUCT8_INTEGER8(A_DATA_TYPE, B_DATA_TYPE, C_DATA_TYPE, ((a).s01234567), ((b).s01234567), c);     \
        DOT_PRODUCT2_INTEGER8(A_DATA_TYPE, B_DATA_TYPE, C_DATA_TYPE, ((a).s89), ((b).s89), c);     \
    })
#define DOT_PRODUCT11_INTEGER8(A_DATA_TYPE, B_DATA_TYPE, C_DATA_TYPE, a, b, c) \
    ({                                                \
        DOT_PRODUCT8_INTEGER8(A_DATA_TYPE, B_DATA_TYPE, C_DATA_TYPE, ((a).s01234567), ((b).s01234567), c);     \
        DOT_PRODUCT3_INTEGER8(A_DATA_TYPE, B_DATA_TYPE, C_DATA_TYPE, ((a).s89A), ((b).s89A), c);     \
    })
#define DOT_PRODUCT12_INTEGER8(A_DATA_TYPE, B_DATA_TYPE, C_DATA_TYPE, a, b, c) \
    ({                                                \
        DOT_PRODUCT8_INTEGER8(A_DATA_TYPE, B_DATA_TYPE, C_DATA_TYPE, ((a).s01234567), ((b).s01234567), c);     \
        DOT_PRODUCT4_INTEGER8(A_DATA_TYPE, B_DATA_TYPE, C_DATA_TYPE, ((a).s89AB), ((b).s89AB), c);     \
    })
#define DOT_PRODUCT13_INTEGER8(A_DATA_TYPE, B_DATA_TYPE, C_DATA_TYPE, a, b, c) \
    ({                                                \
        DOT_PRODUCT8_INTEGER8(A_DATA_TYPE, B_DATA_TYPE, C_DATA_TYPE, ((a).s01234567), ((b).s01234567), c);     \
        DOT_PRODUCT4_INTEGER8(A_DATA_TYPE, B_DATA_TYPE, C_DATA_TYPE, ((a).s89AB), ((b).s89AB), c);     \
        DOT_PRODUCT1_INTEGER8(A_DATA_TYPE, B_DATA_TYPE, C_DATA_TYPE, ((a).sC), ((b).sC), c);     \
    })
#define DOT_PRODUCT14_INTEGER8(A_DATA_TYPE, B_DATA_TYPE, C_DATA_TYPE, a, b, c) \
    ({                                                \
        DOT_PRODUCT8_INTEGER8(A_DATA_TYPE, B_DATA_TYPE, C_DATA_TYPE, ((a).s01234567), ((b).s01234567), c);     \
        DOT_PRODUCT4_INTEGER8(A_DATA_TYPE, B_DATA_TYPE, C_DATA_TYPE, ((a).s89AB), ((b).s89AB), c);     \
        DOT_PRODUCT2_INTEGER8(A_DATA_TYPE, B_DATA_TYPE, C_DATA_TYPE, ((a).sCD), ((b).sCD), c);     \
    })
#define DOT_PRODUCT15_INTEGER8(A_DATA_TYPE, B_DATA_TYPE, C_DATA_TYPE, a, b, c) \
    ({                                                \
        DOT_PRODUCT8_INTEGER8(A_DATA_TYPE, B_DATA_TYPE, C_DATA_TYPE, ((a).s01234567), ((b).s01234567), c);     \
        DOT_PRODUCT4_INTEGER8(A_DATA_TYPE, B_DATA_TYPE, C_DATA_TYPE, ((a).s89AB), ((b).s89AB), c);     \
        DOT_PRODUCT3_INTEGER8(A_DATA_TYPE, B_DATA_TYPE, C_DATA_TYPE, ((a).sCDE), ((b).sCDE), c);     \
    })
#define DOT_PRODUCT16_INTEGER8(A_DATA_TYPE, B_DATA_TYPE, C_DATA_TYPE, a, b, c) \
    ({                                                 \
        DOT_PRODUCT8_INTEGER8(A_DATA_TYPE, B_DATA_TYPE, C_DATA_TYPE, ((a).lo), ((b).lo), c);      \
        DOT_PRODUCT8_INTEGER8(A_DATA_TYPE, B_DATA_TYPE, C_DATA_TYPE, ((a).hi), ((b).hi), c);      \
    })

/** Dot product integet 8bit function
 *
 *  @note Performs: c += dot(a, b)
 *
 * @param[in] A_DATA_TYPE A (lhs) data type
 * @param[in] B_DATA_TYPE B (rhs) data type
 * @param[in] C_DATA_TYPE C (accumulator) data type
 * @param[in] K0          Number of accumulations
 * @param[in] a           OpenCL vector a
 * @param[in] c           Scalar variable c
 */
#define REDUCE_INTEGER8(A_DATA_TYPE, B_DATA_TYPE, C_DATA_TYPE, K0, a, c) REDUCE_INTEGER8_STR(A_DATA_TYPE, B_DATA_TYPE, C_DATA_TYPE, K0, a, c)
#define REDUCE_INTEGER8_STR(A_DATA_TYPE, B_DATA_TYPE, C_DATA_TYPE, K0, a, c) DOT_PRODUCT_INTEGER8(A_DATA_TYPE, B_DATA_TYPE, C_DATA_TYPE, K0, a, (TILE_VECTOR_TYPE##K0(B_DATA_TYPE))1, c)

/** Load a vector from global memory (tensor)
 *
 * @param[in] DATA_TYPE   Data type
 * @param[in] WIDTH       Number of dst columns
 * @param[in] TENSOR_TYPE Type of cl_type used to store the tensor in global memory (BUFFER=cl_buffer, IMAGE=cl_image).
 *                        In case of cl_image, only WIDTH multiples of 4 are supported (4, 8, 16)
 * @param[in] TENSOR      Tensor basename
 * @param[in] X           Starting X position
 * @param[in] Y           Starting Y position
 * @param[in] STRIDE_Y    Stride Y (in bytes)
 */
#define V_LOAD(DATA_TYPE, WIDTH, TENSOR_TYPE, TENSOR, X, Y, STRIDE_Y) V_LOAD_STR(DATA_TYPE, WIDTH, TENSOR_TYPE, TENSOR, X, Y, STRIDE_Y)
#define V_LOAD_STR(DATA_TYPE, WIDTH, TENSOR_TYPE, TENSOR, X, Y, STRIDE_Y) V_LOAD_##TENSOR_TYPE(DATA_TYPE, WIDTH, TENSOR, X, Y, STRIDE_Y)
#define V_LOAD_BUFFER(DATA_TYPE, WIDTH, TENSOR, X, Y, STRIDE_Y) \
    VLOAD(WIDTH)                                                \
    (0, (__global DATA_TYPE *)(TENSOR##_ptr + TENSOR##_offset_first_element_in_bytes + (X) * sizeof(DATA_TYPE) + (Y) * (STRIDE_Y)))
#define V_LOAD_IMAGE(DATA_TYPE, WIDTH, TENSOR, X, Y, STRIDE_Y) READ_IMAGE2D(DATA_TYPE, CONVERT_VECTOR_SIZE_TO_PIXEL_UNIT(WIDTH), TENSOR##_img, (X) / 4, (Y))

/** Store a vector in global memory (tensor)
 *
 * @param[in] DATA_TYPE   Data type
 * @param[in] WIDTH       Number of dst columns
 * @param[in] TENSOR_TYPE Type of cl_type used to store the tensor in global memory (BUFFER=cl_buffer, IMAGE=cl_image).
 *                        In case of cl_image, only WIDTH multiples of 4 are supported (4, 8, 16)
 * @param[in] TENSOR      Tensor basename
 * @param[in] X           Starting X position
 * @param[in] Y           Starting Y position
 * @param[in] STRIDE_Y    Stride Y (in bytes)
 * @param[in] VALUES      Values to store in memory
 */
#define V_STORE(DATA_TYPE, WIDTH, TENSOR_TYPE, TENSOR, X, Y, STRIDE_Y, VALUES) V_STORE_STR(DATA_TYPE, WIDTH, TENSOR_TYPE, TENSOR, X, Y, STRIDE_Y, VALUES)
#define V_STORE_STR(DATA_TYPE, WIDTH, TENSOR_TYPE, TENSOR, X, Y, STRIDE_Y, VALUES) V_STORE_##TENSOR_TYPE(DATA_TYPE, WIDTH, TENSOR, X, Y, STRIDE_Y, VALUES)
#define V_STORE_BUFFER(DATA_TYPE, WIDTH, TENSOR, X, Y, STRIDE_Y, VALUES) \
    VSTORE(WIDTH)                                                \
    (VALUES, 0, (__global DATA_TYPE *)(TENSOR##_ptr + TENSOR##_offset_first_element_in_bytes + (X) * sizeof(DATA_TYPE) + (Y) * (STRIDE_Y)))
#define V_STORE_IMAGE(DATA_TYPE, WIDTH, TENSOR, X, Y, STRIDE_Y, VALUES) WRITE_IMAGE2D(DATA_TYPE, CONVERT_VECTOR_SIZE_TO_PIXEL_UNIT(WIDTH), TENSOR##_img, (X) / 4, (Y), VALUES)

/** Load a tile from global memory (tensor)
 *
 * @param[in]  DATA_TYPE     Data type
 * @param[in]  HEIGHT        Number of dst rows
 * @param[in]  WIDTH         Number of dst columns
 * @param[in]  TENSOR_TYPE   Type of cl_type used to store the tensor in global memory (BUFFER=cl_buffer, IMAGE=cl_image).
 *                           In case of cl_image, only WIDTH multiples of 4 are supported (4, 8, 16)
 * @param[in]  TENSOR        Tensor basename
 * @param[in]  X             Starting X position
 * @param[in]  Y             Starting Y position
 * @param[in]  YI_MULTIPLIER Parameter used to multiply the internal row increment (_i).
 *                           In common cases should be 1 but it becomes useful when we want to load rows which are multiple of STRIDE_Y. (e.g. loading the weights of convolution layer).
 *                           In this case the address calculation is performed as: (Y + _i * Y_MULTIPLIER) * STRIDE_Y
 * @param[in]  STRIDE_Y      Stride Y (in bytes) used to load each row.
 * @param[out] dst           Output tile
 */
#define T_LOAD(DATA_TYPE, HEIGHT, WIDTH, TENSOR_TYPE, TENSOR, X, Y, YI_MULTIPLIER, STRIDE_Y, dst)                      \
    ({                                                                                                                 \
        LOOP_UNROLLING(int, _i, 0, 1, HEIGHT,                                                                          \
        {                                                                                                              \
            dst[_i].v = V_LOAD(DATA_TYPE, WIDTH, TENSOR_TYPE, TENSOR, X, ((Y) + _i * (int)(YI_MULTIPLIER)), STRIDE_Y); \
        })                                                                                                             \
    })

/** Store a VECTOR variable (e.g. int4, int8, char2 etc.) to a specified column in the TILE object
 *
 * @param[in]      VECTOR Vector variable to store
 * @param[in, out] TILE   Tile variable to store to
 * @param[in]      WIDTH  Width of the vector variable, also height of the tile (e.g. 2 if char2)
 * @param[in]      COLUMN Column index of the tile
 */
#define COPY_VECTOR_TO_TILE_COLUMN(VECTOR, TILE, WIDTH, COLUMN) COPY_VECTOR_TO_TILE_COLUMN_STR(VECTOR, TILE, WIDTH, COLUMN)
#define COPY_VECTOR_TO_TILE_COLUMN_STR(VECTOR, TILE, WIDTH, COLUMN) COPY_##WIDTH##_VECTOR_TO_TILE_COLUMN(VECTOR, TILE, COLUMN)
#define COPY_1_VECTOR_TO_TILE_COLUMN(VECTOR, TILE, COLUMN) \
    ({                                                      \
        TILE[0].s[COLUMN] = VECTOR;                         \
    })

#define COPY_2_VECTOR_TO_TILE_COLUMN(VECTOR, TILE, COLUMN) \
    ({                                                      \
        TILE[0].s[COLUMN] = VECTOR.s0;                      \
        TILE[1].s[COLUMN] = VECTOR.s1;                      \
    })

#define COPY_3_VECTOR_TO_TILE_COLUMN(VECTOR, TILE, COLUMN) \
    ({                                                      \
        TILE[0].s[COLUMN] = VECTOR.s0;                      \
        TILE[1].s[COLUMN] = VECTOR.s1;                      \
        TILE[2].s[COLUMN] = VECTOR.s2;                      \
    })

#define COPY_4_VECTOR_TO_TILE_COLUMN(VECTOR, TILE, COLUMN) \
    ({                                                      \
        TILE[0].s[COLUMN] = VECTOR.s0;                      \
        TILE[1].s[COLUMN] = VECTOR.s1;                      \
        TILE[2].s[COLUMN] = VECTOR.s2;                      \
        TILE[3].s[COLUMN] = VECTOR.s3;                      \
    })

#define COPY_8_VECTOR_TO_TILE_COLUMN(VECTOR, TILE, COLUMN) \
    ({                                                      \
        TILE[0].s[COLUMN] = VECTOR.s0;                      \
        TILE[1].s[COLUMN] = VECTOR.s1;                      \
        TILE[2].s[COLUMN] = VECTOR.s2;                      \
        TILE[3].s[COLUMN] = VECTOR.s3;                      \
        TILE[4].s[COLUMN] = VECTOR.s4;                      \
        TILE[5].s[COLUMN] = VECTOR.s5;                      \
        TILE[6].s[COLUMN] = VECTOR.s6;                      \
        TILE[7].s[COLUMN] = VECTOR.s7;                      \
    })

#define COPY_16_VECTOR_TO_TILE_COLUMN(VECTOR, TILE, COLUMN) \
    ({                                                      \
        TILE[0].s[COLUMN] = VECTOR.s0;                      \
        TILE[1].s[COLUMN] = VECTOR.s1;                      \
        TILE[2].s[COLUMN] = VECTOR.s2;                      \
        TILE[3].s[COLUMN] = VECTOR.s3;                      \
        TILE[4].s[COLUMN] = VECTOR.s4;                      \
        TILE[5].s[COLUMN] = VECTOR.s5;                      \
        TILE[6].s[COLUMN] = VECTOR.s6;                      \
        TILE[7].s[COLUMN] = VECTOR.s7;                      \
        TILE[8].s[COLUMN] = VECTOR.s8;                      \
        TILE[9].s[COLUMN] = VECTOR.s9;                      \
        TILE[10].s[COLUMN] = VECTOR.sA;                     \
        TILE[11].s[COLUMN] = VECTOR.sB;                     \
        TILE[12].s[COLUMN] = VECTOR.sC;                     \
        TILE[13].s[COLUMN] = VECTOR.sD;                     \
        TILE[14].s[COLUMN] = VECTOR.sE;                     \
        TILE[15].s[COLUMN] = VECTOR.sF;                     \
    })

/** Load SRC_HEIGHT x SRC_WIDTH elements from global memory (tensor), and store them in a SRC_WIDTH x SRC_HEIGHT tile
 *
 * @param[in]  DATA_TYPE     Data type
 * @param[in]  SRC_HEIGHT    Number of source rows, or number of columns of the output tile
 * @param[in]  SRC_WIDTH     Number of source columns, or number of tile rows
 * @param[in]  TENSOR_TYPE   Type of cl_type used to store the tensor in global memory (BUFFER=cl_buffer, IMAGE=cl_image).
 *                           In case of cl_image, only WIDTH multiples of 4 are supported (4, 8, 16)
 * @param[in]  TENSOR        Tensor basename
 * @param[in]  X             Starting X position
 * @param[in]  Y             Starting Y position
 * @param[in]  YI_MULTIPLIER Parameter used to multiply the internal row increment (_i).
 *                           In common cases should be 1 but it becomes useful when we want to load rows which are multiple of STRIDE_Y.
 *                           (e.g. loading the weights of convolution layer).
 *                           In this case the address calculation is performed as: (Y + _i * Y_MULTIPLIER) * STRIDE_Y
 * @param[in]  STRIDE_Y      Stride Y (in bytes) used to load each row.
 * @param[out] dst           Output tile
 */
#define T_LOAD_TRANSPOSED(DATA_TYPE, SRC_HEIGHT, SRC_WIDTH, TENSOR_TYPE, TENSOR, X, Y, YI_MULTIPLIER, STRIDE_Y, dst)     \
    ({                                                                                                                   \
        LOOP_UNROLLING(int, _i, 0, 1, SRC_HEIGHT,                                                                        \
        {                                                                                                                \
            VEC_DATA_TYPE(DATA_TYPE, SRC_WIDTH)                                                                          \
                tmp = V_LOAD(DATA_TYPE, SRC_WIDTH, TENSOR_TYPE, TENSOR, X, ((Y) + _i * (int)(YI_MULTIPLIER)), STRIDE_Y); \
            COPY_VECTOR_TO_TILE_COLUMN(tmp, dst, SRC_WIDTH, _i);                                                         \
        })                                                                                                               \
    })

/** Load a tile from global memory (tensor) using an indirect Y index tile
 *
 * @param[in]  DATA_TYPE   Data type
 * @param[in]  HEIGHT      Number of dst rows
 * @param[in]  WIDTH       Number of dst columns
 * @param[in]  TENSOR_TYPE Type of cl_type used to store the tensor in global memory (BUFFER=cl_buffer, IMAGE=cl_image). Currently BUFFER only is supported
 *                         In case of cl_image, only WIDTH multiples of 4 are supported (4, 8, 16)
 * @param[in]  TENSOR      Tensor basename
 * @param[in]  X           Starting X position
 * @param[in]  STRIDE_Y    Stride Y (in bytes)
 * @param[in]  indirect_y  Indirect Y index tile
 * @param[out] dst         Output tile
 */
#define T_LOAD_INDIRECT(DATA_TYPE, HEIGHT, WIDTH, TENSOR_TYPE, TENSOR, X, STRIDE_Y, indirect_y, dst)    \
    ({                                                                                                  \
        LOOP_UNROLLING(int, _i, 0, 1, HEIGHT,                                                           \
        {                                                                                               \
            dst[_i].v = V_LOAD(DATA_TYPE, WIDTH, TENSOR_TYPE, TENSOR, X, (indirect_y[_i].v), STRIDE_Y); \
        })                                                                                              \
    })

/** Load a tile from global memory (tensor) using an indirect Y index tile and conditionally use a different length for the load
 *
 * @note If WIDTH1_CONDITION is true, the load will use the WIDTH1 length for the store
 * @note The vectors are stored in reverse order so the invalid rows are overwritten by the valid ones
 *
 * @param[in]  DATA_TYPE        Data type
 * @param[in]  HEIGHT           Number of dst rows
 * @param[in]  WIDTH0           Store width to use if WIDTH1_CONDITION = false
 * @param[in]  WIDTH1           Store width to use if WIDTH1_CONDITION = true
 * @param[in]  TENSOR_TYPE      Type of cl_type used to store the tensor in global memory (BUFFER=cl_buffer, IMAGE=cl_image).
 *                              In case of cl_image, only WIDTH multiples of 4 are supported (4, 8, 16)
 * @param[in]  TENSOR           Tensor basename
 * @param[in]  X                Starting X position
 * @param[in]  STRIDE_Y         Stride Y (in bytes) used to load each row.
 * @param[in]  WIDTH1_CONDITION Condition to select the WIDTH1 store
 * @param[out] dst              Output tile
 * @param[out] indirect_y       Indirect Y index tile
 */
#define T_LOAD_INDIRECT_WIDTH_SELECT(DATA_TYPE, HEIGHT, WIDTH0, WIDTH1, TENSOR_TYPE, TENSOR, X, STRIDE_Y, WIDTH1_CONDITION, dst, indirect_y)                                                      \
    ({                                                                                                                                                                                             \
        if(WIDTH1_CONDITION)                                                                                                                                                                       \
        {                                                                                                                                                                                          \
            LOOP_UNROLLING(int, _i, 0, 1, HEIGHT,                                                                                                                                                  \
            {                                                                                                                                                                                      \
                VLOAD_PARTIAL(WIDTH0, WIDTH1)                                                         \
                (dst[HEIGHT - 1 - _i].v, 0, (__global DATA_TYPE *)(TENSOR##_ptr + TENSOR##_offset_first_element_in_bytes + (X) * sizeof(DATA_TYPE) + (indirect_y[HEIGHT - 1 - _i].v) * STRIDE_Y));               \
            })                                                                                                                                                                                     \
        }                                                                                                                                                                                          \
        else                                                                                                                                                                                       \
        {                                                                                                                                                                                          \
            LOOP_UNROLLING(int, _i, 0, 1, HEIGHT,                                                                                                                                                  \
            {                                                                                                                                                                                      \
                dst[HEIGHT - 1 - _i].v = V_LOAD(DATA_TYPE, WIDTH0, TENSOR_TYPE, TENSOR, X, (indirect_y[HEIGHT - 1 - _i].v), STRIDE_Y); \
            })                                                                                                                                                                                     \
        }                                                                                                                                                                                          \
    })
/** Load a tile from global memory (tensor) when the tensor is stored using a NHWC layout
 *
 * @param[in]  DATA_TYPE     Data type
 * @param[in]  TILE_HEIGHT   Number of elements to load from Y (height) dimension
 * @param[in]  TILE_WIDTH    Number of elements to load from X (width) dimension
 * @param[in]  TILE_CHANNELS Number of elements to load from C (channel) dimension
 * @param[in]  TENSOR_TYPE   Type of cl_type used to store the tensor in global memory (BUFFER=cl_buffer, IMAGE=cl_image). Currently BUFFER only is supported
 *                           In case of cl_image, only TILE_CHANNELS multiples of 4 are supported (4, 8, 16)
 * @param[in]  TENSOR        Tensor basename
 * @param[in]  B             Starting batch index
 * @param[in]  Y             Starting Y index
 * @param[in]  X             Starting X index
 * @param[in]  C             Starting C index
 * @param[in]  TENSOR_HEIGHT Number of elements to load from Y (height) dimension
 * @param[in]  TENSOR_WIDTH  Number of elements to load from X (width) dimension
 * @param[in]  STRIDE_Y      Stride Y (in bytes)
 * @param[out] dst           Output tile
 */
#define T_LOAD_NHWC(DATA_TYPE, TILE_HEIGHT, TILE_WIDTH, TILE_CHANNELS, TENSOR_TYPE, TENSOR, B, Y, X, C, TENSOR_WIDTH, TENSOR_HEIGHT, STRIDE_Y, dst)   \
    ({                                                                                                                                                \
        LOOP_UNROLLING(int, _yk, 0, 1, TILE_HEIGHT,                                                                                                   \
        {                                                                                                                                             \
            LOOP_UNROLLING(int, _xk, 0, 1, TILE_WIDTH,                                                                                                \
            {                                                                                                                                         \
                int _src_y = (X) + _xk + ((Y) + _yk) * (TENSOR_WIDTH);                                                                                \
                _src_y    += (B) * (int)(TENSOR_WIDTH) * (int)(TENSOR_HEIGHT);                                                                        \
                int _src_valid_y = (((X) + _xk) >= 0 && ((X) + _xk) < (int)(TENSOR_WIDTH) && ((Y) + _yk) >= 0 && ((Y) + _yk) < (int)(TENSOR_HEIGHT)); \
                if(_src_valid_y != 0)                                                                                                                 \
                {                                                                                                                                     \
                    dst[_xk + _yk * (TILE_WIDTH)].v = V_LOAD(DATA_TYPE, TILE_CHANNELS, TENSOR_TYPE, TENSOR, C, _src_y, STRIDE_Y);                     \
                }                                                                                                                                     \
            })                                                                                                                                        \
        })                                                                                                                                            \
    })

/** Load a tile from global memory (tensor) when the tensor is stored using a NHWC layout with dilation for the X and Y increments
 *
 * @param[in]  DATA_TYPE      Data type
 * @param[in]  TILE_HEIGHT    Number of elements to load from Y (height) dimension
 * @param[in]  TILE_WIDTH     Number of elements to load from X (width) dimension
 * @param[in]  TILE_CHANNELS  Number of elements to load from C (channel) dimension
 * @param[in]  TENSOR_TYPE    Type of cl_type used to store the tensor in global memory (BUFFER=cl_buffer, IMAGE=cl_image). Currently BUFFER only is supported
 *                            In case of cl_image, only TILE_CHANNELS multiples of 4 are supported (4, 8, 16)
 * @param[in]  TENSOR         Tensor basename
 * @param[in]  B              Starting batch index
 * @param[in]  Y              Starting Y index
 * @param[in]  X              Starting X index
 * @param[in]  C              Starting C index
 * @param[in]  TENSOR_HEIGHT  Number of elements to load from Y (height) dimension
 * @param[in]  TENSOR_WIDTH   Number of elements to load from X (width) dimension
 * @param[in]  DILATION_X     Dilation for the X increment
 * @param[in]  DILATION_Y     Dilation for the Y increment
 * @param[in]  BOUNDARY_CHECK Boundary check flag. If true, it checks for any out-of-bound reads
 * @param[out] dst            Output tile
 */
#define T_LOAD_NHWC_WITH_DILATION(DATA_TYPE, TILE_HEIGHT, TILE_WIDTH, TILE_CHANNELS, TENSOR_TYPE, TENSOR, B, Y, X, C, TENSOR_WIDTH, TENSOR_HEIGHT, DILATION_X, DILATION_Y, BOUNDARY_CHECK, dst)         \
    ({ \
        LOOP_UNROLLING(int, _yk, 0, 1, TILE_HEIGHT, \
        { \
            LOOP_UNROLLING(int, _xk, 0, 1, TILE_WIDTH, \
            { \
                int _src_y = (X) + _xk * (DILATION_X); \
                int _src_z = ((Y) + _yk * (DILATION_Y)); \
                int _src_w    = (B); \
                bool _src_valid_y = (((X) + _xk * (DILATION_X)) >= 0) && (((X) + _xk * (DILATION_X)) < (int)(TENSOR_WIDTH)) && (((Y) + _yk * (DILATION_Y)) >= 0) && (((Y) + _yk * (DILATION_Y)) < (int)(TENSOR_HEIGHT)); \
                if(!(BOUNDARY_CHECK)) \
                { \
                    dst[_xk + _yk * (TILE_WIDTH)].v = VLOAD(TILE_CHANNELS)                                                \
                    (0, (__global DATA_TYPE *)(TENSOR##_ptr + TENSOR##_offset_first_element_in_bytes + (C) * sizeof(DATA_TYPE) + (_src_y) * (TENSOR##_stride_y) + (_src_z) * (TENSOR##_stride_z) + (_src_w) * (TENSOR##_stride_w))); \
                } \
                else \
                { \
                    if(_src_valid_y) \
                    { \
                        dst[_xk + _yk * (TILE_WIDTH)].v = VLOAD(TILE_CHANNELS)                                                \
                    (0, (__global DATA_TYPE *)(TENSOR##_ptr + TENSOR##_offset_first_element_in_bytes + (C) * sizeof(DATA_TYPE) + (_src_y) * (TENSOR##_stride_y) + (_src_z) * (TENSOR##_stride_z) + (_src_w) * (TENSOR##_stride_w))); \
                    }                                                                                                                                                                                                 \
                } \
            })                                                                                                                                                                                                             \
        })                                                                                                                                                                                                             \
    })

/** Load a tile from global memory (tensor) when the tensor is stored using a NHWC layout using indirect X and Y coordinates
 *
 * @param[in]  DATA_TYPE     Data type
 * @param[in]  TILE_AREA     Number of elements to load from Y (height) dimension * Number of elements to load from X (width) dimension
 * @param[in]  TILE_CHANNELS Number of elements to load from C (channel) dimension
 * @param[in]  TENSOR_TYPE   Type of cl_type used to store the tensor in global memory (BUFFER=cl_buffer, IMAGE=cl_image). Currently BUFFER only is supported
 *                           In case of cl_image, only TILE_CHANNELS multiples of 4 are supported (4, 8, 16)
 * @param[in]  TENSOR        Tensor basename
 * @param[in]  B             Starting batch index
 * @param[in]  Y             Starting Y index
 * @param[in]  X             Starting X index
 * @param[in]  C             Starting C index
 * @param[in]  TENSOR_WIDTH  Number of elements to load from X (width) dimension
 * @param[in]  TENSOR_HEIGHT Number of elements to load from Y (height) dimension
 * @param[in]  STRIDE_Y      Stride Y (in bytes)
 * @param[out] xi            A tile with (TILE_WIDTH x TILE_HEIGHT) values with the indirect X coordinate
 * @param[out] yi            A tile with (TILE_WIDTH x TILE_HEIGHT) values with the indirect Y coordinate
 * @param[out] dst           Output tile
 */
#define T_LOAD_NHWC_INDIRECT(DATA_TYPE, TILE_AREA, TILE_CHANNELS, TENSOR_TYPE, TENSOR, B, Y, X, C, TENSOR_WIDTH, TENSOR_HEIGHT, STRIDE_Y, xi, yi, dst)                \
    ({                                                                                                                                                                \
        LOOP_UNROLLING(int, _i, 0, 1, TILE_AREA,                                                                                                                      \
        {                                                                                                                                                             \
            int _src_y = (X) + xi[_i].v + ((Y) + yi[_i].v) * (TENSOR_WIDTH);                                                                                          \
            _src_y += (B) * (int)(TENSOR_WIDTH) * (int)(TENSOR_HEIGHT);                                                                                               \
            int _src_valid_y = (((X) + xi[_i].v) >= 0 && ((X) + xi[_i].v) < (int)(TENSOR_WIDTH) && ((Y) + yi[_i].v) >= 0 && ((Y) + yi[_i].v) < (int)(TENSOR_HEIGHT)); \
            if(_src_valid_y != 0)                                                                                                                                     \
            {                                                                                                                                                         \
                dst[_i].v = V_LOAD(DATA_TYPE, TILE_CHANNELS, TENSOR_TYPE, TENSOR, C, _src_y, STRIDE_Y);                                                               \
            }                                                                                                                                                         \
        })                                                                                                                                                            \
    })

/** Load a tile from global memory (tensor) using an indirect buffer for the Y coordinates
 *
 * @param[in]  DATA_TYPE     Data type
 * @param[in]  TILE_AREA     Number of elements to load from Y (height) dimension * Number of elements to load from X (width) dimension
 * @param[in]  TILE_CHANNELS Number of elements to load from C (channel) dimension
 * @param[in]  TENSOR_TYPE   Type of cl_type used to store the tensor in global memory (BUFFER=cl_buffer, IMAGE=cl_image).
 *                           When TENSOR_TYPE=IMAGE, the if condition for the out-of-bound check can be skipped
 *                           In case of cl_image, only TILE_CHANNELS multiples of 4 are supported (4, 8, 16)
 * @param[in]  TENSOR        Tensor basename
 * @param[in]  C             Starting C index
 * @param[in]  STRIDE_Y      Stride Y (in bytes)
 * @param[out] yi            A tile with (TILE_WIDTH x TILE_HEIGHT) values with the indirect Y coordinate
 *                           16 is the maximum indirect buffer size.
 * @param[out] dst           Output tile
 */
#define T_LOAD2D_INDIRECT(DATA_TYPE, TILE_AREA, TILE_CHANNELS, TENSOR_TYPE, TENSOR, C, STRIDE_Y, yi, dst) T_LOAD2D_INDIRECT_STR(DATA_TYPE, TILE_AREA, TILE_CHANNELS, TENSOR_TYPE, TENSOR, C, STRIDE_Y, yi, dst)
#define T_LOAD2D_INDIRECT_STR(DATA_TYPE, TILE_AREA, TILE_CHANNELS, TENSOR_TYPE, TENSOR, C, STRIDE_Y, yi, dst) T_LOAD2D_INDIRECT_##TENSOR_TYPE(DATA_TYPE, TILE_AREA, TILE_CHANNELS, TENSOR_TYPE, TENSOR, C, STRIDE_Y, yi, dst)
#define T_LOAD2D_INDIRECT_BUFFER(DATA_TYPE, TILE_AREA, TILE_CHANNELS, TENSOR_TYPE, TENSOR, C, STRIDE_Y, yi, dst) \
    ({ \
        LOOP_UNROLLING(int, _i, 0, 1, TILE_AREA, \
        { \
            if(yi[0].s[_i] >= 0) \
            { \
                dst[_i].v = V_LOAD(DATA_TYPE, TILE_CHANNELS, TENSOR_TYPE, TENSOR, C, yi[0].s[_i], STRIDE_Y); \
            } \
        }) \
    })

#define T_LOAD2D_INDIRECT_IMAGE(DATA_TYPE, TILE_AREA, TILE_CHANNELS, TENSOR_TYPE, TENSOR, C, STRIDE_Y, yi, dst) \
    ({ \
        LOOP_UNROLLING(int, _i, 0, 1, TILE_AREA, \
        { \
            dst[_i].v = V_LOAD(DATA_TYPE, TILE_CHANNELS, TENSOR_TYPE, TENSOR, C, yi[0].s[_i], STRIDE_Y); \
        }) \
    })

/** Load a tile from global memory (tensor) when the tensor is stored using a NDHWC layout using indirect X, Y and Z coordinates
 *
 * @param[in]  DATA_TYPE     Data type
 * @param[in]  TILE_AREA     Number of elements to load from Y (height) dimension * Number of elements to load from X (width) dimension
 * @param[in]  TILE_CHANNELS Number of elements to load from C (channel) dimension
 * @param[in]  TENSOR_TYPE   Type of cl_type used to store the tensor in global memory (BUFFER=cl_buffer, IMAGE=cl_image). Currently BUFFER only is supported
 *                           In case of cl_image, only TILE_CHANNELS multiples of 4 are supported (4, 8, 16)
 * @param[in]  TENSOR        Tensor basename
 * @param[in]  B             Starting batch index
 * @param[in]  Z             Starting Z index
 * @param[in]  Y             Starting Y index
 * @param[in]  X             Starting X index
 * @param[in]  C             Starting C index
 * @param[in]  TENSOR_WIDTH  Number of elements to load from X (width) dimension
 * @param[in]  TENSOR_HEIGHT Number of elements to load from Y (height) dimension
 * @param[in]  TENSOR_DEPTH  Number of elements to load from Z (depth) dimension
 * @param[in]  STRIDE_Y      Stride Y (in bytes)
 * @param[out] xi            A tile with (TILE_WIDTH x TILE_HEIGHT) values with the indirect X coordinate
 * @param[out] yi            A tile with (TILE_WIDTH x TILE_HEIGHT) values with the indirect Y coordinate
 * @param[out] zi            A tile with (TILE_WIDTH x TILE_HEIGHT) values with the indirect Z coordinate
 * @param[out] dst           Output tile
 */
#define T_LOAD_NDHWC_INDIRECT(DATA_TYPE, TILE_AREA, TILE_CHANNELS, TENSOR_TYPE, TENSOR, B, Z, Y, X, C, TENSOR_WIDTH, TENSOR_HEIGHT, TENSOR_DEPTH, STRIDE_Y, xi, yi, zi, dst) \
    ({                                                                                                                                                                \
        LOOP_UNROLLING(int, _i, 0, 1, TILE_AREA,                                                                                                                      \
        {                                                                                                                                                             \
            int _src_y = (X) + xi[_i].v + ((Y) + yi[_i].v) * (TENSOR_WIDTH) + ((Z) + zi[_i].v) * (TENSOR_WIDTH * TENSOR_HEIGHT);                                      \
            _src_y += (B) * (int)(TENSOR_WIDTH) * (int)(TENSOR_HEIGHT) * (int)(TENSOR_DEPTH);                                                                         \
            int _src_valid_y = (((X) + xi[_i].v) >= 0 && ((X) + xi[_i].v) < (int)(TENSOR_WIDTH) && ((Y) + yi[_i].v) >= 0 && ((Y) + yi[_i].v) < (int)(TENSOR_HEIGHT)   \
                             && ((Z) + zi[_i].v) >= 0 && ((Z) + zi[_i].v) < (int)(TENSOR_DEPTH));                                                                     \
            if(_src_valid_y != 0)                                                                                                                                     \
            {                                                                                                                                                         \
                dst[_i].v = V_LOAD(DATA_TYPE, TILE_CHANNELS, TENSOR_TYPE, TENSOR, C, _src_y, STRIDE_Y);                                                               \
            }                                                                                                                                                         \
        })                                                                                                                                                            \
    })

/** Store a tile to global memory (tensor) using an indirect Y index tile and conditionally use a different length for the store
 *
 * @note If WIDTH1_CONDITION is true, the store will use the WIDTH1 length for the store
 * @note The vectors are stored in reverse order so the invalid rows are overwritten by the valid ones
 *
 * @param[in] DATA_TYPE        Data type
 * @param[in] HEIGHT           Number of src rows
 * @param[in] WIDTH0           Store width to use if WIDTH1_CONDITION = false
 * @param[in] WIDTH1           Store width to use if WIDTH1_CONDITION = true
 * @param[in] TENSOR_TYPE      Type of cl_type used to store the tensor in global memory (BUFFER=cl_buffer, IMAGE=cl_image). Currently BUFFER only is supported
 *                             cl_image is not supported.
 * @param[in] TENSOR           Tensor basename
 * @param[in] X                Starting X position
 * @param[in] STRIDE_Y         Stride Y (in bytes)
 * @param[in] WIDTH1_CONDITION Condition to select the WIDTH1 store
 * @param[in] src              Input tile
 * @param[in] indirect_y       Indirect Y index tile
 */
#define T_STORE_INDIRECT_WIDTH_SELECT(DATA_TYPE, HEIGHT, WIDTH0, WIDTH1, TENSOR_TYPE, TENSOR, X, STRIDE_Y, WIDTH1_CONDITION, src, indirect_y)                                                      \
    ({                                                                                                                                                                                             \
        if(WIDTH1_CONDITION)                                                                                                                                                                       \
        {                                                                                                                                                                                          \
            LOOP_UNROLLING(int, _i, 0, 1, HEIGHT,                                                                                                                                                  \
            {                                                                                                                                                                                      \
                VSTORE_PARTIAL(WIDTH0, WIDTH1)                                                                                                                                                     \
                (CONVERT(src[HEIGHT - 1 - _i].v, VEC_DATA_TYPE(DATA_TYPE, WIDTH0)), 0, (__global DATA_TYPE *)(TENSOR##_ptr + TENSOR##_offset_first_element_in_bytes + (X) * sizeof(DATA_TYPE) + (indirect_y[HEIGHT - 1 - _i].v) * STRIDE_Y)); \
            })                                                                                                                                                                                     \
        }                                                                                                                                                                                          \
        else                                                                                                                                                                                       \
        {                                                                                                                                                                                          \
            LOOP_UNROLLING(int, _i, 0, 1, HEIGHT,                                                                                                                                                  \
            {                                                                                                                                                                                      \
                VSTORE(WIDTH0)                                                                                                                                                                     \
                (CONVERT(src[HEIGHT - 1 - _i].v, VEC_DATA_TYPE(DATA_TYPE, WIDTH0)), 0, (__global DATA_TYPE *)(TENSOR##_ptr + TENSOR##_offset_first_element_in_bytes + (X) * sizeof(DATA_TYPE) + (indirect_y[HEIGHT - 1 - _i].v) * STRIDE_Y)); \
            })                                                                                                                                                                                     \
        }                                                                                                                                                                                          \
    })

/** Offset correction for the QASYMM8 computation
 *
 * @param[in]  ACC_DATA_TYPE Accumulator data type
 * @param[in]  M0            Number of src/dst rows
 * @param[in]  N0            Number of src/dst columns
 * @param[in]  K0            Number of src columns
 * @param[in]  SRC_OFFSET    Source quantization offset
 * @param[in]  WEI_OFFSET    Weights quantization shift
 * @param[in]  lhs           LHS tile
 * @param[in]  rhs           RHS tile
 * @param[out] dst           DST tile
 */
#define T_OFFSET_CORRECTION(ACC_DATA_TYPE, M0, N0, K0, SRC_OFFSET, WEI_OFFSET, lhs, rhs, dst)        \
    ({                                                                                               \
        LOOP_UNROLLING(int, _m0, 0, 1, M0,                                                           \
        {                                                                                            \
            ACC_DATA_TYPE _tm = 0;                                                                   \
            LOOP_UNROLLING(int, _k0, 0, 1, K0,                                                       \
            {                                                                                        \
                _tm += ((ACC_DATA_TYPE)lhs[_m0].s[_k0] * (ACC_DATA_TYPE)WEI_OFFSET);                 \
            })                                                                                       \
            LOOP_UNROLLING(int, _n0, 0, 1, N0,                                                       \
            {                                                                                        \
                dst[_m0].s[_n0] += _tm;                                                              \
                LOOP_UNROLLING(int, _k0, 0, 1, K0,                                                   \
                {                                                                                    \
                    dst[_m0].s[_n0] += ((ACC_DATA_TYPE)rhs[_n0].s[_k0] * (ACC_DATA_TYPE)SRC_OFFSET); \
                })                                                                                   \
            })                                                                                       \
        })                                                                                          \
    })

/** 8-bit quantization with fixed-point scale
 *
 * @param[in]  SRC_DATA_TYPE     SRC data type
 * @param[in]  DST_DATA_TYPE     DST data type
 * @param[in]  QUANTIZATION_TYPE Quantization type (PER_TENSOR or PER_CHANNEL)
 * @param[in]  M0                Number of src/dst rows
 * @param[in]  N0                Number of src/dst columns
 * @param[in]  DST_OFFSET        Quantization offset used for both the per-tensor and per-channel quantization
 * @param[in]  DST_SHIFT         Quantization shift for the per-tensor quantization
 * @param[in]  DST_MULTIPLIER    Quantization multiplier for the per-tensor quantization
 * @param[in]  src               Input tile
 * @param[in]  dst_multipliers   Output multipliers tile for the per-channel quantization
 * @param[in]  dst_shifts        Output shift tile for the per-channel quantization
 * @param[out] dst               Output tile
 */
#define T_QUANTIZE8(SRC_DATA_TYPE, DST_DATA_TYPE, QUANTIZATION_TYPE, M0, N0, DST_OFFSET, DST_SHIFT, DST_MULTIPLIER, src, dst_multipliers, dst_shifts, dst) T_QUANTIZE8_STR(SRC_DATA_TYPE, DST_DATA_TYPE, QUANTIZATION_TYPE, M0, N0, DST_OFFSET, DST_SHIFT, DST_MULTIPLIER, src, dst_multipliers, dst_shifts, dst)
#define T_QUANTIZE8_STR(SRC_DATA_TYPE, DST_DATA_TYPE, QUANTIZATION_TYPE, M0, N0, DST_OFFSET, DST_SHIFT, DST_MULTIPLIER, src, dst_multipliers, dst_shifts, dst) T_QUANTIZE8_##QUANTIZATION_TYPE(SRC_DATA_TYPE, DST_DATA_TYPE, M0, N0, DST_OFFSET, DST_SHIFT, DST_MULTIPLIER, src, dst_multipliers, dst_shifts, dst)

/** 8-bit per-tensor quantization with fixed-point scale
 *
 * @param[in]  SRC_DATA_TYPE   SRC data type
 * @param[in]  DST_DATA_TYPE   DST data type
 * @param[in]  M0              Number of src/dst rows
 * @param[in]  N0              Number of src/dst columns
 * @param[in]  DST_OFFSET      Quantization offset
 * @param[in]  DST_SHIFT       Quantization shift for the per-tensor quantization
 * @param[in]  DST_MULTIPLIER  Quantization multiplier for the per-tensor quantization
 * @param[in]  src             Input tile
 * @param[in]  dst_multipliers (unused)
 * @param[in]  dst_shifts      (unused)
 * @param[out] dst             Output tile
 */
#define T_QUANTIZE8_PER_TENSOR(SRC_DATA_TYPE, DST_DATA_TYPE, M0, N0, DST_OFFSET, DST_SHIFT, DST_MULTIPLIER, src, dst_multipliers, dst_shifts, dst)                          \
    ({ \
        LOOP_UNROLLING(int, _m0, 0, 1, M0, \
        { \
            LOOP_UNROLLING(int, _n0, 0, 1, N0, \
            { \
                SRC_DATA_TYPE _tmp = 0; \
                SRC_DATA_TYPE _src = src[_m0].s[_n0]; \
                _src *= select((SRC_DATA_TYPE)1, ((SRC_DATA_TYPE)1 << (SRC_DATA_TYPE)(-DST_SHIFT)), ((SRC_DATA_TYPE)DST_SHIFT < (SRC_DATA_TYPE)0)); \
                SRC_DATA_TYPE overflow = _src == DST_MULTIPLIER && _src == INT_MIN; \
                long a_64 = (long)(_src); \
                long b_64 = (long)(DST_MULTIPLIER); \
                long ab_64 = a_64 * b_64; \
                long mask1 = 1 << 30; \
                long mask2 = 1 - (1 << 30); \
                long is_positive_or_zero = ab_64 >= 0; \
                long nudge = select(mask2, mask1, is_positive_or_zero); \
                SRC_DATA_TYPE ab_x2_high32 = CONVERT((ab_64 + nudge) / (long)(1ll << 31), SRC_DATA_TYPE); \
                _tmp = select(ab_x2_high32, (SRC_DATA_TYPE)INT_MAX, overflow); \
                if(DST_SHIFT >= 0) \
                { \
                    long mask = ((((int)1) << DST_SHIFT) - (long)1); \
                    long threshold = _tmp < (int)0 ? (mask >> 1) + (long)1 : (mask >> 1) + 0; \
                    _tmp = (_tmp & mask) > threshold ? (_tmp >> DST_SHIFT) + (int)1 : (_tmp >> DST_SHIFT); \
                } \
                _tmp += DST_OFFSET; \
                dst[_m0].s[_n0] = CONVERT_SAT(_tmp, DST_DATA_TYPE);                                                                            \
            })                                                                                                                                          \
        })                                                                                                                                          \
    })

/** 8-bit per-channel quantization with fixed-point scale
 *
 * @param[in]  SRC_DATA_TYPE   SRC data type
 * @param[in]  DST_DATA_TYPE   DST data type
 * @param[in]  M0              Number of src/dst rows
 * @param[in]  N0              Number of src/dst columns
 * @param[in]  DST_OFFSET      Quantization offset
 * @param[in]  DST_SHIFT       (unused)
 * @param[in]  DST_MULTIPLIER  (unused)
 * @param[in]  src             Input tile
 * @param[in]  dst_multipliers Output multipliers tile for the per-channel quantization
 * @param[in]  dst_shifts      Output shift tile for the per-channel quantization
 * @param[out] dst             Output tile
 */
#define T_QUANTIZE8_PER_CHANNEL(SRC_DATA_TYPE, DST_DATA_TYPE, M0, N0, DST_OFFSET, DST_SHIFT, DST_MULTIPLIER, src, dst_multipliers, dst_shifts, dst)                          \
    ({ \
        LOOP_UNROLLING(int, _m0, 0, 1, M0, \
        { \
            LOOP_UNROLLING(int, _n0, 0, 1, N0, \
            { \
                SRC_DATA_TYPE _tmp = 0; \
                SRC_DATA_TYPE _tmp2 = 0; \
                SRC_DATA_TYPE _src = src[_m0].s[_n0]; \
                SRC_DATA_TYPE _dst_multiplier = dst_multipliers[0].s[_n0]; \
                SRC_DATA_TYPE _dst_shift = dst_shifts[0].s[_n0]; \
                _src *= select((SRC_DATA_TYPE)1, ((SRC_DATA_TYPE)1 << (SRC_DATA_TYPE)(-_dst_shift)), ((SRC_DATA_TYPE)_dst_shift < (SRC_DATA_TYPE)0)); \
                SRC_DATA_TYPE overflow = _src == _dst_multiplier && _src == INT_MIN; \
                long a_64 = (long)(_src); \
                long b_64 = (long)(_dst_multiplier); \
                long ab_64 = a_64 * b_64; \
                long mask1 = 1 << 30; \
                long mask2 = 1 - (1 << 30); \
                long is_positive_or_zero = ab_64 >= 0; \
                long nudge = select(mask2, mask1, is_positive_or_zero); \
                SRC_DATA_TYPE ab_x2_high32 = CONVERT((ab_64 + nudge) / (long)(1ll << 31), SRC_DATA_TYPE); \
                _tmp = select(ab_x2_high32, (SRC_DATA_TYPE)INT_MAX, overflow); \
                long mask = ((((int)1) << _dst_shift) - (int)1); \
                long threshold = (mask >> 1) + any(_tmp); \
                _tmp2 = _tmp >> _dst_shift; \
                _tmp2 += select(0, 1, (_tmp & mask) > threshold); \
                _tmp = select(_tmp, _tmp2, _dst_shift >= 0); \
                _tmp += DST_OFFSET; \
                dst[_m0].s[_n0] = CONVERT_SAT(_tmp, DST_DATA_TYPE);                                                                            \
            })                                                                                                                                          \
        })                                                                                                                                         \
    })

/** Quantized the 8-bit tile with fixed-point scale for asymmetric
 *
 * @param[in]  SRC_DATA_TYPE  SRC data type
 * @param[in]  DST_DATA_TYPE  DST data type
 * @param[in]  M0             Number of src/dst rows
 * @param[in]  N0             Number of src/dst columns
 * @param[in]  DST_OFFSET     Quantization offset used for both the per-tensor and per-channel quantization
 * @param[in]  DST_SHIFT      Quantization shift for the per-tensor quantization
 * @param[in]  DST_MULTIPLIER Quantization multiplier for the per-tensor quantization
 * @param[in]  src            Input tile
 * @param[out] dst            Output tile
 */
#define T_QUANTIZE8_ASYMMETRIC(SRC_DATA_TYPE, DST_DATA_TYPE, M0, N0, DST_OFFSET, DST_SHIFT, DST_MULTIPLIER, src, dst)                          \
    ({ \
        LOOP_UNROLLING(int, _m0, 0, 1, M0, \
        { \
            LOOP_UNROLLING(int, _n0, 0, 1, N0, \
            { \
                SRC_DATA_TYPE _tmp = 0; \
                SRC_DATA_TYPE _src = src[_m0].s[_n0]; \
                _src *= select((SRC_DATA_TYPE)1, ((SRC_DATA_TYPE)1 << (SRC_DATA_TYPE)(-DST_SHIFT)), ((SRC_DATA_TYPE)DST_SHIFT < (SRC_DATA_TYPE)0)); \
                SRC_DATA_TYPE overflow = _src == DST_MULTIPLIER && _src == INT_MIN; \
                long a_64 = (long)(_src); \
                long b_64 = (long)(DST_MULTIPLIER); \
                long ab_64 = a_64 * b_64; \
                long mask1 = 1 << 30; \
                long mask2 = 1 - (1 << 30); \
                long is_positive_or_zero = ab_64 >= 0; \
                long nudge = select(mask2, mask1, is_positive_or_zero); \
                SRC_DATA_TYPE ab_x2_high32 = CONVERT((ab_64 + nudge) / (long)(1ll << 31), SRC_DATA_TYPE); \
                _tmp = select(ab_x2_high32, (SRC_DATA_TYPE)INT_MAX, overflow); \
                if(DST_SHIFT >= 0) \
                { \
                    long mask = ((((int)1) << DST_SHIFT) - (int)1); \
                    long threshold = _tmp < (int)0 ? (mask >> 1) + (long)1 : (mask >> 1) + 0; \
                    _tmp = (_tmp & mask) > threshold ? (_tmp >> DST_SHIFT) + (int)1 : (_tmp >> DST_SHIFT); \
                } \
                _tmp += DST_OFFSET; \
                dst[_m0].s[_n0] = CONVERT_SAT(_tmp, DST_DATA_TYPE);                                                                            \
            })                                                                                                                                          \
        })                                                                                                                                          \
    })

/** Conditional rowset (memset by row)
 *
 * @note Set the row to VALUE_TO_SET if the corresponding mask == 0
 *
 * @param[in]      DATA_TYPE    Data type
 * @param[in]      M0           Number of LHS rows
 * @param[in]      N0           Number of LHS columns
 * @param[in]      VALUE_TO_SET Value to set the row
 * @param[in, out] a            Input/output tile
 * @param[out]     mask         Mask to check for setting the row to VALUE_TO_SET
 */
#define T_ROWSET_MASK(DATA_TYPE, M0, N0, VALUE_TO_SET, a, mask)                                                                                            \
    ({                                                                                                                                                     \
        LOOP_UNROLLING(int, _m0, 0, 1, M0,                                                                                                                 \
        {                                                                                                                                                  \
            LOOP_UNROLLING(int, _n0, 0, 1, N0,                                                                                                             \
            {                                                                                                                                              \
                a[_m0].s[_n0] = select((DATA_TYPE)(a[_m0].s[_n0]), (DATA_TYPE)(VALUE_TO_SET), (SELECT_DATA_TYPE(DATA_TYPE))(mask[_m0].v == (DATA_TYPE)0)); \
            })                                                                                                                                             \
        })                                                                                                                                                 \
    })

/** Element-wise activation for floating point types
 *
 * @note Performs: activation(LHS) = DST
 *
 * @param[in]  DATA_TYPE       SRC/DST data type
 * @param[in]  M0              Number of SRC/DST rows
 * @param[in]  N0              Number of SRC/DST columns
 * @param[in]  ACTIVATION_TYPE Activation type
 * @param[in]  A_VAL           A value used for the activation (e.g. tanh_op, brelu,..)
 * @param[in]  B_VAL           B value used for the activation (e.g. tanh_op, brelu,..)
 * @param[out] src             SRC tile
 * @param[out] dst             DST tile
 */
#define T_ACTIVATION(DATA_TYPE, M0, N0, ACTIVATION_TYPE, A_VAL, B_VAL, src, dst)               \
    ({                                                                                         \
        LOOP_UNROLLING(int, _m0, 0, 1, M0,                                                     \
        {                                                                                      \
            dst[_m0].v = ACTIVATION(ACTIVATION_TYPE, DATA_TYPE, N0, src[_m0].v, A_VAL, B_VAL); \
        })                                                                                     \
    })

// RELU Activation
#define relu_op_quantized(DATA_TYPE, VEC_SIZE, ZERO_VALUE, A_VAL, B_VAL, x) (max((DATA_TYPE)ZERO_VALUE, x))
// Bounded RELU Activation
#define brelu_op_quantized(DATA_TYPE, VEC_SIZE, ZERO_VALUE, A_VAL, B_VAL, x) (min((DATA_TYPE)A_VAL, max((DATA_TYPE)ZERO_VALUE, x)))
// Lower Upper Bounded RELU Activation
#define lu_brelu_op_quantized(DATA_TYPE, VEC_SIZE, ZERO_VALUE, A_VAL, B_VAL, x) (min(max(x, (DATA_TYPE)B_VAL), (DATA_TYPE)A_VAL))
// Hard Swish Activation
#define hard_swish_op_quantized(DATA_TYPE, VEC_SIZE, ZERO_VALUE, A_VAL, B_VAL, x) (x * ((min(max((DATA_TYPE)(x + (DATA_TYPE)3.f), (DATA_TYPE)0.f), (DATA_TYPE)6.f)) * (DATA_TYPE)0.166666667f))
// Identity Activation
#define identity_op_quantized(DATA_TYPE, VEC_SIZE, ZERO_VALUE, A_VAL, B_VAL, x) (x)

#define ACT_OP_QUANTIZED(op, DATA_TYPE, VEC_SIZE, ZERO_VALUE, A_VAL, B_VAL, x) op##_op_quantized(DATA_TYPE, VEC_SIZE, ZERO_VALUE, A_VAL, B_VAL, x)
#define ACTIVATION_QUANTIZED(op, DATA_TYPE, VEC_SIZE, ZERO_VALUE, A_VAL, B_VAL, x) ACT_OP_QUANTIZED(op, DATA_TYPE, VEC_SIZE, ZERO_VALUE, A_VAL, B_VAL, x)

#define V_ADD(A_VAL, B_VAL) ((A_VAL) + (B_VAL))
#define V_SUB(A_VAL, B_VAL) ((A_VAL) - (B_VAL))
#define V_DIV(A_VAL, B_VAL) ((A_VAL) / (B_VAL))
#define V_MUL(A_VAL, B_VAL) ((A_VAL) * (B_VAL))

/** Element-wise activation for quantized types
 *
 * @note Performs: activation(LHS) = DST
 *
 * @param[in]  DATA_TYPE       SRC/DST data type
 * @param[in]  M0              Number of SRC/DST rows
 * @param[in]  N0              Number of SRC/DST columns
 * @param[in]  ACTIVATION_TYPE Activation type
 * @param[in]  ZERO_VALUE      The zero value to consider in the computation
 * @param[in]  A_VAL           A value used for the activation (e.g. tanh_op, brelu,..)
 * @param[in]  B_VAL           B value used for the activation (e.g. tanh_op, brelu,..)
 * @param[out] src             SRC tile
 * @param[out] dst             DST tile
 */
#define T_ACTIVATION_QUANTIZED(DATA_TYPE, M0, N0, ACTIVATION_TYPE, ZERO_VALUE, A_VAL, B_VAL, src, dst)               \
    ({ \
        LOOP_UNROLLING(int, _m0, 0, 1, M0, \
        { \
            dst[_m0].v = ACTIVATION_QUANTIZED(ACTIVATION_TYPE, DATA_TYPE, N0, ZERO_VALUE, A_VAL, B_VAL, src[_m0].v); \
        })                                                                                          \
    })

/** Element-wise addition between two tiles
 *
 * @note Performs: LHS + RHS = DST
 *
 * @param[in]  DATA_TYPE LHS/RHS/DST data type
 * @param[in]  M0        Number of LHS rows
 * @param[in]  N0        Number of LHS columns
 * @param[in]  lhs       LHS tile
 * @param[in]  rhs       Constant RHS tile
 * @param[out] dst       DST tile
 */
#define T_ADD(DATA_TYPE, M0, N0, lhs, rhs, dst) \
    ({                                                            \
        LOOP_UNROLLING(int, _m0, 0, 1, M0,                        \
        {                                                         \
            dst[_m0].v = lhs[_m0].v + rhs[_m0].v; \
        })                                                        \
    })

/** Element-wise addition with a constant value
 *
 * @note Performs: LHS + constant = DST
 *
 * @param[in]  DATA_TYPE    LHS/RHS/DST data type
 * @param[in]  M0           Number of LHS rows
 * @param[in]  N0           Number of LHS columns
 * @param[in]  lhs          LHS tile
 * @param[in]  rhs_constant Constant value
 * @param[out] dst          DST tile
 */
#define T_ADD_CONSTANT(DATA_TYPE, M0, N0, lhs, rhs_constant, dst) \
    ({                                                            \
        LOOP_UNROLLING(int, _m0, 0, 1, M0,                        \
        {                                                         \
            dst[_m0].v = lhs[_m0].v + (DATA_TYPE)rhs_constant;               \
        })                                                        \
    })

#define T_ELTWISE_BROADCAST_ADD_X(DST_DATA_TYPE, M0, N0, lhs, rhs, dst) T_ELTWISE_BROADCAST_X(V_ADD, DST_DATA_TYPE, M0, N0, lhs, rhs, dst)
#define T_ELTWISE_BROADCAST_LHS_X_ADD(DST_DATA_TYPE, M0, N0, lhs, rhs, dst) T_ELTWISE_BROADCAST_LHS_X(V_ADD, DST_DATA_TYPE, M0, N0, lhs, rhs, dst)
#define T_ELTWISE_BROADCAST_RHS_X_ADD(DST_DATA_TYPE, M0, N0, lhs, rhs, dst) T_ELTWISE_BROADCAST_X(V_ADD, DST_DATA_TYPE, M0, N0, lhs, rhs, dst)

#define T_ELTWISE_BROADCAST_LHS_X_SUB(DST_DATA_TYPE, M0, N0, lhs, rhs, dst) T_ELTWISE_BROADCAST_LHS_X(V_SUB, DST_DATA_TYPE, M0, N0, lhs, rhs, dst)
#define T_ELTWISE_BROADCAST_RHS_X_SUB(DST_DATA_TYPE, M0, N0, lhs, rhs, dst) T_ELTWISE_BROADCAST_X(V_SUB, DST_DATA_TYPE, M0, N0, lhs, rhs, dst)

#define T_ELTWISE_BROADCAST_DIV_X(DST_DATA_TYPE, M0, N0, lhs, rhs, dst) T_ELTWISE_BROADCAST_X(V_DIV, DST_DATA_TYPE, M0, N0, lhs, rhs, dst)

#define T_ELTWISE_BROADCAST_LHS_X_MUL(DST_DATA_TYPE, M0, N0, lhs, rhs, dst) T_ELTWISE_BROADCAST_LHS_X(V_MUL, DST_DATA_TYPE, M0, N0, lhs, rhs, dst)
#define T_ELTWISE_BROADCAST_RHS_X_MUL(DST_DATA_TYPE, M0, N0, lhs, rhs, dst) T_ELTWISE_BROADCAST_X(V_MUL, DST_DATA_TYPE, M0, N0, lhs, rhs, dst)

/** Element-wise scale with a constant value
 *
 * @note Performs: LHS * constant = DST
 *
 * @param[in]  DATA_TYPE    LHS/RHS/DST data type
 * @param[in]  M0           Number of LHS rows
 * @param[in]  N0           Number of LHS columns
 * @param[in]  lhs          LHS tile
 * @param[in]  rhs_constant Constant value
 * @param[out] dst          DST tile
 */
#define T_SCALE_CONSTANT(DATA_TYPE, M0, N0, lhs, rhs_constant, dst) \
    ({                                                            \
        LOOP_UNROLLING(int, _m0, 0, 1, M0,                        \
        {                                                         \
            dst[_m0].v = lhs[_m0].v * (DATA_TYPE)rhs_constant; \
        })                                                        \
    })

/** Element-wise operation with RHS broadcasted (RHS has the X dimension only)
 *
 * @note Performs: LHS OP RHS[broadcasted] = DST
 * @note Both tiles must have same data type
 *
 * @param[in]  T_ELWISE_OP   Elementwise operator to perform
 * @param[in]  DST_DATA_TYPE DST data type
 * @param[in]  M0            Number of LHS rows
 * @param[in]  N0            Number of LHS columns
 * @param[in]  lhs           LHS tile
 * @param[in]  rhs           RHS tile
 * @param[out] dst           DST tile
 */
#define T_ELTWISE_BROADCAST_X(T_ELWISE_OP, DST_DATA_TYPE, M0, N0, lhs, rhs, dst) \
    ({                                                      \
        LOOP_UNROLLING(int, _m0, 0, 1, M0,                  \
        {                                                   \
            dst[_m0].v = T_ELWISE_OP(CONVERT(lhs[_m0].v, VEC_DATA_TYPE(DST_DATA_TYPE, N0)), CONVERT(rhs[0].v, VEC_DATA_TYPE(DST_DATA_TYPE, N0)));             \
        })                                                  \
    })

/** Element-wise operation with LHS broadcasted (LHS has the X dimension only)
 *
 * @note Performs: LHS[broadcasted] OP RHS = DST
 * @note Both tiles must have same data type
 *
 * @param[in]  T_ELWISE_OP   Elementwise operator to perform
 * @param[in]  DST_DATA_TYPE DST data type
 * @param[in]  M0            Number of RHS rows
 * @param[in]  N0            Number of RHS columns
 * @param[in]  lhs           LHS tile
 * @param[in]  rhs           RHS tile
 * @param[out] dst           DST tile
 */
#define T_ELTWISE_BROADCAST_LHS_X(T_ELWISE_OP, DST_DATA_TYPE, M0, N0, lhs, rhs, dst) \
    ({                                                      \
        LOOP_UNROLLING(int, _m0, 0, 1, M0,                  \
        {                                                   \
            dst[_m0].v = T_ELWISE_OP(CONVERT(lhs[0].v, VEC_DATA_TYPE(DST_DATA_TYPE, N0)), CONVERT(rhs[_m0].v, VEC_DATA_TYPE(DST_DATA_TYPE, N0)));             \
        })                                                  \
    })

#define T_ELTWISE_ADD(DST_DATA_TYPE, M0, N0, lhs, rhs, dst) T_ELTWISE(V_ADD, DST_DATA_TYPE, M0, N0, lhs, rhs, dst)
#define T_ELTWISE_SUB(DST_DATA_TYPE, M0, N0, lhs, rhs, dst) T_ELTWISE(V_SUB, DST_DATA_TYPE, M0, N0, lhs, rhs, dst)
#define T_ELTWISE_DIV(DST_DATA_TYPE, M0, N0, lhs, rhs, dst) T_ELTWISE(V_DIV, DST_DATA_TYPE, M0, N0, lhs, rhs, dst)
#define T_ELTWISE_MUL(DST_DATA_TYPE, M0, N0, lhs, rhs, dst) T_ELTWISE(V_MUL, DST_DATA_TYPE, M0, N0, lhs, rhs, dst)

/** Element-wise operation between two tiles (LHS and RHS)
 *
 * @note Performs: LHS OP RHS = DST
 * @note Both tiles must have same data type
 *
 * @param[in]  T_ELWISE_OP   Elementwise operator to perform
 * @param[in]  DST_DATA_TYPE DST data type
 * @param[in]  M0            Number of LHS rows
 * @param[in]  N0            Number of LHS columns
 * @param[in]  lhs           LHS tile
 * @param[in]  rhs           RHS tile
 * @param[out] dst           DST tile
 */
#define T_ELTWISE(T_ELWISE_OP, DST_DATA_TYPE, M0, N0, lhs, rhs, dst) \
    ({                                                      \
        LOOP_UNROLLING(int, _m0, 0, 1, M0,                  \
        {                                                   \
            dst[_m0].v = T_ELWISE_OP(CONVERT(lhs[_m0].v, VEC_DATA_TYPE(DST_DATA_TYPE, N0)), CONVERT(rhs[_m0].v, VEC_DATA_TYPE(DST_DATA_TYPE, N0)));             \
        })                                                  \
    })

/** Floor operation on a tile
 *
 * @note Performs: floor(SRC) = DST
 * @note Both tiles must have same data type
 *
 * @param[in]  DST_DATA_TYPE DST data type
 * @param[in]  M0            Number of SRC rows
 * @param[in]  N0            Number of SRC columns
 * @param[in]  src           LHS tile
 * @param[out] dst           DST tile
 */
#define T_FLOOR(DST_DATA_TYPE, M0, N0, src, dst) \
    ({                                                      \
        LOOP_UNROLLING(int, _m0, 0, 1, M0,                  \
        {                                                   \
            dst[_m0].v = floor(CONVERT(src[_m0].v, VEC_DATA_TYPE(DST_DATA_TYPE, N0)));             \
        })                                                  \
    })

/** Matrix multiplication
 *
 * @note Performs: LHS X RHS + DST = DST
 *
 * @param[in]      LHS_DATA_TYPE LHS tile data type
 * @param[in]      RHS_DATA_TYPE RHS tile data type
 * @param[in]      DST_DATA_TYPE RHS tile data type
 * @param[in]      M0            Number of LHS rows
 * @param[in]      N0            Number of RHS columns
 * @param[in]      K0            Number of LHS columns
 * @param[in]      LHS_LAYOUT    LHS layout (T= transposed, NT= not transposed)
 * @param[in]      RHS_LAYOUT    RHS layout (T= transposed, NT= not transposed)
 * @param[in]      lhs           LHS tile
 * @param[in]      rhs           RHS tile
 * @param[in, out] dst           DST tile
 *
 * @note For Int8/UInt8 multiplications, we only have T_MMUL_NT_T because we need
 *       the multiply the rows of Lhs and Rhs tensors to utilize dot product extension.
 *       Addition of other versions requires dealing with on the fly transposition of
 *       these tile elements and therefore is not favored.
 */
#define T_MMUL(LHS_DATA_TYPE, RHS_DATA_TYPE, DST_DATA_TYPE, M0, N0, K0, LHS_LAYOUT, RHS_LAYOUT, lhs, rhs, dst) T_MMUL_##LHS_LAYOUT##_##RHS_LAYOUT(LHS_DATA_TYPE, RHS_DATA_TYPE, DST_DATA_TYPE, M0, N0, K0, lhs, rhs, dst)
#define T_MMUL_NT_T(LHS_DATA_TYPE, RHS_DATA_TYPE, DST_DATA_TYPE, M0, N0, K0, lhs, rhs, dst) T_MMUL_NT_T_##LHS_DATA_TYPE##_##RHS_DATA_TYPE##_##DST_DATA_TYPE(LHS_DATA_TYPE, RHS_DATA_TYPE, DST_DATA_TYPE, M0, N0, K0, lhs, rhs, dst)
#define T_MMUL_NT_T_float_float_float(LHS_DATA_TYPE, RHS_DATA_TYPE, DST_DATA_TYPE, M0, N0, K0, lhs, rhs, dst) T_MMUL_NT_T_FLOAT(LHS_DATA_TYPE, RHS_DATA_TYPE, DST_DATA_TYPE, M0, N0, K0, lhs, rhs, dst)
#define T_MMUL_NT_T_half_half_float(LHS_DATA_TYPE, RHS_DATA_TYPE, DST_DATA_TYPE, M0, N0, K0, lhs, rhs, dst) T_MMUL_NT_T_FLOAT(LHS_DATA_TYPE, RHS_DATA_TYPE, DST_DATA_TYPE, M0, N0, K0, lhs, rhs, dst)
#define T_MMUL_NT_T_half_half_half(LHS_DATA_TYPE, RHS_DATA_TYPE, DST_DATA_TYPE, M0, N0, K0, lhs, rhs, dst) T_MMUL_NT_T_FLOAT(LHS_DATA_TYPE, RHS_DATA_TYPE, DST_DATA_TYPE, M0, N0, K0, lhs, rhs, dst)
#define T_MMUL_NT_T_char_char_int(LHS_DATA_TYPE, RHS_DATA_TYPE, DST_DATA_TYPE, M0, N0, K0, lhs, rhs, dst) T_MMUL_NT_T_INTEGER8(LHS_DATA_TYPE, RHS_DATA_TYPE, DST_DATA_TYPE, M0, N0, K0, lhs, rhs, dst)
#define T_MMUL_NT_T_uchar_uchar_uint(LHS_DATA_TYPE, RHS_DATA_TYPE, DST_DATA_TYPE, M0, N0, K0, lhs, rhs, dst) T_MMUL_NT_T_INTEGER8(LHS_DATA_TYPE, RHS_DATA_TYPE, DST_DATA_TYPE, M0, N0, K0, lhs, rhs, dst)
#define T_MMUL_NT_T_uchar_uchar_int(LHS_DATA_TYPE, RHS_DATA_TYPE, DST_DATA_TYPE, M0, N0, K0, lhs, rhs, dst) T_MMUL_NT_T_INTEGER8(LHS_DATA_TYPE, RHS_DATA_TYPE, DST_DATA_TYPE, M0, N0, K0, lhs, rhs, dst)
#define T_MMUL_NT_T_FLOAT(LHS_DATA_TYPE, RHS_DATA_TYPE, DST_DATA_TYPE, M0, N0, K0, lhs, rhs, dst)                       \
    {                                                                                     \
        LOOP_UNROLLING(int, _m, 0, 1, M0,                                                 \
        {                                                                                 \
            LOOP_UNROLLING(int, _n, 0, 1, N0,                                             \
            {                                                                             \
                LOOP_UNROLLING(int, _k, 0, 1, K0,                                         \
                {                                                                         \
                    dst[_m].s[_n] = fma((DST_DATA_TYPE)(lhs[_m].s[_k]), (DST_DATA_TYPE)(rhs[_n].s[_k]), dst[_m].s[_n]); \
                })                                                                        \
            })                                                                            \
        })                                                                                \
    }

#define T_MMUL_NT_NT(LHS_DATA_TYPE, RHS_DATA_TYPE, DST_DATA_TYPE, M0, N0, K0, lhs, rhs, dst) T_MMUL_NT_NT_##LHS_DATA_TYPE##_##RHS_DATA_TYPE##_##DST_DATA_TYPE(LHS_DATA_TYPE, RHS_DATA_TYPE, DST_DATA_TYPE, M0, N0, K0, lhs, rhs, dst)
#define T_MMUL_NT_NT_float_float_float(LHS_DATA_TYPE, RHS_DATA_TYPE, DST_DATA_TYPE, M0, N0, K0, lhs, rhs, dst) T_MMUL_NT_NT_FLOAT(LHS_DATA_TYPE, RHS_DATA_TYPE, DST_DATA_TYPE, M0, N0, K0, lhs, rhs, dst)
#define T_MMUL_NT_NT_half_half_float(LHS_DATA_TYPE, RHS_DATA_TYPE, DST_DATA_TYPE, M0, N0, K0, lhs, rhs, dst) T_MMUL_NT_NT_FLOAT(LHS_DATA_TYPE, RHS_DATA_TYPE, DST_DATA_TYPE, M0, N0, K0, lhs, rhs, dst)
#define T_MMUL_NT_NT_half_half_half(LHS_DATA_TYPE, RHS_DATA_TYPE, DST_DATA_TYPE, M0, N0, K0, lhs, rhs, dst) T_MMUL_NT_NT_FLOAT(LHS_DATA_TYPE, RHS_DATA_TYPE, DST_DATA_TYPE, M0, N0, K0, lhs, rhs, dst)
#define T_MMUL_NT_NT_FLOAT(LHS_DATA_TYPE, RHS_DATA_TYPE, DST_DATA_TYPE, M0, N0, K0, lhs, rhs, dst)                       \
    {                                                                                                                    \
        LOOP_UNROLLING(int, _m, 0, 1, M0,                                                                                \
        {                                                                                                                \
            LOOP_UNROLLING(int, _k, 0, 1, K0,                                                                            \
            {                                                                                                            \
                dst[_m].v = fma((DST_DATA_TYPE)(lhs[_m].s[_k]), (rhs[_k].v), dst[_m].v);                                 \
            })                                                                                                           \
        })                                                                                                               \
    }

#define T_MMUL_T_NT(LHS_DATA_TYPE, RHS_DATA_TYPE, DST_DATA_TYPE, M0, N0, K0, lhs, rhs, dst) T_MMUL_T_NT_##LHS_DATA_TYPE##_##RHS_DATA_TYPE##_##DST_DATA_TYPE(LHS_DATA_TYPE, RHS_DATA_TYPE, DST_DATA_TYPE, M0, N0, K0, lhs, rhs, dst)
#define T_MMUL_T_NT_float_float_float(LHS_DATA_TYPE, RHS_DATA_TYPE, DST_DATA_TYPE, M0, N0, K0, lhs, rhs, dst) T_MMUL_T_NT_FLOAT(LHS_DATA_TYPE, RHS_DATA_TYPE, DST_DATA_TYPE, M0, N0, K0, lhs, rhs, dst)
#define T_MMUL_T_NT_half_half_float(LHS_DATA_TYPE, RHS_DATA_TYPE, DST_DATA_TYPE, M0, N0, K0, lhs, rhs, dst) T_MMUL_T_NT_FLOAT(LHS_DATA_TYPE, RHS_DATA_TYPE, DST_DATA_TYPE, M0, N0, K0, lhs, rhs, dst)
#define T_MMUL_T_NT_half_half_half(LHS_DATA_TYPE, RHS_DATA_TYPE, DST_DATA_TYPE, M0, N0, K0, lhs, rhs, dst) T_MMUL_T_NT_FLOAT(LHS_DATA_TYPE, RHS_DATA_TYPE, DST_DATA_TYPE, M0, N0, K0, lhs, rhs, dst)
#define T_MMUL_T_NT_FLOAT(LHS_DATA_TYPE, RHS_DATA_TYPE, DST_DATA_TYPE, M0, N0, K0, lhs, rhs, dst)                       \
    {                                                                                     \
        LOOP_UNROLLING(int, _m, 0, 1, M0,                                                 \
        {                                                                                 \
            LOOP_UNROLLING(int, _n, 0, 1, N0,                                             \
            {                                                                             \
                LOOP_UNROLLING(int, _k, 0, 1, K0,                                         \
                {                                                                         \
                    dst[_m].s[_n] = fma((DST_DATA_TYPE)(lhs[_k].s[_m]), (DST_DATA_TYPE)(rhs[_k].s[_n]), dst[_m].s[_n]); \
                })                                                                        \
            })                                                                            \
        })                                                                                \
    }

#define T_MMUL_T_T(LHS_DATA_TYPE, RHS_DATA_TYPE, DST_DATA_TYPE, M0, N0, K0, lhs, rhs, dst) T_MMUL_T_T_##LHS_DATA_TYPE##_##RHS_DATA_TYPE##_##DST_DATA_TYPE(LHS_DATA_TYPE, RHS_DATA_TYPE, DST_DATA_TYPE, M0, N0, K0, lhs, rhs, dst)
#define T_MMUL_T_T_float_float_float(LHS_DATA_TYPE, RHS_DATA_TYPE, DST_DATA_TYPE, M0, N0, K0, lhs, rhs, dst) T_MMUL_T_T_FLOAT(LHS_DATA_TYPE, RHS_DATA_TYPE, DST_DATA_TYPE, M0, N0, K0, lhs, rhs, dst)
#define T_MMUL_T_T_half_half_float(LHS_DATA_TYPE, RHS_DATA_TYPE, DST_DATA_TYPE, M0, N0, K0, lhs, rhs, dst) T_MMUL_T_T_FLOAT(LHS_DATA_TYPE, RHS_DATA_TYPE, DST_DATA_TYPE, M0, N0, K0, lhs, rhs, dst)
#define T_MMUL_T_T_half_half_half(LHS_DATA_TYPE, RHS_DATA_TYPE, DST_DATA_TYPE, M0, N0, K0, lhs, rhs, dst) T_MMUL_T_T_FLOAT(LHS_DATA_TYPE, RHS_DATA_TYPE, DST_DATA_TYPE, M0, N0, K0, lhs, rhs, dst)
#define T_MMUL_T_T_FLOAT(LHS_DATA_TYPE, RHS_DATA_TYPE, DST_DATA_TYPE, M0, N0, K0, lhs, rhs, dst)                       \
    {                                                                                     \
        LOOP_UNROLLING(int, _m, 0, 1, M0,                                                 \
        {                                                                                 \
            LOOP_UNROLLING(int, _n, 0, 1, N0,                                             \
            {                                                                             \
                LOOP_UNROLLING(int, _k, 0, 1, K0,                                         \
                {                                                                         \
                    dst[_m].s[_n] = fma((DST_DATA_TYPE)(lhs[_k].s[_m]), (DST_DATA_TYPE)(rhs[_n].s[_k]), dst[_m].s[_n]); \
                })                                                                        \
            })                                                                            \
        })                                                                                \
    }

#define T_MMUL_NT_T_INTEGER8(LHS_DATA_TYPE, RHS_DATA_TYPE, DST_DATA_TYPE, M0, N0, K0, lhs, rhs, dst)                            \
    ({ \
        LOOP_UNROLLING(int, _m, 0, 1, M0, \
        { \
            LOOP_UNROLLING(int, _n, 0, 1, N0, \
            { \
                DOT_PRODUCT_INTEGER8(LHS_DATA_TYPE, RHS_DATA_TYPE, DST_DATA_TYPE, K0, (lhs[_m].v), (rhs[_n].v), dst[_m].s[_n]); \
            })                                                                                             \
        })                                                                                             \
    })

#endif /* ACL_SRC_CORE_CL_CL_KERNELS_TILE_HELPERS */
