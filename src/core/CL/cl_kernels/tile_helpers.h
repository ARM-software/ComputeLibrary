/*
 * Copyright (c) 2021 Arm Limited.
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

/** Tile object
 *  A tile object is a 2D memory block and can be accessed using the following syntax:
 *  -# a[m0].v    = access the the vector at row "m0" (OpenCL vector)
 *  -# a[m0].s[x] = access the scalar element at row "m0" and column "n0" (scalar access)
 *
 * @param[in] DATA_TYPE Data type of the tile
 * @param[in] H         Number of tile rows
 * @param[in] W         Number of tile colums
 * @param[in] BASENAME  Tile's name
 */
#define TILE(DATA_TYPE, H, W, BASENAME) TILE_STR(DATA_TYPE, H, W, BASENAME)
#define TILE_STR(DATA_TYPE, H, W, BASENAME) \
    union {                                 \
        DATA_TYPE    s[W];                  \
        DATA_TYPE##W v;                     \
    } BASENAME[H]

#define TENSOR4D_IMAGE(name)             \
    __read_only image2d_t name##_img,    \
    __global uchar *name##_ptr,      \
    uint            name##_stride_x, \
    uint            name##_step_x,   \
    uint            name##_stride_y, \
    uint            name##_step_y,   \
    uint            name##_stride_z, \
    uint            name##_step_z,   \
    uint            name##_stride_w, \
    uint            name##_step_w,   \
    uint            name##_offset_first_element_in_bytes

#define TENSOR4D_BUFFER(name)        \
    __global uchar *name##_ptr,      \
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

/** Loop unrolling */
#define LOOP_UNROLLING(DATA_TYPE, VAR, START_IDX, NUM_ITERATIONS, STEP) \
    _Pragma("unroll") for(DATA_TYPE VAR = START_IDX; VAR < NUM_ITERATIONS; VAR += STEP)

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

/** Offset (in bytes) calculation for a 1D BUFFER (cl_buffer) tensor */
#define OFFSET1D(base, data_type, x) (base##_offset_first_element_in_bytes + x * sizeof(data_type))

/** Offset (in bytes) calculation for a 2D BUFFER (cl_buffer) tensor */
#define OFFSET2D(base, data_type, x, y) (base##_offset_first_element_in_bytes + x * sizeof(data_type) + y * base##_stride_y)

/** Offset (in bytes) calculation for a 3D BUFFER (cl_buffer) tensor */
#define OFFSET3D(base, data_type, x, y, z) (base##_offset_first_element_in_bytes + x * sizeof(data_type) + y * base##_stride_y + z * base##_stride_z)

/** Offset (in bytes) calculation for a 4D BUFFER (cl_buffer) tensor */
#define OFFSET4D(base, data_type, x, y, z, w) (base##_offset_first_element_in_bytes + x * sizeof(data_type) + y * base##_stride_y + z * base##_stride_z + w * base##_stride_w)

/** Dot product integet 8bit function
 *
 *  @note Performs: c += dot(a, b)
 *
 * @param[in] DST_DATA_TYPE Accumulator data type
 * @param[in] K0            Number of accumulations
 * @param[in] a             OpenCL vector a
 * @param[in] b             OpenCL vector b
 * @param[in] c             Scalar variable c
 */
#define DOT_PRODUCT_INTEGER8(DST_DATA_TYPE, K0, a, b, c) DOT_PRODUCT_INTEGER8_STR(DST_DATA_TYPE, K0, a, b, c)
#define DOT_PRODUCT_INTEGER8_STR(DST_DATA_TYPE, K0, a, b, c) DOT_PRODUCT##K0##_INTEGER8(DST_DATA_TYPE, a, b, c)
#define DOT_PRODUCT1_INTEGER8(DST_DATA_TYPE, a, b, c) \
    ({                                                \
        c += (DST_DATA_TYPE)a * (DST_DATA_TYPE)b;     \
    })
#define DOT_PRODUCT2_INTEGER8(DST_DATA_TYPE, a, b, c)   \
    ({                                                  \
        c += (DST_DATA_TYPE)a.s0 * (DST_DATA_TYPE)b.s0; \
        c += (DST_DATA_TYPE)a.s1 * (DST_DATA_TYPE)b.s1; \
    })
#define DOT_PRODUCT3_INTEGER8(DST_DATA_TYPE, a, b, c)   \
    ({                                                  \
        DOT_PRODUCT2_INTEGER8(DST_DATA_TYPE, a, b, c);  \
        c += (DST_DATA_TYPE)a.s2 * (DST_DATA_TYPE)b.s2; \
    })
#if defined(ARM_COMPUTE_OPENCL_DOT8_ACC_ENABLED) && defined(cl_arm_integer_dot_product_accumulate_int8)
#define DOT_PRODUCT4_INTEGER8(DST_DATA_TYPE, x, y, val) val = arm_dot_acc((x), (y), (val));
#elif defined(ARM_COMPUTE_OPENCL_DOT8_ENABLED) && defined(cl_arm_integer_dot_product_int8) // defined(ARM_COMPUTE_OPENCL_DOT8_ENABLED) && defined(cl_arm_integer_dot_product_int8)
#define DOT_PRODUCT4_INTEGER8(DST_DATA_TYPE, x, y, val) val += arm_dot((x), (y));
#else // defined(ARM_COMPUTE_OPENCL_DOT8_ACC_ENABLED) && defined(cl_arm_integer_dot_product_accumulate_int8)
#define DOT_PRODUCT4_INTEGER8(DST_DATA_TYPE, x, y, val)   \
    ({                                                    \
        val += (DST_DATA_TYPE)x.s0 * (DST_DATA_TYPE)y.s0; \
        val += (DST_DATA_TYPE)x.s1 * (DST_DATA_TYPE)y.s1; \
        val += (DST_DATA_TYPE)x.s2 * (DST_DATA_TYPE)y.s2; \
        val += (DST_DATA_TYPE)x.s3 * (DST_DATA_TYPE)y.s3; \
    })
#endif // defined(ARM_COMPUTE_OPENCL_DOT8_ACC_ENABLED) && defined(cl_arm_integer_dot_product_accumulate_int8)
#define DOT_PRODUCT8_INTEGER8(DST_DATA_TYPE, a, b, c) \
    ({                                                \
        DOT_PRODUCT4_INTEGER8((a.lo), (b.lo), c);     \
        DOT_PRODUCT4_INTEGER8((a.hi), (b.hi), c);     \
    })
#define DOT_PRODUCT16_INTEGER8(DST_DATA_TYPE, a, b, c) \
    ({                                                 \
        DOT_PRODUCT8_INTEGER8((a.lo), (b.lo), c);      \
        DOT_PRODUCT8_INTEGER8((a.hi), (b.hi), c);      \
    })

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
    (0, (__global DATA_TYPE *)(TENSOR##_ptr + TENSOR##_offset_first_element_in_bytes + (X) * sizeof(DATA_TYPE) + (Y)*STRIDE_Y))
#define V_LOAD_IMAGE(DATA_TYPE, WIDTH, TENSOR, X, Y, STRIDE_Y) READ_IMAGE2D(DATA_TYPE, CONVERT_VECTOR_SIZE_TO_PIXEL_UNIT(WIDTH), TENSOR##_img, (X) / 4, (Y))

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
    ({ \
        LOOP_UNROLLING(int, _i, 0, HEIGHT, 1) \
        { \
            dst[_i].v = V_LOAD(DATA_TYPE, WIDTH, TENSOR_TYPE, TENSOR, X, ((Y) + _i * (int)(YI_MULTIPLIER)), STRIDE_Y); \
        }                                                                                                                 \
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
    ({ \
        LOOP_UNROLLING(int, _i, 0, HEIGHT, 1) \
        { \
            dst[_i].v = V_LOAD(DATA_TYPE, WIDTH, TENSOR_TYPE, TENSOR, X, (indirect_y[_i].v), STRIDE_Y); \
        }                                                                                                   \
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
    ({ \
        LOOP_UNROLLING(int, _yk, 0, (TILE_HEIGHT), 1) \
        { \
            LOOP_UNROLLING(int, _xk, 0, (TILE_WIDTH), 1) \
            { \
                int _src_y = (X) + _xk + ((Y) + _yk) * (TENSOR_WIDTH); \
                _src_y    += (B) * (int)(TENSOR_WIDTH) * (int)(TENSOR_HEIGHT); \
                int _src_valid_y = (((X) + _xk) >= 0 && ((X) + _xk) < (int)(TENSOR_WIDTH) && ((Y) + _yk) >= 0 && ((Y) + _yk) < (int)(TENSOR_HEIGHT)); \
                if(_src_valid_y != 0) \
                { \
                    dst[_xk + _yk * (TILE_WIDTH)].v = V_LOAD(DATA_TYPE, TILE_CHANNELS, TENSOR_TYPE, TENSOR, C, _src_y, STRIDE_Y); \
                }                                                                                                                                     \
            }                                                                                                                                                 \
        }                                                                                                                                                 \
    })

/** Load a tile from global memory (tensor) when the tensor is stored using a NHWC layout using indirect X and Y coordinates
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
 * @param[out] xi            A tile with (TILE_WIDTH x TILE_HEIGHT) values with the indirect X coordinate
 * @param[out] yi            A tile with (TILE_WIDTH x TILE_HEIGHT) values with the indirect Y coordinate
 * @param[out] dst           Output tile
 */
#define T_LOAD_NHWC_INDIRECT(DATA_TYPE, TILE_HEIGHT, TILE_WIDTH, TILE_CHANNELS, TENSOR_TYPE, TENSOR, B, Y, X, C, TENSOR_WIDTH, TENSOR_HEIGHT, STRIDE_Y, xi, yi, dst)  \
    ({ \
        LOOP_UNROLLING(int, _i, 0, (TILE_WIDTH * TILE_HEIGHT), 1) \
        { \
            int _src_y = (X) + xi[_i].v + ((Y) + yi[_i].v) * (TENSOR_WIDTH); \
            _src_y    += (B) * (int)(TENSOR_WIDTH) * (int)(TENSOR_HEIGHT); \
            int _src_valid_y = (((X) + xi[_i].v) >= 0 && ((X) + xi[_i].v) < (int)(TENSOR_WIDTH) && ((Y) + yi[_i].v) >= 0 && ((Y) + yi[_i].v) < (int)(TENSOR_HEIGHT)); \
            if(_src_valid_y != 0) \
            { \
                dst[_i].v = V_LOAD(DATA_TYPE, TILE_CHANNELS, TENSOR_TYPE, TENSOR, C, _src_y, STRIDE_Y); \
            }                                                                                                                                                         \
        }                                                                                                                                                                 \
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
            LOOP_UNROLLING(int, _i, 0, HEIGHT, 1)                                                                                                                                                  \
            {                                                                                                                                                                                      \
                VSTORE_PARTIAL(WIDTH0, WIDTH1)                                                                                                                                                     \
                (src[HEIGHT - 1 - _i].v, 0, (__global DATA_TYPE *)(TENSOR##_ptr + TENSOR##_offset_first_element_in_bytes + (X) * sizeof(DATA_TYPE) + (indirect_y[HEIGHT - 1 - _i].v) * STRIDE_Y)); \
            }                                                                                                                                                                                      \
        }                                                                                                                                                                                          \
        else                                                                                                                                                                                       \
        {                                                                                                                                                                                          \
            LOOP_UNROLLING(int, _i, 0, HEIGHT, 1)                                                                                                                                                  \
            {                                                                                                                                                                                      \
                VSTORE(WIDTH0)                                                                                                                                                                     \
                (src[HEIGHT - 1 - _i].v, 0, (__global DATA_TYPE *)(TENSOR##_ptr + TENSOR##_offset_first_element_in_bytes + (X) * sizeof(DATA_TYPE) + (indirect_y[HEIGHT - 1 - _i].v) * STRIDE_Y)); \
            }                                                                                                                                                                                      \
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
#define T_OFFSET_CORRECTION(ACC_DATA_TYPE, M0, N0, K0, SRC_OFFSET, WEI_OFFSET, lhs, rhs, dst)                                                                                                                                                               \
    ({ \
        LOOP_UNROLLING(int, _m0, 0, M0, 1) \
        { \
            ACC_DATA_TYPE _tm = 0; \
            LOOP_UNROLLING(int, _k0, 0, K0, 1) \
            { \
                _tm += ((ACC_DATA_TYPE)lhs[_m0].s[_k0] * (ACC_DATA_TYPE)WEI_OFFSET); \
            } \
            LOOP_UNROLLING(int, _n0, 0, N0, 1) \
            { \
                dst[_m0].s[_n0] += _tm; \
                LOOP_UNROLLING(int, _k0, 0, K0, 1) \
                { \
                    dst[_m0].s[_n0] += ((ACC_DATA_TYPE)rhs[_n0].s[_k0] * (ACC_DATA_TYPE)SRC_OFFSET); \
                } \
            }                                                                                                                                                                                                                                               \
        }                                                                                                                                                                                                                                                       \
    })

/** Quantized the tile (ASYMMETRIC) with fixed-point scale
 *
 * @param[in]  SRC_DATA_TYPE  SRC data type
 * @param[in]  DST_DATA_TYPE  DST data type
 * @param[in]  M0             Number of src/dst rows
 * @param[in]  N0             Number of src/dst columns
 * @param[in]  DST_OFFSET     Quantization offset
 * @param[in]  DST_SHIFT      Quantization shift
 * @param[in]  DST_MULTIPLIER Quantization multiplier
 * @param[in]  src            Input tile
 * @param[out] dst            Output tile
 */
#define T_QUANTIZE8_ASYMMETRIC(SRC_DATA_TYPE, DST_DATA_TYPE, M0, N0, DST_OFFSET, DST_SHIFT, DST_MULTIPLIER, src, dst)                          \
    ({ \
        LOOP_UNROLLING(int, _m0, 0, M0, 1) \
        { \
            LOOP_UNROLLING(int, _n0, 0, N0, 1) \
            { \
                SRC_DATA_TYPE _tmp = 0; \
                if(DST_SHIFT < 0) \
                { \
                    _tmp = ASYMM_MULT_BY_QUANT_MULTIPLIER_GREATER_THAN_ONE(src[_m0].s[_n0], DST_MULTIPLIER, DST_SHIFT, 1); \
                } \
                else \
                { \
                    _tmp = ASYMM_MULT_BY_QUANT_MULTIPLIER_LESS_THAN_ONE(src[_m0].s[_n0], DST_MULTIPLIER, DST_SHIFT, 1); \
                } \
                _tmp += DST_OFFSET; \
                dst[_m0].s[_n0] = CONVERT_SAT(_tmp, DST_DATA_TYPE);                                                                            \
            }                                                                                                                                          \
        }                                                                                                                                          \
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
    ({ \
        LOOP_UNROLLING(int, _m0, 0, M0, 1) \
        { \
            LOOP_UNROLLING(int, _n0, 0, N0, 1) \
            { \
                a[_m0].s[_n0] = select((DATA_TYPE)(a[_m0].s[_n0]), (DATA_TYPE)(VALUE_TO_SET), (SELECT_DATA_TYPE(DATA_TYPE))(mask[_m0].v == (DATA_TYPE)0)); \
            }                                                                                                                                                      \
        }                                                                                                                                                      \
    })

/** Element-wise activation
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
    ({ \
        LOOP_UNROLLING(int, _m0, 0, M0, 1) \
        { \
            dst[_m0].v = ACTIVATION(ACTIVATION_TYPE, DATA_TYPE, N0, src[_m0].v, A_VAL, B_VAL); \
        }                                                                                          \
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
    ({ \
        LOOP_UNROLLING(int, _m0, 0, M0, 1) \
        { \
            LOOP_UNROLLING(int, _n0, 0, N0, 1) \
            { \
                dst[_m0].s[_n0] = lhs[_m0].s[_n0] + rhs_constant; \
            }                                                             \
        }                                                             \
    })

/** Element-wise addition with RHS broadcasted (RHS has the X dimension only)
 *
 * @note Performs: LHS + RHS[broadcasted] = DST
 * @note Both tiles must have same data type
 *
 * @param[in]  DATA_TYPE LHS/RHS/DST data type
 * @param[in]  M0        Number of LHS rows
 * @param[in]  N0        Number of LHS columns
 * @param[in]  lhs       LHS tile
 * @param[in]  rhs       RHS tile
 * @param[out] dst       DST tile
 */
#define T_ADD_BROADCAST_X(DATA_TYPE, M0, N0, lhs, rhs, dst) \
    ({ \
        LOOP_UNROLLING(int, _m0, 0, M0, 1) \
        { \
            dst[_m0].v = lhs[_m0].v + rhs[0].v;             \
        }                                                       \
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
 */
#define T_MMUL(LHS_DATA_TYPE, RHS_DATA_TYPE, DST_DATA_TYPE, M0, N0, K0, LHS_LAYOUT, RHS_LAYOUT, lhs, rhs, dst) T_MMUL_##LHS_LAYOUT##_##RHS_LAYOUT(LHS_DATA_TYPE, RHS_DATA_TYPE, DST_DATA_TYPE, M0, N0, K0, lhs, rhs, dst)
#define T_MMUL_NT_T(LHS_DATA_TYPE, RHS_DATA_TYPE, DST_DATA_TYPE, M0, N0, K0, lhs, rhs, dst) T_MMUL_NT_T_##LHS_DATA_TYPE##_##RHS_DATA_TYPE##_##DST_DATA_TYPE(DST_DATA_TYPE, M0, N0, K0, lhs, rhs, dst)
#define T_MMUL_NT_T_float_float_float(DST_DATA_TYPE, M0, N0, K0, lhs, rhs, dst) T_MMUL_NT_T_FLOAT(DST_DATA_TYPE, M0, N0, K0, lhs, rhs, dst)
#define T_MMUL_NT_T_half_half_half(DST_DATA_TYPE, M0, N0, K0, lhs, rhs, dst) T_MMUL_NT_T_FLOAT(DST_DATA_TYPE, M0, N0, K0, lhs, rhs, dst)
#define T_MMUL_NT_T_char_char_int(DST_DATA_TYPE, M0, N0, K0, lhs, rhs, dst) T_MMUL_NT_T_INTEGER8(DST_DATA_TYPE, M0, N0, K0, lhs, rhs, dst)
#define T_MMUL_NT_T_uchar_uchar_uint(DST_DATA_TYPE, M0, N0, K0, lhs, rhs, dst) T_MMUL_NT_T_INTEGER8(DST_DATA_TYPE, M0, N0, K0, lhs, rhs, dst)
#define T_MMUL_NT_T_uchar_uchar_int(DST_DATA_TYPE, M0, N0, K0, lhs, rhs, dst) T_MMUL_NT_T_INTEGER8(DST_DATA_TYPE, M0, N0, K0, lhs, rhs, dst)
#define T_MMUL_NT_T_FLOAT(DST_DATA_TYPE, M0, N0, K0, lhs, rhs, dst)                       \
    {                                                                                     \
        LOOP_UNROLLING(int, _m, 0, M0, 1)                                                 \
        {                                                                                 \
            LOOP_UNROLLING(int, _n, 0, N0, 1)                                             \
            {                                                                             \
                LOOP_UNROLLING(int, _k, 0, K0, 1)                                         \
                {                                                                         \
                    dst[_m].s[_n] = fma((lhs[_m].s[_k]), (rhs[_n].s[_k]), dst[_m].s[_n]); \
                }                                                                         \
            }                                                                             \
        }                                                                                 \
    }
#define T_MMUL_NT_T_INTEGER8(DST_DATA_TYPE, M0, N0, K0, lhs, rhs, dst)                            \
    ({ \
        LOOP_UNROLLING(int, _m, 0, M0, 1) \
        { \
            LOOP_UNROLLING(int, _n, 0, N0, 1) \
            { \
                DOT_PRODUCT_INTEGER8(DST_DATA_TYPE, K0, (lhs[_m].v), (rhs[_n].v), dst[_m].s[_n]); \
            }                                                                                             \
        }                                                                                             \
    })