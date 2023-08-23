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
#ifndef ACL_ARM_COMPUTE_CORE_CORETYPES
#define ACL_ARM_COMPUTE_CORE_CORETYPES

#include "arm_compute/core/Strides.h"
#include "support/Half.h"

/** CoreTypes.h groups together essential small types that are used across functions */

namespace arm_compute
{
/** 16-bit floating point type */
using half = half_float::half;
/** Permutation vector */
using PermutationVector = Strides;

/** Available channels */
enum class Channel
{
    UNKNOWN, /** Unknown channel format */
    C0,      /**< First channel (used by formats with unknown channel types). */
    C1,      /**< Second channel (used by formats with unknown channel types). */
    C2,      /**< Third channel (used by formats with unknown channel types). */
    C3,      /**< Fourth channel (used by formats with unknown channel types). */
    R,       /**< Red channel. */
    G,       /**< Green channel. */
    B,       /**< Blue channel. */
    A,       /**< Alpha channel. */
    Y,       /**< Luma channel. */
    U,       /**< Cb/U channel. */
    V        /**< Cr/V/Value channel. */
};

/** Image colour formats */
enum class Format
{
    UNKNOWN,  /**< Unknown image format */
    U8,       /**< 1 channel, 1 U8 per channel */
    S16,      /**< 1 channel, 1 S16 per channel */
    U16,      /**< 1 channel, 1 U16 per channel */
    S32,      /**< 1 channel, 1 S32 per channel */
    U32,      /**< 1 channel, 1 U32 per channel */
    S64,      /**< 1 channel, 1 S64 per channel */
    U64,      /**< 1 channel, 1 U64 per channel */
    BFLOAT16, /**< 16-bit brain floating-point number */
    F16,      /**< 1 channel, 1 F16 per channel */
    F32,      /**< 1 channel, 1 F32 per channel */
    UV88,     /**< 2 channel, 1 U8 per channel */
    RGB888,   /**< 3 channels, 1 U8 per channel */
    RGBA8888, /**< 4 channels, 1 U8 per channel */
    YUV444,   /**< A 3 plane of 8 bit 4:4:4 sampled Y, U, V planes */
    YUYV422,  /**< A single plane of 32-bit macro pixel of Y0, U0, Y1, V0 bytes */
    NV12,     /**< A 2 plane YUV format of Luma (Y) and interleaved UV data at 4:2:0 sampling */
    NV21,     /**< A 2 plane YUV format of Luma (Y) and interleaved VU data at 4:2:0 sampling */
    IYUV,     /**< A 3 plane of 8-bit 4:2:0 sampled Y, U, V planes */
    UYVY422   /**< A single plane of 32-bit macro pixel of U0, Y0, V0, Y1 byte */
};

/** Available data types */
enum class DataType
{
    UNKNOWN,            /**< Unknown data type */
    U8,                 /**< unsigned 8-bit number */
    S8,                 /**< signed 8-bit number */
    QSYMM8,             /**< quantized, symmetric fixed-point 8-bit number */
    QASYMM8,            /**< quantized, asymmetric fixed-point 8-bit number unsigned */
    QASYMM8_SIGNED,     /**< quantized, asymmetric fixed-point 8-bit number signed */
    QSYMM8_PER_CHANNEL, /**< quantized, symmetric per channel fixed-point 8-bit number */
    U16,                /**< unsigned 16-bit number */
    S16,                /**< signed 16-bit number */
    QSYMM16,            /**< quantized, symmetric fixed-point 16-bit number */
    QASYMM16,           /**< quantized, asymmetric fixed-point 16-bit number */
    U32,                /**< unsigned 32-bit number */
    S32,                /**< signed 32-bit number */
    U64,                /**< unsigned 64-bit number */
    S64,                /**< signed 64-bit number */
    BFLOAT16,           /**< 16-bit brain floating-point number */
    F16,                /**< 16-bit floating-point number */
    F32,                /**< 32-bit floating-point number */
    F64,                /**< 64-bit floating-point number */
    SIZET               /**< size_t */
};

/** [DataLayout enum definition] **/

/** Supported tensor data layouts */
enum class DataLayout
{
    UNKNOWN, /**< Unknown data layout */
    NCHW,    /**< Num samples, channels, height, width */
    NHWC,    /**< Num samples, height, width, channels */
    NCDHW,   /**< Num samples, channels, depth, height, width */
    NDHWC    /**< Num samples, depth, height, width, channels */
};
/** [DataLayout enum definition] **/

/** Supported tensor data layout dimensions */
enum class DataLayoutDimension
{
    CHANNEL, /**< channel */
    HEIGHT,  /**< height */
    WIDTH,   /**< width */
    DEPTH,   /**< depth */
    BATCHES  /**< batches */
};

/** Dimension rounding type when down-scaling on CNNs
 * @note Used in pooling and convolution layer
 */
enum class DimensionRoundingType
{
    FLOOR, /**< Floor rounding */
    CEIL   /**< Ceil rounding */
};

class PadStrideInfo
{
public:
    /** Constructor
     *
     * @param[in] stride_x (Optional) Stride, in elements, across x. Defaults to 1.
     * @param[in] stride_y (Optional) Stride, in elements, across y. Defaults to 1.
     * @param[in] pad_x    (Optional) Padding, in elements, across x. Defaults to 0.
     * @param[in] pad_y    (Optional) Padding, in elements, across y. Defaults to 0.
     * @param[in] round    (Optional) Dimensions rounding. Defaults to @ref DimensionRoundingType::FLOOR.
     */
    PadStrideInfo(unsigned int stride_x = 1, unsigned int stride_y = 1,
                  unsigned int pad_x = 0, unsigned int pad_y = 0,
                  DimensionRoundingType round = DimensionRoundingType::FLOOR)
        : _stride(std::make_pair(stride_x, stride_y)),
          _pad_left(pad_x),
          _pad_top(pad_y),
          _pad_right(pad_x),
          _pad_bottom(pad_y),
          _round_type(round)
    {
    }
    /** Constructor
     *
     * @param[in] stride_x   Stride, in elements, across x.
     * @param[in] stride_y   Stride, in elements, across y.
     * @param[in] pad_left   Padding across x on the left, in elements.
     * @param[in] pad_right  Padding across x on the right, in elements.
     * @param[in] pad_top    Padding across y on the top, in elements.
     * @param[in] pad_bottom Padding across y on the bottom, in elements.
     * @param[in] round      Dimensions rounding.
     */
    PadStrideInfo(unsigned int stride_x, unsigned int stride_y,
                  unsigned int pad_left, unsigned int pad_right,
                  unsigned int pad_top, unsigned int pad_bottom,
                  DimensionRoundingType round)
        : _stride(std::make_pair(stride_x, stride_y)),
          _pad_left(pad_left),
          _pad_top(pad_top),
          _pad_right(pad_right),
          _pad_bottom(pad_bottom),
          _round_type(round)
    {
    }
    /** Get the stride.
     *
     * @return a pair: stride x, stride y.
     */
    std::pair<unsigned int, unsigned int> stride() const
    {
        return _stride;
    }
    /** Check whether the padding is symmetric.
     *
     * @return True if the padding is symmetric.
     */
    bool padding_is_symmetric() const
    {
        return (_pad_left == _pad_right) && (_pad_top == _pad_bottom);
    }
    /** Get the padding.
     *
     * @note This should only be used when the padding is symmetric.
     *
     * @return a pair: padding left/right, padding top/bottom
     */
    std::pair<unsigned int, unsigned int> pad() const
    {
        //this accessor should be used only when padding is symmetric
        ARM_COMPUTE_ERROR_ON(!padding_is_symmetric());
        return std::make_pair(_pad_left, _pad_top);
    }

    /** Get the left padding */
    unsigned int pad_left() const
    {
        return _pad_left;
    }
    /** Get the right padding */
    unsigned int pad_right() const
    {
        return _pad_right;
    }
    /** Get the top padding */
    unsigned int pad_top() const
    {
        return _pad_top;
    }
    /** Get the bottom padding */
    unsigned int pad_bottom() const
    {
        return _pad_bottom;
    }

    /** Get the rounding type */
    DimensionRoundingType round() const
    {
        return _round_type;
    }

    /** Check whether this has any padding */
    bool has_padding() const
    {
        return (_pad_left != 0 || _pad_top != 0 || _pad_right != 0 || _pad_bottom != 0);
    }

private:
    std::pair<unsigned int, unsigned int> _stride;
    unsigned int _pad_left;
    unsigned int _pad_top;
    unsigned int _pad_right;
    unsigned int _pad_bottom;

    DimensionRoundingType _round_type;
};

/** Memory layouts for the weights tensor.
 *
 * * UNSPECIFIED is used to select kernels that do not run in
 *    variable weights mode.
 *
 * * ANY is used to query the kernel database to retrieve any of the
 *   kernels that runs in variable weights mode. Once a kernel is
 *   found, the specific format expected by the kernel can be
 *   retrieved by the user for reordering the weights tensor
 *   accordingly.
 *
 * The other values OHWIo{interleave_by}i{block_by} describe the
 * memory layout of a 4D tensor with layout OHWI that has been
 * transformed into a 4D tensor with dimensions O'HWI' where:
 *
 * O' = first multiple of {interleave_by} s.t. O<=O'
 * I' = first multiple of {block_by} s.t. I<=I'
 *
 * The total size of the dst tensor is O' x H x W x I'
 *
 * The access function of the tensor with layout
 * OHWIo{interleave_by}i{block_by} and size O'HWI' is a 6-parameter
 * access function, where the 6 parameters are computed as follows:
 *
 * x5 = floor(o/{interleave_by}) RANGE [0, O'/{interleave_by} -1] SIZE: O'/{interleave_by}
 *
 * x4 = h                        RANGE [0, H-1]                   SIZE: H
 * x3 = w                        RANGE [0, W-1]                   SIZE: W
 * x2 = floor(i/{block_by})      RANGE [0, I'/{block_by} -1]      SIZE: I'/{block_by}
 * x1 = o%{interleave_by}        RANGE [0, {interleave_by} -1]    SIZE: {interleave_by}
 * x0 = i%{block_by}             RANGE [0, {block_by} -1]         SIZE: {block_by}
 *                                                          TOTAL SIZE: O' * H * W * I'
 *
 *        4D                       6D
 * -----------------   -----------------------------------
 * value(o, h, w, i) =   x5 * H * W * I' * {interleave_by}
 *                     + x4 * W * I' * {interleave_by}
 *                     + x3 * I' * {interleave_by}
 *                     + x2 * {interleave_by} * {block_by}
 *                     + x1 * {block_by}
 *                     + x0
 *
 * Notice that in arm_gemm the 4D tensor of dimension O'HWI' created
 * for the OHWIo{interleave_by}i{block_by} format is in reality seen
 * as a 2D tensor, where the number of rows is O'/{interleave_by}
 * and the number of columns is {interleave_by} * H * W * I'.
 *
 * The postfix *_bf16 is for the memory layout needed for the
 * fast-mode kernels, in which the weights are passed in bfloat16
 * format.
 */
enum class WeightFormat
{
    UNSPECIFIED    = 0x1,
    ANY            = 0x2,
    OHWI           = 0x100100,
    OHWIo2         = 0x100200,
    OHWIo4         = 0x100400,
    OHWIo8         = 0x100800,
    OHWIo16        = 0x101000,
    OHWIo32        = 0x102000,
    OHWIo64        = 0x104000,
    OHWIo128       = 0x108000,
    OHWIo4i2       = 0x200400,
    OHWIo4i2_bf16  = 0x200410,
    OHWIo8i2       = 0x200800,
    OHWIo8i2_bf16  = 0x200810,
    OHWIo16i2      = 0x201000,
    OHWIo16i2_bf16 = 0x201010,
    OHWIo32i2      = 0x202000,
    OHWIo32i2_bf16 = 0x202010,
    OHWIo64i2      = 0x204000,
    OHWIo64i2_bf16 = 0x204010,
    OHWIo4i4       = 0x400400,
    OHWIo4i4_bf16  = 0x400410,
    OHWIo8i4       = 0x400800,
    OHWIo8i4_bf16  = 0x400810,
    OHWIo16i4      = 0x401000,
    OHWIo16i4_bf16 = 0x401010,
    OHWIo32i4      = 0x402000,
    OHWIo32i4_bf16 = 0x402010,
    OHWIo64i4      = 0x404000,
    OHWIo64i4_bf16 = 0x404010,
    OHWIo2i8       = 0x800200,
    OHWIo4i8       = 0x800400,
    OHWIo8i8       = 0x800800,
    OHWIo16i8      = 0x801000,
    OHWIo32i8      = 0x802000,
    OHWIo64i8      = 0x804000
};

} // namespace arm_compute
#endif /* ACL_ARM_COMPUTE_CORE_CORETYPES */
