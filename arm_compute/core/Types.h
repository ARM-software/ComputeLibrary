/*
 * Copyright (c) 2016-2018 ARM Limited.
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
#ifndef __ARM_COMPUTE_TYPES_H__
#define __ARM_COMPUTE_TYPES_H__

#include "arm_compute/core/Coordinates.h"
#include "arm_compute/core/QAsymm8.h"
#include "arm_compute/core/Rounding.h"
#include "arm_compute/core/Size2D.h"
#include "arm_compute/core/Strides.h"
#include "arm_compute/core/TensorShape.h"
#include "support/Half.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

namespace arm_compute
{
/** 16-bit floating point type */
using half = half_float::half;

/** Permutation vector */
using PermutationVector = Strides;
/** Bidirectional strides */
using BiStrides = Coordinates;

/** Image colour formats */
enum class Format
{
    UNKNOWN,  /**< Unknown image format */
    U8,       /**< 1 channel, 1 U8 per channel */
    S16,      /**< 1 channel, 1 S16 per channel */
    U16,      /**< 1 channel, 1 U16 per channel */
    S32,      /**< 1 channel, 1 S32 per channel */
    U32,      /**< 1 channel, 1 U32 per channel */
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
    UNKNOWN, /**< Unknown data type */
    U8,      /**< unsigned 8-bit number */
    S8,      /**< signed 8-bit number */
    QASYMM8, /**< quantized, asymmetric fixed-point 8-bit number */
    U16,     /**< unsigned 16-bit number */
    S16,     /**< signed 16-bit number */
    U32,     /**< unsigned 32-bit number */
    S32,     /**< signed 32-bit number */
    U64,     /**< unsigned 64-bit number */
    S64,     /**< signed 64-bit number */
    F16,     /**< 16-bit floating-point number */
    F32,     /**< 32-bit floating-point number */
    F64,     /**< 64-bit floating-point number */
    SIZET    /**< size_t */
};

/** Available Sampling Policies */
enum class SamplingPolicy
{
    CENTER,  /**< Samples are taken at pixel center */
    TOP_LEFT /**< Samples are taken at pixel top left corner */
};

/** Constant value of the border pixels when using BorderMode::CONSTANT */
constexpr uint8_t CONSTANT_BORDER_VALUE = 199;

/** Constant value used to indicate a half-scale pyramid */
constexpr float SCALE_PYRAMID_HALF = 0.5f;

/** Constant value used to indicate a ORB scaled pyramid */
constexpr float SCALE_PYRAMID_ORB = 8.408964152537146130583778358414e-01;

/** Supported tensor data layouts */
enum class DataLayout
{
    UNKNOWN, /**< Unknown data layout */
    NCHW,    /**< Num samples, channels, height, width */
    NHWC     /**< Num samples, height, width, channels */
};

/** Supported tensor data layout dimensions */
enum class DataLayoutDimension
{
    CHANNEL, /**< channel */
    HEIGHT,  /**< height */
    WIDTH,   /**< width */
    BATCHES  /**< batches */
};

/** Quantization settings (used for QASYMM8 data type) */
struct QuantizationInfo
{
    /** Default constructor */
    QuantizationInfo() noexcept
        : scale(0.0f),
          offset(0)
    {
    }

    /** Construct quantization info.
     *
     * @param[in] scale  Scale.
     * @param[in] offset Offset.
     */
    QuantizationInfo(float scale, int offset)
        : scale(scale), offset(offset)
    {
    }

    /** Check whether equal to a given quantization info.
     *
     * @param[in] other Other quantization info.
     *
     * @return True if the given quantization info is the same.
     */
    bool operator==(const QuantizationInfo &other) const
    {
        return scale == other.scale && offset == other.offset;
    }

    /** Check whether not equal to a given quantization info.
     *
     * @param[in] other Other quantization info.
     *
     * @return True if the given quantization info is not the same.
     */
    bool operator!=(const QuantizationInfo &other) const
    {
        return !(*this == other);
    }

    float scale;  /**< scale */
    int   offset; /**< offset */

    /** Quantizes a value using the scale/offset in this QuantizationInfo
     *
     * @param[in] value           Value to quantize.
     * @param[in] rounding_policy Policy to use when rounding.
     *
     * @return the quantized value.
     */
    qasymm8_t quantize(float value, RoundingPolicy rounding_policy) const
    {
        ARM_COMPUTE_ERROR_ON_MSG(scale == 0, "QuantizationInfo::quantize: scale == 0");
        return sqcvt_qasymm8_f32(value, scale, offset, rounding_policy);
    }

    /** Dequantizes a value using the scale/offset in this QuantizationInfo
     *
     * @param[in] value Value to dequantize.
     *
     * @return the original value before quantization.
     */
    float dequantize(qasymm8_t value) const
    {
        ARM_COMPUTE_ERROR_ON_MSG(scale == 0, "QuantizationInfo::dequantize: scale == 0");
        return scvt_f32_qasymm8(value, scale, offset);
    }

    /** Indicates whether this QuantizationInfo has valid settings or not
     *
     * @return True if the this has invalid settings.
     */
    bool empty() const
    {
        return scale == 0;
    }
};

/** Container for valid region of a window */
struct ValidRegion
{
    /** Default constructor */
    ValidRegion()
        : anchor{}, shape{}
    {
    }

    /** Allow instances of this class to be copy constructed */
    ValidRegion(const ValidRegion &) = default;
    /** Allow instances of this class to be move constructed */
    ValidRegion(ValidRegion &&) = default;
    /** Allow instances of this class to be copied */
    ValidRegion &operator=(const ValidRegion &) = default;
    /** Allow instances of this class to be moved */
    ValidRegion &operator=(ValidRegion &&) = default;
    /** Default destructor */
    ~ValidRegion() = default;

    /** Constructor for a valid region with default number of dimensions
     *
     * @param[in] an_anchor Anchor for the start of the valid region.
     * @param[in] a_shape   Shape of the valid region.
     *
     */
    ValidRegion(const Coordinates &an_anchor, const TensorShape &a_shape)
        : anchor{ an_anchor }, shape{ a_shape }
    {
        anchor.set_num_dimensions(std::max(anchor.num_dimensions(), shape.num_dimensions()));
    }

    /** Constructor for a valid region with specified number of dimensions
     *
     * @param[in] an_anchor      Anchor for the start of the valid region.
     * @param[in] a_shape        Shape of the valid region.
     * @param[in] num_dimensions Number of dimensions (must be >= number of dimensions of anchor and shape).
     *
     */
    ValidRegion(const Coordinates &an_anchor, const TensorShape &a_shape, size_t num_dimensions)
        : anchor{ an_anchor }, shape{ a_shape }
    {
        ARM_COMPUTE_ERROR_ON(num_dimensions < std::max(anchor.num_dimensions(), shape.num_dimensions()));
        anchor.set_num_dimensions(num_dimensions);
    }

    /** Return the start of the valid region for the given dimension @p d */
    int start(unsigned int d) const
    {
        return anchor[d];
    }

    /** Return the end of the valid region for the given dimension @p d */
    int end(unsigned int d) const
    {
        return anchor[d] + shape[d];
    }

    /** Accessor to set the value of anchor and shape for one of the dimensions.
     *
     * @param[in] dimension Dimension for which the value is set.
     * @param[in] start     Value to be set in anchor for the dimension.
     * @param[in] size      Value to be set in shape for the dimension.
     *
     * @return *this.
     */
    ValidRegion &set(size_t dimension, int start, size_t size)
    {
        anchor.set(dimension, start);
        shape.set(dimension, size);
        return *this;
    }

    Coordinates anchor; /**< Anchor for the start of the valid region. */
    TensorShape shape;  /**< Shape of the valid region. */
};

/** Methods available to handle borders */
enum class BorderMode
{
    UNDEFINED, /**< Borders are left undefined */
    CONSTANT,  /**< Pixels outside the image are assumed to have a constant value */
    REPLICATE  /**< Pixels outside the image are assumed to have the same value as the closest image pixel */
};

/** Container for 2D border size */
struct BorderSize
{
    /** Empty border, i.e. no border */
    constexpr BorderSize()
        : top{ 0 }, right{ 0 }, bottom{ 0 }, left{ 0 }
    {
    }

    /** Border with equal size around the 2D plane */
    explicit constexpr BorderSize(unsigned int size)
        : top{ size }, right{ size }, bottom{ size }, left{ size }
    {
    }

    /** Border with same size for top/bottom and left/right */
    constexpr BorderSize(unsigned int top_bottom, unsigned int left_right)
        : top{ top_bottom }, right{ left_right }, bottom{ top_bottom }, left{ left_right }
    {
    }

    /** Border with different sizes */
    constexpr BorderSize(unsigned int top, unsigned int right, unsigned int bottom, unsigned int left)
        : top{ top }, right{ right }, bottom{ bottom }, left{ left }
    {
    }

    /** Check if the entire border is zero */
    constexpr bool empty() const
    {
        return top == 0 && right == 0 && bottom == 0 && left == 0;
    }

    /** Check if the border is the same size on all sides */
    constexpr bool uniform() const
    {
        return top == right && top == bottom && top == left;
    }

    /** Scale this border size.
     *
     * @param[in] scale Scale to multiply border size by.
     *
     * @return *this.
     */
    BorderSize &operator*=(float scale)
    {
        top *= scale;
        right *= scale;
        bottom *= scale;
        left *= scale;

        return *this;
    }

    /** Scale a copy of this border size.
     *
     * @param[in] scale Scale to multiply border size by.
     *
     * @return a scaled copy of this.
     */
    BorderSize operator*(float scale)
    {
        BorderSize size = *this;
        size *= scale;

        return size;
    }

    /** Limit this border size.
     *
     * @param[in] limit Border size to limit this border size to.
     */
    void limit(const BorderSize &limit)
    {
        top    = std::min(top, limit.top);
        right  = std::min(right, limit.right);
        bottom = std::min(bottom, limit.bottom);
        left   = std::min(left, limit.left);
    }

    unsigned int top;    /**< top of the border */
    unsigned int right;  /**< right of the border */
    unsigned int bottom; /**< bottom of the border */
    unsigned int left;   /**< left of the border */
};

/** Container for 2D padding size */
using PaddingSize = BorderSize;

/** Policy to handle overflow */
enum class ConvertPolicy
{
    WRAP,    /**< Wrap around */
    SATURATE /**< Saturate */
};

/** Interpolation method */
enum class InterpolationPolicy
{
    NEAREST_NEIGHBOR, /**< Output values are defined to match the source pixel whose center is nearest to the sample position */
    BILINEAR,         /**< Output values are defined by bilinear interpolation between the pixels */
    AREA,             /**< Output values are determined by averaging the source pixels whose areas fall under the area of the destination pixel, projected onto the source image */
};

/** Bilinear Interpolation method used by LKTracker */
enum class BilinearInterpolation
{
    BILINEAR_OLD_NEW, /**< Old-new method */
    BILINEAR_SCHARR   /**< Scharr method */
};

/** Threshold mode */
enum class ThresholdType
{
    BINARY, /**< Threshold with one value */
    RANGE   /**< Threshold with two values*/
};

/** Termination criteria */
enum class Termination
{
    TERM_CRITERIA_EPSILON,    /**< Terminate when within epsilon of a threshold */
    TERM_CRITERIA_ITERATIONS, /**< Terminate after a maximum number of iterations */
    TERM_CRITERIA_BOTH        /**< Terminate on whichever of the other conditions occurs first */
};

/** Magnitude calculation type. */
enum class MagnitudeType
{
    L1NORM, /**< L1 normalization type */
    L2NORM  /**< L2 normalization type */
};

/** Phase calculation type.
 *
 * @note When PhaseType == SIGNED, each angle is mapped to the range 0 to 255 inclusive otherwise angles between 0 and 180
 */
enum class PhaseType
{
    SIGNED,  /**< Angle range: [0, 360] */
    UNSIGNED /**< Angle range: [0, 180] */
};

/** Keypoint type */
struct KeyPoint
{
    int32_t x{ 0 };               /**< X coordinates */
    int32_t y{ 0 };               /**< Y coordinates */
    float   strength{ 0.f };      /**< Strength of the point */
    float   scale{ 0.f };         /**< Scale initialized to 0 by the corner detector */
    float   orientation{ 0.f };   /**< Orientation initialized to 0 by the corner detector */
    int32_t tracking_status{ 0 }; /**< Status initialized to 1 by the corner detector, set to 0 when the point is lost */
    float   error{ 0.f };         /**< Tracking error initialized to 0 by the corner detector */
};

/** Internal key point */
using InternalKeypoint = std::tuple<float, float, float>; /* x,y,strength */

/** Rectangle type */
struct Rectangle
{
    uint16_t x;      /**< Top-left x coordinate */
    uint16_t y;      /**< Top-left y coordinate */
    uint16_t width;  /**< Width of the rectangle */
    uint16_t height; /**< Height of the rectangle */
};

/** Coordinate type */
struct Coordinates2D
{
    int32_t x; /**< X coordinates */
    int32_t y; /**< Y coordinates */
};

/** Coordinate type */
struct Coordinates3D
{
    uint32_t x; /**< X coordinates */
    uint32_t y; /**< Y coordinates */
    uint32_t z; /**< Z coordinates */
};

/** Padding information as a pair of unsigned int start/end */
using PaddingInfo = std::pair<uint32_t, uint32_t>;

/** List of padding information */
using PaddingList = std::vector<PaddingInfo>;

/** Region of interest */
struct ROI
{
    Rectangle rect;      /**< Rectangle specifying the region of interest */
    uint16_t  batch_idx; /**< The batch index of the region of interest */
};

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

/** Available matrix patterns */
enum class MatrixPattern
{
    BOX,   /**< Box pattern matrix. */
    CROSS, /**< Cross pattern matrix. */
    DISK,  /**< Disk pattern matrix. */
    OTHER  /**< Any other matrix pattern. */
};

/** Available non linear functions. */
enum class NonLinearFilterFunction : unsigned
{
    MEDIAN = 0, /**< Non linear median filter. */
    MIN    = 1, /**< Non linear erode. */
    MAX    = 2, /**< Non linear dilate. */
};

/** Available reduction operations */
enum class ReductionOperation
{
    SUM_SQUARE, /**< Sum of squares */
    SUM,        /**< Sum */
    MEAN_SUM,   /**< Mean of sum */
};

/** The normalization type used for the normalization layer */
enum class NormType
{
    IN_MAP_1D, /**< Normalization applied within the same map in 1D region */
    IN_MAP_2D, /**< Normalization applied within the same map in 2D region */
    CROSS_MAP  /**< Normalization applied cross maps */
};

/** Normalization type for Histogram of Oriented Gradients (HOG) */
enum class HOGNormType
{
    L2_NORM    = 1, /**< L2-norm */
    L2HYS_NORM = 2, /**< L2-norm followed by clipping */
    L1_NORM    = 3  /**< L1 norm */
};

/** Detection window used for the object detection. The detection window keeps the following information:
 *
 *  -# Geometry of the rectangular window (x/y of top-left corner and width/height)
 *  -# Index of the class used for evaluating which class the detection window belongs to
 *  -# Confidence value (score) obtained with the classifier
 */
struct DetectionWindow
{
    uint16_t x{ 0 };         /**< Top-left x coordinate */
    uint16_t y{ 0 };         /**< Top-left y coordinate */
    uint16_t width{ 0 };     /**< Width of the detection window */
    uint16_t height{ 0 };    /**< Height of the detection window */
    uint16_t idx_class{ 0 }; /**< Index of the class */
    float    score{ 0.f };   /**< Confidence value for the detection window */
};

/** Dimension rounding type when down-scaling on CNNs
 * @note Used in pooling and convolution layer
 */
enum class DimensionRoundingType
{
    FLOOR, /**< Floor rounding */
    CEIL   /**< Ceil rounding */
};

/** Available pooling types */
enum class PoolingType
{
    MAX, /**< Max Pooling */
    AVG, /**< Average Pooling */
    L2   /**< L2 Pooling */
};

/** Padding and stride information class */
class PadStrideInfo
{
public:
    /** Constructor
     *
     * @param[in] stride_x (Optional) Stride, in elements, across x. Defaults to 1.
     * @param[in] stride_y (Optional) Stride, in elements, across y. Defaults to 1.
     * @param[in] pad_x    (Optional) Padding, in elements, across x. Defaults to 0.
     * @param[in] pad_y    (Optional) Padding, in elements, across y. Defaults to 0.
     * @param[in] round    (Optional) Dimensions rounding. Defaults to @ref FLOOR.
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
     * @param[in] pad_top    Padding across y on the top, in elements.
     * @param[in] pad_right  Padding across x on the right, in elements.
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

/** Fully connected layer info */
struct FullyConnectedLayerInfo
{
    DataLayout weights_trained_layout{ DataLayout::NCHW }; /**<  Layout that the weights have been trained with. */
    bool       transpose_weights{ true };                  /**<  Transpose weights if true. */
    bool       are_weights_reshaped{ false };              /**<  Reshape the weights tensor if false. */
    bool       retain_internal_weights{ false };           /**<  Retain internal reshaped weights. */

    /** Sets the weights trained data layout
     *
     * @param[in] layout Data layout that the weights were trained with
     *
     * @return Updated object
     */
    FullyConnectedLayerInfo &set_weights_trained_layout(DataLayout layout)
    {
        weights_trained_layout = layout;
        return *this;
    }
    /** Sets the transpose weights flag
     *
     * @param[in] should_transpose_weights Boolean flag indicating if weights should be transposed
     *
     * @return Updated object
     */
    FullyConnectedLayerInfo &set_transpose_weights(bool should_transpose_weights)
    {
        transpose_weights = should_transpose_weights;
        return *this;
    }
};

/** Pooling Layer Information class */
class PoolingLayerInfo
{
public:
    /** Default Constructor */
    PoolingLayerInfo()
        : _pool_type(PoolingType::MAX), _pool_size(Size2D()), _pad_stride_info(PadStrideInfo()), _exclude_padding(false), _is_global_pooling(false)
    {
    }
    /** Default Constructor
     *
     * @param[in] pool_type       Pooling type @ref PoolingType.
     * @param[in] pool_size       Pooling size, in elements, across  x and y.
     * @param[in] pad_stride_info (Optional) Padding and stride information @ref PadStrideInfo
     * @param[in] exclude_padding (Optional) Strategy when accounting padding in calculations.
     *                             True will exclude padding while false will not (Used in AVG/L2 pooling to determine the pooling area).
     *                             Defaults to false;
     */
    explicit PoolingLayerInfo(PoolingType   pool_type,
                              unsigned int  pool_size,
                              PadStrideInfo pad_stride_info = PadStrideInfo(),
                              bool          exclude_padding = false)
        : _pool_type(pool_type), _pool_size(Size2D(pool_size, pool_size)), _pad_stride_info(pad_stride_info), _exclude_padding(exclude_padding), _is_global_pooling(false)
    {
    }
    /** Default Constructor
     *
     * @param[in] pool_type       Pooling type @ref PoolingType.
     * @param[in] pool_size       Pooling size, in elements, across  x and y.
     * @param[in] pad_stride_info (Optional) Padding and stride information @ref PadStrideInfo
     * @param[in] exclude_padding (Optional) Strategy when accounting padding in calculations.
     *                             True will exclude padding while false will not (Used in AVG/L2 pooling to determine the pooling area).
     *                             Defaults to false;
     */
    explicit PoolingLayerInfo(PoolingType   pool_type,
                              Size2D        pool_size,
                              PadStrideInfo pad_stride_info = PadStrideInfo(),
                              bool          exclude_padding = false)
        : _pool_type(pool_type), _pool_size(pool_size), _pad_stride_info(pad_stride_info), _exclude_padding(exclude_padding), _is_global_pooling(false)
    {
    }
    /** Default Constructor
     *
     * @note This constructor is used for global pooling
     *
     * @param[in] pool_type Pooling type @ref PoolingType.
     */
    explicit PoolingLayerInfo(PoolingType pool_type)
        : _pool_type(pool_type), _pool_size(Size2D()), _pad_stride_info(PadStrideInfo(1, 1, 0, 0)), _exclude_padding(false), _is_global_pooling(true)
    {
    }
    /** Get the pooling type */
    PoolingType pool_type() const
    {
        return _pool_type;
    }
    /** Get the pooling size */
    const Size2D &pool_size() const
    {
        return _pool_size;
    }
    /** Get the padding and stride */
    PadStrideInfo pad_stride_info() const
    {
        return _pad_stride_info;
    }
    /** Check if padding is excluded in calculations */
    bool exclude_padding() const
    {
        return _exclude_padding;
    }
    /** Check if is global pooling */
    bool is_global_pooling() const
    {
        return _is_global_pooling;
    }

private:
    PoolingType   _pool_type;
    Size2D        _pool_size;
    PadStrideInfo _pad_stride_info;
    bool          _exclude_padding;
    bool          _is_global_pooling;
};

/** ROI Pooling Layer Information class */
class ROIPoolingLayerInfo final
{
public:
    /** Constructor
     *
     * @param[in] pooled_width   Pooled width of the layer.
     * @param[in] pooled_height  Pooled height of the layer.
     * @param[in] spatial_scale  Spatial scale to be applied to the ROI coordinates and dimensions.
     * @param[in] sampling_ratio Number of samples to include in each pooling region (if set to zero, a ceil(roi_dims/pooling_dims))
     */
    ROIPoolingLayerInfo(unsigned int pooled_width, unsigned int pooled_height, float spatial_scale, unsigned int sampling_ratio = 0)
        : _pooled_width(pooled_width), _pooled_height(pooled_height), _spatial_scale(spatial_scale), _sampling_ratio(sampling_ratio)
    {
    }
    /** Get the pooled width of the layer */
    unsigned int pooled_width() const
    {
        return _pooled_width;
    }
    /** Get the pooled height of the layer */
    unsigned int pooled_height() const
    {
        return _pooled_height;
    }
    /** Get the spatial scale */
    float spatial_scale() const
    {
        return _spatial_scale;
    }
    /** Get sampling ratio */
    unsigned int sampling_ratio() const
    {
        return _sampling_ratio;
    }

private:
    unsigned int _pooled_width;
    unsigned int _pooled_height;
    float        _spatial_scale;
    unsigned int _sampling_ratio;
};

/** Activation Layer Information class */
class ActivationLayerInfo
{
public:
    /** Available activation functions */
    enum class ActivationFunction
    {
        LOGISTIC,        /**< Logistic ( \f$ f(x) = \frac{1}{1 + e^{-x}} \f$ ) */
        TANH,            /**< Hyperbolic tangent ( \f$ f(x) = a \cdot tanh(b \cdot x) \f$ ) */
        RELU,            /**< Rectifier ( \f$ f(x) = max(0,x) \f$ ) */
        BOUNDED_RELU,    /**< Upper Bounded Rectifier ( \f$ f(x) = min(a, max(0,x)) \f$ ) */
        LU_BOUNDED_RELU, /**< Lower and Upper Bounded Rectifier ( \f$ f(x) = min(a, max(b,x)) \f$ ) */
        LEAKY_RELU,      /**< Leaky Rectifier ( \f$ f(x)= log(1+e^x) \f$ ) */
        SOFT_RELU,       /**< Soft Rectifier ( \f$ f(x)= log(1+e^x) \f$ ) */
        ABS,             /**< Absolute ( \f$ f(x)= |x| \f$ ) */
        SQUARE,          /**< Square ( \f$ f(x)= x^2 \f$ )*/
        SQRT,            /**< Square root ( \f$ f(x) = \sqrt{x} \f$ )*/
        LINEAR           /**< Linear ( \f$ f(x)= ax + b \f$ ) */
    };

    ActivationLayerInfo() = default;
    /** Default Constructor
     *
     * @param[in] f The activation function to use.
     * @param[in] a (Optional) The alpha parameter used by some activation functions
     *              (@ref ActivationFunction::BOUNDED_RELU, @ref ActivationFunction::LU_BOUNDED_RELU, @ref ActivationFunction::LINEAR, @ref ActivationFunction::TANH).
     * @param[in] b (Optional) The beta parameter used by some activation functions (@ref ActivationFunction::LINEAR, @ref ActivationFunction::LU_BOUNDED_RELU, @ref ActivationFunction::TANH).
     */
    ActivationLayerInfo(ActivationFunction f, float a = 0.0f, float b = 0.0f)
        : _act(f), _a(a), _b(b), _enabled(true)
    {
    }
    /** Get the type of activation function */
    ActivationFunction activation() const
    {
        return _act;
    }
    /** Get the alpha value */
    float a() const
    {
        return _a;
    }
    /** Get the beta value */
    float b() const
    {
        return _b;
    }
    /** Check if initialised */
    bool enabled() const
    {
        return _enabled;
    }

private:
    ActivationFunction _act     = { ActivationLayerInfo::ActivationFunction::LOGISTIC };
    float              _a       = {};
    float              _b       = {};
    bool               _enabled = { false };
};

/** Normalization Layer Information class */
class NormalizationLayerInfo
{
public:
    /** Default Constructor
     *
     * @param[in] type      The normalization type. Can be @ref NormType::IN_MAP_1D, @ref NormType::IN_MAP_2D or @ref NORM_TYPE::CROSS_MAP
     * @param[in] norm_size The normalization size is the number of elements to normalize across. Defaults to 5.
     * @param[in] alpha     (Optional) Alpha parameter used by normalization equation. Defaults to 0.0001.
     * @param[in] beta      (Optional) Beta parameter used by normalization equation. Defaults to 0.5.
     * @param[in] kappa     (Optional) Kappa parameter used by [Krichevksy 2012] Across Channel Local Brightness Normalization equation.
     * @param[in] is_scaled (Optional) Boolean that specifies if alpha will be scaled by the normalization size or not.
     *                      Should be false to follow [Krichevksy 2012].
     */
    NormalizationLayerInfo(NormType type, uint32_t norm_size = 5, float alpha = 0.0001f, float beta = 0.5f, float kappa = 1.f, bool is_scaled = true)
        : _type(type), _norm_size(norm_size), _alpha(alpha), _beta(beta), _kappa(kappa), _is_scaled(is_scaled)
    {
    }
    /** Get the normalization type */
    NormType type() const
    {
        return _type;
    }
    /** Get the normalization size */
    uint32_t norm_size() const
    {
        return _norm_size;
    }
    /** Get the alpha value */
    float alpha() const
    {
        return _alpha;
    }
    /** Get the beta value */
    float beta() const
    {
        return _beta;
    }
    /** Get the kappa value */
    float kappa() const
    {
        return _kappa;
    }
    /** Check if normalization is cross map */
    bool is_cross_map() const
    {
        return _type == NormType::CROSS_MAP;
    }
    /** Check if normalization is not cross map */
    bool is_in_map() const
    {
        return !is_cross_map();
    }
    /** Return the scaling factor of the normalization function.
     *
     * If is_scaled is set to false then [Krichevksy 2012] normalization scaling is performed,
     * where alpha is returned plainly, else alpha is scaled by the total number of elements used for the normalization.
     *
     * @return The normalization scaling factor.
     */
    float scale_coeff() const
    {
        const uint32_t size = (_type == NormType::IN_MAP_2D) ? _norm_size * _norm_size : _norm_size;
        return (_is_scaled) ? (_alpha / size) : _alpha;
    }

private:
    NormType _type;
    uint32_t _norm_size;
    float    _alpha;
    float    _beta;
    float    _kappa;
    bool     _is_scaled;
};

/** Convolution Layer Weights Information class. This class stores the necessary information to compute convolution layer when the weights are already reshaped */
class WeightsInfo
{
public:
    /** Default constructor */
    WeightsInfo()
        : _are_reshaped(false), _kernel_width(0), _kernel_height(0), _num_kernels(0), _retain_internal_weights(false)
    {
    }
    /** Constructor
     *
     * @param[in] are_reshaped            True if the weights have been reshaped
     * @param[in] kernel_width            Kernel width.
     * @param[in] kernel_height           Kernel height.
     * @param[in] num_kernels             Number of convolution kernels.
     * @param[in] retain_internal_weights (Optional) True if internal reshaped weights must be retained. Used for reconfiguration purposes. Default is false.
     */
    WeightsInfo(bool are_reshaped, unsigned int kernel_width, unsigned int kernel_height, unsigned int num_kernels, bool retain_internal_weights = false)
        : _are_reshaped(are_reshaped), _kernel_width(kernel_width), _kernel_height(kernel_height), _num_kernels(num_kernels), _retain_internal_weights(retain_internal_weights)
    {
    }
    /** Flag which specifies if the weights tensor has been reshaped.
     *
     * @return True if the weights tensors has been reshaped
     */
    bool are_reshaped() const
    {
        return _are_reshaped;
    };
    /** Return the number of convolution kernels
     *
     * @return The number of convolution kernels
     */
    unsigned int num_kernels() const
    {
        return _num_kernels;
    };
    /** Return the width and height of the kernel
     *
     * @return The width and height of the kernel
     */
    std::pair<unsigned int, unsigned int> kernel_size() const
    {
        return std::make_pair(_kernel_width, _kernel_height);
    }
    bool retain_internal_weights() const
    {
        return _retain_internal_weights;
    }

private:
    const bool         _are_reshaped;
    const unsigned int _kernel_width;
    const unsigned int _kernel_height;
    const unsigned int _num_kernels;
    const bool         _retain_internal_weights;
};

/** GEMM reshape information class. This class stores the necessary information about matrix A and matrix B reshape.
 *
 * The matrix A can only be reshaped through @ref CLGEMMInterleave4x4Kernel or  @ref NEGEMMInterleave4x4Kernel or  @ref GCGEMMInterleave4x4Kernel
 * Note: Optionally just for @ref CLGEMMInterleave4x4Kernel is it possible to set mult_interleave4x4_height, the multiplication factor for the height of the 4x4 interleaved block
 *
 * The matrix B can only be reshaped through @ref CLGEMMTranspose1xWKernel or  @ref NEGEMMTranspose1xWKernel or  @ref GCGEMMTranspose1xWKernel
 * Note: Optionally just for @ref CLGEMMTranspose1xWKernel is it possible to set mult_transpose1xW_width, the multiplication factor for the width of the 1xW transposed block
 *
 */
class GEMMReshapeInfo final
{
public:
    /** Default constructor */
    GEMMReshapeInfo()
        : _m(1), _n(1), _k(1), _mult_transpose1xW_width(1), _mult_interleave4x4_height(1), _depth_output_gemm3d(1), _reinterpret_input_as_3d(false)
    {
    }
    /** Constructor
     *
     * @param[in] m                         Number of matrix A rows
     * @param[in] n                         Number of matrix B columns
     * @param[in] k                         Number of matrix A columns or matrix B rows
     * @param[in] mult_transpose1xW_width   (Optional) Multiplication factor for the width of the 1xW transposed block
     * @param[in] mult_interleave4x4_height (Optional) Multiplication factor for the height of the 4x4 interleaved block
     * @param[in] depth_output_gemm3d       (Optional) Depth (third dimension) of the output tensor to be used with the GEMM3D kernel
     * @param[in] reinterpret_input_as_3d   (Optional) Reinterpret the input as 3D tensor. (i.e. this flag should be set to true when GEMM is used
     *                                                 to perform 1x1 convolutions with the NHWC data layout)
     */
    GEMMReshapeInfo(int m, int n, int k, int mult_transpose1xW_width = 1, int mult_interleave4x4_height = 1, int depth_output_gemm3d = 1, bool reinterpret_input_as_3d = false)
        : _m(m), _n(n), _k(k), _mult_transpose1xW_width(mult_transpose1xW_width), _mult_interleave4x4_height(mult_interleave4x4_height), _depth_output_gemm3d(depth_output_gemm3d),
          _reinterpret_input_as_3d(reinterpret_input_as_3d)
    {
    }
    /** Number of matrix A rows
     *
     * @return the number of matrix A rows
     */
    int m() const
    {
        return _m;
    }
    /** Number of matrix B columns
     *
     * @return the number of matrix B columns
     */
    int n() const
    {
        return _n;
    }
    /** Number of matrix A columns or matrix B rows
     *
     * @return the number of matrix A columns or matrix B rows
     */
    int k() const
    {
        return _k;
    }
    /** Multiplication factor for the width of the 1xW transposed block
     *
     * @return the multiplication factor for the width of the 1xW transposed block
     */
    int mult_transpose1xW_width() const
    {
        return _mult_transpose1xW_width;
    }
    /** Multiplication factor for the height of the 4x4 interleaved block
     *
     * @return the multiplication factor for the height of the 4x4 interleaved block
     */
    int mult_interleave4x4_height() const
    {
        return _mult_interleave4x4_height;
    }
    /** Depth (third dimension) of the output tensor to be used with the GEMM3D kernel
     *
     * @note GEMM3D kernel is used when the output has to be reinterpret as 3D tensor. In that case:
     *       m = depth_output_gemm3d * output_height
     *
     * @return the depth of the output tensor to be used with the GEMM3D kernel
     */
    int depth_output_gemm3d() const
    {
        return _depth_output_gemm3d;
    }
    /** Flag which specifies if the input tensor has to be reinterpreted as 3D
     *
     * @return True if the input tensor has to be reinterpreted as 3D tensor
     */
    bool reinterpret_input_as_3d() const
    {
        return _reinterpret_input_as_3d;
    };

private:
    const int  _m;
    const int  _n;
    const int  _k;
    const int  _mult_transpose1xW_width;
    const int  _mult_interleave4x4_height;
    const int  _depth_output_gemm3d;
    const bool _reinterpret_input_as_3d;
};

/** GEMM information class. This class stores the necessary information to compute GEMM functions
 *
 * This object also contains the information about how matrix A and matrix B have been reshaped
 *
 */
class GEMMInfo
{
public:
    /** Default constructor */
    GEMMInfo()
        : _is_a_reshaped(false), _is_b_reshaped(false), _reshape_b_only_on_first_run(false), _depth_output_gemm3d(1), _reinterpret_input_as_3d(false), _retain_internal_weights(false)
    {
    }
    /** Constructor
     *
     * @param[in] is_a_reshaped               True if the matrix A has been reshaped
     * @param[in] is_b_reshaped               True if the matrix B has been reshaped
     * @param[in] reshape_b_only_on_first_run Reshape matrix B only for the first run
     * @param[in] depth_output_gemm3d         (Optional) Depth (third dimension) of the output tensor to be used with the GEMM3D kernel
     * @param[in] reinterpret_input_as_3d     (Optional) Reinterpret the input as 3D tensor. (i.e. this flag should be set to true when GEMM is used
     *                                        to perform 1x1 convolutions with the NHWC data layout)
     * @param[in] retain_internal_weights     (Optional) Retain the weights tensor from previous run
     *
     */
    GEMMInfo(bool is_a_reshaped, bool is_b_reshaped, bool reshape_b_only_on_first_run, int depth_output_gemm3d = 1, bool reinterpret_input_as_3d = false, bool retain_internal_weights = false)
        : _is_a_reshaped(is_a_reshaped), _is_b_reshaped(is_b_reshaped), _reshape_b_only_on_first_run(reshape_b_only_on_first_run), _depth_output_gemm3d(depth_output_gemm3d),
          _reinterpret_input_as_3d(reinterpret_input_as_3d), _retain_internal_weights(retain_internal_weights)
    {
    }
    /** Flag which specifies if the matrix A has been reshaped
     *
     * @return True if the matrix A has been reshaped
     */
    bool is_a_reshaped() const
    {
        return _is_a_reshaped;
    };
    /** Flag which specifies if the matrix B has been reshaped
     *
     * @return True if the matrix B has been reshaped
     */
    bool is_b_reshaped() const
    {
        return _is_b_reshaped;
    };
    /** Flag which specifies if the reshape of matrix B should executed only for the first
     *
     * @note This flag could be set to TRUE when GEMM is used to accelerate convolution layer
     *
     * @return True if the reshaped of matrix B happens only for the first run
     */
    bool reshape_b_only_on_first_run() const
    {
        return _reshape_b_only_on_first_run;
    };
    /** Depth of the output when GEMM output is reinterpreted as 3D tensor
     *
     * @return the depth of the output tensor
     */
    int depth_output_gemm3d() const
    {
        return _depth_output_gemm3d;
    };
    /** Flag which specifies if the input tensor has to be reinterpreted as 3D
     *
     * @return True if the input tensor has to be reinterpreted as 3D tensor
     */
    bool reinterpret_input_as_3d() const
    {
        return _reinterpret_input_as_3d;
    };
    /** Flag which specifies if the weights tensor has to be retained from previous run
     *
     * @return True if the weights tensor has to be retained
     */
    bool retain_internal_weights() const
    {
        return _retain_internal_weights;
    };

private:
    const bool _is_a_reshaped;
    const bool _is_b_reshaped;
    const bool _reshape_b_only_on_first_run;
    const int  _depth_output_gemm3d;
    const bool _reinterpret_input_as_3d;
    const bool _retain_internal_weights;
};

/** Winograd information */
struct WinogradInfo
{
    /** Default constructor
     *
     * @param[in] output_tile_sz Width and height of the output tile
     * @param[in] kernel_sz      Width and height of the kernel
     * @param[in] input_dims     Width and height of the input tensor before the convolution is applied
     * @param[in] conv_info      Convolution info (Pads, strides)
     * @param[in] data_layout    Data layout to use for the output tensor once the convolution has been applied
     */
    WinogradInfo(Size2D output_tile_sz, Size2D kernel_sz, Size2D input_dims, PadStrideInfo conv_info, DataLayout data_layout)
        : output_tile_size(output_tile_sz), kernel_size(kernel_sz), input_dimensions(input_dims), convolution_info(conv_info), output_data_layout(data_layout)
    {
    }

    Size2D        output_tile_size{};                     /**< Width and height of the output tile */
    Size2D        kernel_size{};                          /**< Width and height of the kernel*/
    Size2D        input_dimensions{};                     /**< Width and height of the input tensor before the convolution is applied */
    PadStrideInfo convolution_info{};                     /**< Convolution info (Pads, strides,...) */
    DataLayout    output_data_layout{ DataLayout::NCHW }; /**< Data layout to use for the output tensor once the convolution has been applied (NCHW or NHWC) */
};

/** IO formatting information class*/
struct IOFormatInfo
{
    /** Precision type used when printing floating point numbers */
    enum class PrecisionType
    {
        Default, /**< Default precision to the one that the current stream has */
        Custom,  /**< Custom precision specified by the user using the precision parameter */
        Full     /**< The maximum precision of the floating point representation */
    };

    /** Specifies the area to be printed, used by Tensor objects */
    enum class PrintRegion
    {
        ValidRegion, /**< Prints the valid region of the Tensor object */
        NoPadding,   /**< Prints the Tensor object without the padding */
        Full         /**< Print the tensor object including padding */
    };

    /** Construct a set of IO formatting information.
     *
     * @param[in] print_region   Area to be printed. Used by Tensor objects. Default: ValidRegion.
     * @param[in] precision_type Precision type for floating point numbers. Default: stream default.
     * @param[in] precision      Precision value for float point numbers. Default: 10.
     * @param[in] align_columns  Whether to align columns when printed. Default: true.
     * @param[in] element_delim  Delimeter between elements. Default: " ".
     * @param[in] row_delim      Delimenter between rows. Default: "\n".
     */
    IOFormatInfo(PrintRegion   print_region   = PrintRegion::ValidRegion,
                 PrecisionType precision_type = PrecisionType::Default,
                 unsigned int  precision      = 10,
                 bool          align_columns  = true,
                 std::string   element_delim  = " ",
                 std::string   row_delim      = "\n")
        : print_region(print_region),
          precision_type(precision_type),
          precision(precision),
          element_delim(element_delim),
          row_delim(row_delim),
          align_columns(align_columns)
    {
    }

    /** Area to be printed by Tensor objects */
    PrintRegion print_region;
    /** Floating point precision type */
    PrecisionType precision_type;
    /** Floating point precision */
    unsigned int precision;
    /** Element delimeter */
    std::string element_delim;
    /** Row delimeter */
    std::string row_delim;
    /** Align columns */
    bool align_columns;
};

/** Available ConvolutionMethod*/
enum class ConvolutionMethod
{
    GEMM,    /**< Convolution using GEMM */
    DIRECT,  /**< Direct convolution */
    WINOGRAD /**< Convolution using Winograd */
};
} // namespace arm_compute
#endif /* __ARM_COMPUTE_TYPES_H__ */
