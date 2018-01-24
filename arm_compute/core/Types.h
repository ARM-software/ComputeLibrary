/*
 * Copyright (c) 2016, 2018 ARM Limited.
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
    UNKNOWN,
    U8,
    S8,
    QS8,
    QASYMM8,
    U16,
    S16,
    QS16,
    U32,
    S32,
    QS32,
    U64,
    S64,
    F16,
    F32,
    F64,
    SIZET
};

/** Available Sampling Policies */
enum class SamplingPolicy
{
    CENTER,  /**< Samples are taken at pixel center */
    TOP_LEFT /**< Samples are taken at pixel top left corner */
};

/** Constant value of the border pixels when using BorderMode::CONSTANT */
constexpr uint8_t CONSTANT_BORDER_VALUE = 199;

/* Constant value used to indicate a half-scale pyramid */
constexpr float SCALE_PYRAMID_HALF = 0.5f;

/* Constant value used to indicate a ORB scaled pyramid */
constexpr float SCALE_PYRAMID_ORB = 8.408964152537146130583778358414e-01;

/** Quantization settings (used for QASYMM8 data type) */
struct QuantizationInfo
{
    QuantizationInfo()
        : scale(0.0f), offset(0)
    {
    }

    QuantizationInfo(float scale, int offset)
        : scale(scale), offset(offset)
    {
    }

    bool operator==(const QuantizationInfo &other)
    {
        return scale == other.scale && offset == other.offset;
    }

    bool operator!=(const QuantizationInfo &other)
    {
        return !(*this == other);
    }

    float scale;  /**< scale */
    int   offset; /**< offset */

    /** Quantizes a value using the scale/offset in this QuantizationInfo */
    qasymm8_t quantize(float value, RoundingPolicy rounding_policy) const
    {
        ARM_COMPUTE_ERROR_ON_MSG(scale == 0, "QuantizationInfo::quantize: scale == 0");
        return sqcvt_qasymm8_f32(value, scale, offset, rounding_policy);
    }

    /** Dequantizes a value using the scale/offset in this QuantizationInfo */
    float dequantize(qasymm8_t value) const
    {
        ARM_COMPUTE_ERROR_ON_MSG(scale == 0, "QuantizationInfo::dequantize: scale == 0");
        return scvt_f32_qasymm8(value, scale, offset);
    }

    /** Indicates whether this QuantizationInfo has valid settings or not */
    bool empty() const
    {
        return scale == 0;
    }
};

struct ValidRegion
{
    ValidRegion()
        : anchor{}, shape{}
    {
    }

    ValidRegion(const ValidRegion &) = default;
    ValidRegion(ValidRegion &&)      = default;
    ValidRegion &operator=(const ValidRegion &) = default;
    ValidRegion &operator=(ValidRegion &&) = default;
    ~ValidRegion()                         = default;

    ValidRegion(const Coordinates &an_anchor, const TensorShape &a_shape)
        : anchor{ an_anchor }, shape{ a_shape }
    {
        anchor.set_num_dimensions(std::max(anchor.num_dimensions(), shape.num_dimensions()));
    }

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

    Coordinates anchor;
    TensorShape shape;
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

    BorderSize &operator*=(float scale)
    {
        top *= scale;
        right *= scale;
        bottom *= scale;
        left *= scale;

        return *this;
    }

    BorderSize operator*(float scale)
    {
        BorderSize size = *this;
        size *= scale;

        return size;
    }

    void limit(const BorderSize &limit)
    {
        top    = std::min(top, limit.top);
        right  = std::min(right, limit.right);
        bottom = std::min(bottom, limit.bottom);
        left   = std::min(left, limit.left);
    }

    unsigned int top;
    unsigned int right;
    unsigned int bottom;
    unsigned int left;
};

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
    BILINEAR_OLD_NEW,
    BILINEAR_SCHARR
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
    TERM_CRITERIA_EPSILON,
    TERM_CRITERIA_ITERATIONS,
    TERM_CRITERIA_BOTH
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
    std::pair<unsigned int, unsigned int> stride() const
    {
        return _stride;
    }
    std::pair<unsigned int, unsigned int> pad() const
    {
        //this accessor should be used only when padding is symmetric
        ARM_COMPUTE_ERROR_ON(_pad_left != _pad_right || _pad_top != _pad_bottom);
        return std::make_pair(_pad_left, _pad_top);
    }

    unsigned int pad_left() const
    {
        return _pad_left;
    }
    unsigned int pad_right() const
    {
        return _pad_right;
    }
    unsigned int pad_top() const
    {
        return _pad_top;
    }
    unsigned int pad_bottom() const
    {
        return _pad_bottom;
    }

    DimensionRoundingType round() const
    {
        return _round_type;
    }

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

/** Pooling Layer Information class */
class PoolingLayerInfo
{
public:
    /** Default Constructor */
    PoolingLayerInfo()
        : _pool_type(PoolingType::MAX), _pool_size(0), _pad_stride_info(PadStrideInfo()), _exclude_padding(false), _is_global_pooling(false)
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
        : _pool_type(pool_type), _pool_size(0), _pad_stride_info(PadStrideInfo(1, 1, 0, 0)), _exclude_padding(false), _is_global_pooling(true)
    {
    }
    PoolingType pool_type() const
    {
        return _pool_type;
    }
    unsigned int pool_size() const
    {
        return _pool_size;
    }
    PadStrideInfo pad_stride_info() const
    {
        return _pad_stride_info;
    }
    bool exclude_padding() const
    {
        return _exclude_padding;
    }
    bool is_global_pooling() const
    {
        return _is_global_pooling;
    }

private:
    PoolingType   _pool_type;
    unsigned int  _pool_size;
    PadStrideInfo _pad_stride_info;
    bool          _exclude_padding;
    bool          _is_global_pooling;
};

/** ROI Pooling Layer Information class */
class ROIPoolingLayerInfo
{
public:
    /** Default Constructor
     *
     * @param[in] pooled_width  Pooled width of the layer.
     * @param[in] pooled_height Pooled height of the layer.
     * @param[in] spatial_scale Spatial scale to be applied to the ROI coordinates and dimensions.
     */
    ROIPoolingLayerInfo(unsigned int pooled_width, unsigned int pooled_height, float spatial_scale)
        : _pooled_width(pooled_width), _pooled_height(pooled_height), _spatial_scale(spatial_scale)
    {
    }
    unsigned int pooled_width() const
    {
        return _pooled_width;
    }
    unsigned int pooled_height() const
    {
        return _pooled_height;
    }
    float spatial_scale() const
    {
        return _spatial_scale;
    }

private:
    unsigned int _pooled_width;
    unsigned int _pooled_height;
    float        _spatial_scale;
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

    /** Default Constructor
     *
     * @param[in] f The activation function to use.
     * @param[in] a (Optional) The alpha parameter used by some activation functions
     *              (@ref ActivationFunction::BOUNDED_RELU, @ref ActivationFunction::LU_BOUNDED_RELU, @ref ActivationFunction::LINEAR, @ref ActivationFunction::TANH).
     * @param[in] b (Optional) The beta parameter used by some activation functions (@ref ActivationFunction::LINEAR, @ref ActivationFunction::LU_BOUNDED_RELU, @ref ActivationFunction::TANH).
     */
    ActivationLayerInfo(ActivationFunction f, float a = 0.0f, float b = 0.0f)
        : _act(f), _a(a), _b(b)
    {
    }
    ActivationFunction activation() const
    {
        return _act;
    }
    float a() const
    {
        return _a;
    }
    float b() const
    {
        return _b;
    }

private:
    ActivationFunction _act;
    float              _a;
    float              _b;
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
    NormType type() const
    {
        return _type;
    }
    uint32_t norm_size() const
    {
        return _norm_size;
    }
    float alpha() const
    {
        return _alpha;
    }
    float beta() const
    {
        return _beta;
    }
    float kappa() const
    {
        return _kappa;
    }
    bool is_cross_map() const
    {
        return _type == NormType::CROSS_MAP;
    }
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
        : _are_reshaped(false), _kernel_width(0), _kernel_height(0), _num_kernels(0)
    {
    }
    /** Constructor
     *
     * @param[in] are_reshaped  True if the weights have been reshaped
     * @param[in] kernel_width  Kernel width.
     * @param[in] kernel_height Kernel height.
     * @param[in] num_kernels   Number of convolution kernels.
     */
    WeightsInfo(bool are_reshaped, unsigned int kernel_width, unsigned int kernel_height, unsigned int num_kernels)
        : _are_reshaped(are_reshaped), _kernel_width(kernel_width), _kernel_height(kernel_height), _num_kernels(num_kernels)
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

private:
    const bool         _are_reshaped;
    const unsigned int _kernel_width;
    const unsigned int _kernel_height;
    const unsigned int _num_kernels;
};

/** GEMM Information class. This class stores the necessary information to compute GEMM functions */
class GEMMInfo
{
public:
    /** Default constructor */
    GEMMInfo()
        : _is_a_reshaped(false), _is_b_reshaped(false), _reshape_b_only_on_first_run(false)
    {
    }
    /** Constructor
     *
     * @param[in] is_a_reshaped               True if the matrix A has been reshaped
     * @param[in] is_b_reshaped               True if the matrix B has been reshaped
     * @param[in] reshape_b_only_on_first_run Reshape matrix B only for the first run
     */
    GEMMInfo(bool is_a_reshaped, bool is_b_reshaped, bool reshape_b_only_on_first_run)
        : _is_a_reshaped(is_a_reshaped), _is_b_reshaped(is_b_reshaped), _reshape_b_only_on_first_run(reshape_b_only_on_first_run)
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

private:
    const bool _is_a_reshaped;
    const bool _is_b_reshaped;
    const bool _reshape_b_only_on_first_run;
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

    PrintRegion   print_region;
    PrecisionType precision_type;
    unsigned int  precision;
    std::string   element_delim;
    std::string   row_delim;
    bool          align_columns;
};
}
#endif /* __ARM_COMPUTE_TYPES_H__ */
