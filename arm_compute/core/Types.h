/*
 * Copyright (c) 2016, 2017 ARM Limited.
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
#include "arm_compute/core/TensorShape.h"
#include "support/Half.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

namespace arm_compute
{
/** 16-bit floating point type */
using half = half_float::half;

/** Image colour formats */
enum class Format
{
    UNKNOWN,  /** Unknown image format */
    U8,       /** 1 channel, 1 U8 per channel */
    S16,      /** 1 channel, 1 S16 per channel */
    U16,      /** 1 channel, 1 U16 per channel */
    S32,      /** 1 channel, 1 S32 per channel */
    U32,      /** 1 channel, 1 U32 per channel */
    F16,      /** 1 channel, 1 F16 per channel */
    F32,      /** 1 channel, 1 F32 per channel */
    UV88,     /** 2 channel, 1 U8 per channel */
    RGB888,   /** 3 channels, 1 U8 per channel */
    RGBA8888, /** 4 channels, 1 U8 per channel */
    YUV444,   /** A 3 plane of 8 bit 4:4:4 sampled Y, U, V planes */
    YUYV422,  /** A single plane of 32-bit macro pixel of Y0, U0, Y1, V0 bytes */
    NV12,     /** A 2 plane YUV format of Luma (Y) and interleaved UV data at 4:2:0 sampling */
    NV21,     /** A 2 plane YUV format of Luma (Y) and interleaved VU data at 4:2:0 sampling */
    IYUV,     /** A 3 plane of 8-bit 4:2:0 sampled Y, U, V planes */
    UYVY422   /** A single plane of 32-bit macro pixel of U0, Y0, V0, Y1 byte */
};

/** Available data types */
enum class DataType
{
    UNKNOWN,
    U8,
    S8,
    QS8,
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

/** Constant value of the border pixels when using BorderMode::CONSTANT */
constexpr uint8_t CONSTANT_BORDER_VALUE = 199;

/* Constant value used to indicate a half-scale pyramid */
constexpr float SCALE_PYRAMID_HALF = 0.5f;

/* Constant value used to indicate a ORB scaled pyramid */
constexpr float SCALE_PYRAMID_ORB = 8.408964152537146130583778358414e-01;

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

    ValidRegion(Coordinates anchor, TensorShape shape)
        : anchor{ anchor }, shape{ shape }
    {
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
    constexpr BorderSize(unsigned int size)
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

/** Rounding method */
enum class RoundingPolicy
{
    TO_ZERO,        /**< Truncates the least significand values that are lost in operations. */
    TO_NEAREST_UP,  /**< Rounds to nearest value; half rounds up */
    TO_NEAREST_EVEN /**< Rounds to nearest value; half rounds to nearest even */
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
          _pad(std::make_pair(pad_x, pad_y)),
          _round_type(round)
    {
    }
    std::pair<unsigned int, unsigned int> stride() const
    {
        return _stride;
    }
    std::pair<unsigned int, unsigned int> pad() const
    {
        return _pad;
    }
    DimensionRoundingType round() const
    {
        return _round_type;
    }

private:
    std::pair<unsigned int, unsigned int> _stride;
    std::pair<unsigned int, unsigned int> _pad;
    DimensionRoundingType _round_type;
};

/** Pooling Layer Information class */
class PoolingLayerInfo
{
public:
    /** Default Constructor
     *
     * @param[in] pool_type       Pooling type @ref PoolingType. Defaults to @ref PoolingType::MAX
     * @param[in] pool_size       (Optional) Pooling size, in elements, across  x and y. Defaults to 2.
     * @param[in] pad_stride_info (Optional) Padding and stride information @ref PadStrideInfo
     */
    PoolingLayerInfo(PoolingType pool_type = PoolingType::MAX, unsigned int pool_size = 2, PadStrideInfo pad_stride_info = PadStrideInfo())
        : _pool_type(pool_type), _pool_size(pool_size), _pad_stride_info(pad_stride_info)
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

private:
    PoolingType   _pool_type;
    unsigned int  _pool_size;
    PadStrideInfo _pad_stride_info;
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
     * @param[in] alpha     Alpha parameter used by normalization equation. Defaults to 0.0001.
     * @param[in] beta      Beta parameter used by normalization equation. Defaults to 0.5.
     * @param[in] kappa     Kappa parameter used by [Krichevksy 2012] Across Channel Local Brightness Normalization equation.
     */
    NormalizationLayerInfo(NormType type, uint32_t norm_size = 5, float alpha = 0.0001f, float beta = 0.5f, float kappa = 1.f)
        : _type(type), _norm_size(norm_size), _alpha(alpha), _beta(beta), _kappa(kappa)
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
    /** Return the scaling factor of the normalization function. If kappa is not
     * 1 then [Krichevksy 2012] normalization scaling is specified. Scaling
     * factor takes into account the total number of elements used for the
     * normalization, so in case of 2 dimensions this is _norm_size^2.
     *
     * @return The normalization scaling factor.
     */
    float scale_coeff() const
    {
        const uint32_t size = (_type == NormType::IN_MAP_2D) ? _norm_size * _norm_size : _norm_size;
        return (_kappa == 1.f) ? (_alpha / size) : _alpha;
    }

private:
    NormType _type;
    uint32_t _norm_size;
    float    _alpha;
    float    _beta;
    float    _kappa;
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
