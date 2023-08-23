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
#ifndef ACL_ARM_COMPUTE_CORE_TYPES
#define ACL_ARM_COMPUTE_CORE_TYPES

/** The following symbols have been moved to:
 * half
 * PermutationVector
 * Format
 * DataType
 * DataLayout
 * DataLayoutDimension
 * PadStrideInfo
 * WeightFormat
 * Channel
 * DimensionRoundingType
 */
#include "arm_compute/core/CoreTypes.h"
/** The following symbols have been moved to:
 * ActivationFunction
 * ActivationLayerInfo
 */
#include "arm_compute/function_info/ActivationLayerInfo.h"
/** The following symbols have been moved to:
 * ConvolutionInfo
 */
#include "arm_compute/function_info/ConvolutionInfo.h"
/** The following symbols have been moved to:
 * FullyConnectedLayerInfo
 */
#include "arm_compute/function_info/FullyConnectedLayerInfo.h"
/** The following symbols have been moved to:
 * GEMMLowpOutputStageType
 * GEMMLowpOutputStageInfo
 * GEMMInfo
 */
#include "arm_compute/function_info/GEMMInfo.h"
/** The following symbols have been moved to:
 * MatMulInfo
 */
#include "arm_compute/function_info/MatMulInfo.h"

#include "arm_compute/core/Coordinates.h"
#include "arm_compute/core/Size2D.h"
#include "arm_compute/core/Size3D.h"
#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/experimental/IPostOp.h"
#include "arm_compute/core/utils/misc/Macros.h"
#include "support/Bfloat16.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <utility>

namespace arm_compute
{
/** Bidirectional strides */
using BiStrides = Coordinates;

/** Available Sampling Policies */
enum class SamplingPolicy
{
    CENTER,  /**< Samples are taken at pixel center */
    TOP_LEFT /**< Samples are taken at pixel top left corner */
};

/** Available ConvolutionMethod*/
enum class ConvolutionMethod
{
    GEMM,        /**< Convolution using GEMM */
    GEMM_CONV2D, /**< Direct 2D GEMM convolution */
    DIRECT,      /**< Direct convolution */
    INDIRECT,    /**< Indirect convolution */
    WINOGRAD,    /**< Convolution using Winograd */
    FFT          /**< Convolution using FFT */
};

/** Available DepthwiseConvolutionFunction*/
enum class DepthwiseConvolutionFunction
{
    OPTIMIZED, /**< Optimized Depthwise Convolution */
    GENERIC,   /**< Generic Depthwise Convolution */
};

/** Available DeconvolutionMethod*/
enum class DeconvolutionMethod
{
    GEMM,          /**< Deconvolution using GEMM */
    DIRECT,        /**< Direct deconvolution */
    UPSCALE_CONV2D /**< Deconvolution with Upscaling */
};

/** Available FuseBatchNormalizationType*/
enum class FuseBatchNormalizationType
{
    CONVOLUTION,         /**< For Convolution weights */
    DEPTHWISECONVOLUTION /**< For Depthwise Convolution weights*/
};

/** Padding mode to use for PadLayer */
enum class PaddingMode
{
    CONSTANT,
    REFLECT,
    SYMMETRIC
};

/** Supported comparison operations */
enum class ComparisonOperation
{
    Equal,        /**< Equal comparison ( \f$ x == y \f$ ) */
    NotEqual,     /**< NotEqual comparison ( \f$ x != y \f$ ) */
    Greater,      /**< Greater comparison ( \f$ x > y \f$ ) */
    GreaterEqual, /**< Greater equal comparison ( \f$ x >= y \f$ ) */
    Less,         /**< Less comparison ( \f$ x < y \f$ ) */
    LessEqual     /**< Less equal comparison ( \f$ x <= y \f$ ) */
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

    /** Check whether two valid regions are equal.
     *
     * @param[in] lhs LHS valid region
     * @param[in] rhs RHS valid region
     *
     * @return True if the valid regions are the same.
     */
    inline friend bool operator==(const ValidRegion &lhs, const ValidRegion &rhs);

    Coordinates anchor; /**< Anchor for the start of the valid region. */
    TensorShape shape;  /**< Shape of the valid region. */
};
inline bool operator==(const ValidRegion &lhs, const ValidRegion &rhs)
{
    return (lhs.anchor == rhs.anchor) && (lhs.shape == rhs.shape);
}

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
    constexpr BorderSize() noexcept
        : top{ 0 },
    right{ 0 },
    bottom{ 0 },
    left{ 0 }
    {
    }

    /** Border with equal size around the 2D plane */
    explicit constexpr BorderSize(unsigned int size) noexcept
        : top{ size },
    right{ size },
    bottom{ size },
    left{ size }
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

    /** Check equality with another BorderSize struct
     *
     * @param[in] rhs other struct to check against
     *
     * @return true if they are equal
     */
    bool operator==(const BorderSize &rhs) const
    {
        return (top == rhs.top) && (right == rhs.right) && (bottom == rhs.bottom) && (left == rhs.left);
    }

    /** Check non-equality with another BorderSize struct
     *
     * @param[in] rhs other struct to check against
     *
     * @return true if they are different
     */
    bool operator!=(const BorderSize &rhs) const
    {
        return !(*this == rhs);
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

/** Policy to handle integer overflow
 *  @note: This is ignored by floating point operations where the overflow behavior adheres to the IEEE-754 standard
 *         which states that in case of overflow ±infinity is returned for the round-to-nearest modes (and follows the
 *         rounding rules for the directed rounding modes) by default.
 */
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

/** Information to produce a tiled version of a Tensor */
using Multiples = std::vector<uint32_t>;

/** Available reduction operations */
enum class ReductionOperation
{
    ARG_IDX_MAX, /**< Index of the max value */
    ARG_IDX_MIN, /**< Index of the min value */
    MEAN_SUM,    /**< Mean of sum */
    PROD,        /**< Product */
    SUM_SQUARE,  /**< Sum of squares */
    SUM,         /**< Sum */
    MIN,         /**< Min */
    MAX,         /**< Max */
};

/** Available element-wise operations */
enum class ArithmeticOperation
{
    ADD,          /**< (x + y) */
    SUB,          /**< (x  - y) */
    DIV,          /**< (x / y) */
    MIN,          /**< Min(x, y) */
    MAX,          /**< Max(x, y) */
    SQUARED_DIFF, /**< (x - y)^2 */
    POWER,        /**< x ^ y */
    PRELU,        /**< y*x if x < 0, x otherwise */
};

/** Available element wise unary operations */
enum class ElementWiseUnary
{
    RSQRT,       /**< Reverse square root */
    EXP,         /**< Exponential */
    NEG,         /**< Negate */
    LOG,         /**< Natural Logarithm */
    ABS,         /**< Absolute value */
    SIN,         /**< Sine */
    ROUND,       /**< Round */
    LOGICAL_NOT, /**< Logical Not */
};

/** Available bitwise operations */
enum class BitwiseOperation
{
    AND, /**< Bitwise AND operation */
    NOT, /**< Bitwise NOT operation */
    OR,  /**< Bitwise OR operation  */
    XOR, /**< Bitwise XOR operation  */
};

/** The normalization type used for the normalization layer */
enum class NormType
{
    IN_MAP_1D, /**< Normalization applied within the same map in 1D region */
    IN_MAP_2D, /**< Normalization applied within the same map in 2D region */
    CROSS_MAP  /**< Normalization applied cross maps */
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

/** Available pooling types */
enum class PoolingType
{
    MAX, /**< Max Pooling */
    AVG, /**< Average Pooling */
    L2   /**< L2 Pooling */
};

/** Available non maxima suppression types */
enum class NMSType
{
    LINEAR,   /**< Linear NMS */
    GAUSSIAN, /**< Gaussian NMS */
    ORIGINAL  /**< Original NMS */
};

/** BoxWithNonMaximaSuppressionLimit Information class */
class BoxNMSLimitInfo final
{
public:
    /** Constructor
     *
     * @param[in] score_thresh             (Optional) Score threshold.
     * @param[in] nms                      (Optional) NMS value
     * @param[in] detections               (Optional) Number of detections
     * @param[in] soft_nms_enabled         (Optional) Enable SoftNMS
     * @param[in] soft_nms_method          (Optional) Soft NMS method
     * @param[in] soft_nms_sigma           (Optional) Soft NMS sigma value
     * @param[in] soft_nms_min_score_thres (Optional) Soft NMS minimum score threshold
     * @param[in] suppress_size            (Optional) Filter out boxes based on their size. Defaults to false
     * @param[in] min_size                 (Optional) Smaller boxes than min_size will be filtered out. Defaults to 1
     * @param[in] im_width                 (Optional) Boxes whose centers (on the x axis) is beyond im_width will be filtered. Defaults to 1
     * @param[in] im_height                (Optional) Boxes whose centers (on the y axis) is beyond im_height will be filtered. Defaults to 1
     */
    BoxNMSLimitInfo(float score_thresh = 0.05f, float nms = 0.3f,
                    int detections = 100, bool soft_nms_enabled = false,
                    NMSType soft_nms_method = NMSType::LINEAR,
                    float soft_nms_sigma = 0.5f, float soft_nms_min_score_thres = 0.001f, bool suppress_size = false, float min_size = 1.0f, float im_width = 1.0f, float im_height = 1.0f)
        : _score_thresh(score_thresh), _nms(nms), _detections_per_im(detections), _soft_nms_enabled(soft_nms_enabled), _soft_nms_method(soft_nms_method), _soft_nms_sigma(soft_nms_sigma),
          _soft_nms_min_score_thres(soft_nms_min_score_thres), _suppress_size(suppress_size), _min_size(min_size), _im_width(im_width), _im_height(im_height)
    {
    }
    /** Get the score threshold */
    float score_thresh() const
    {
        return _score_thresh;
    }
    /** Get the NMS */
    float nms() const
    {
        return _nms;
    }
    /** Get the number of detections */
    int detections_per_im() const
    {
        return _detections_per_im;
    }
    /** Check if soft NMS is enabled */
    bool soft_nms_enabled() const
    {
        return _soft_nms_enabled;
    }
    /** Get soft NMS method */
    NMSType soft_nms_method() const
    {
        return _soft_nms_method;
    }
    /** Get soft NMS sigma */
    float soft_nms_sigma() const
    {
        return _soft_nms_sigma;
    }
    /** Get soft nms min score threshold */
    float soft_nms_min_score_thres() const
    {
        return _soft_nms_min_score_thres;
    }
    /** Get if NMS will suppress boxes based on their size/position */
    bool suppress_size() const
    {
        return _suppress_size;
    }
    /** Get size suppression threshold */
    float min_size() const
    {
        return _min_size;
    }
    /** Get image width (NMS may suppress boxes whose center sits beyond the image width) */
    float im_width() const
    {
        return _im_width;
    }
    /** Get image height (NMS may suppress boxes whose center sits beyond the image height) */
    float im_height() const
    {
        return _im_height;
    }

private:
    float   _score_thresh;
    float   _nms;
    int     _detections_per_im;
    bool    _soft_nms_enabled;
    NMSType _soft_nms_method;
    float   _soft_nms_sigma;
    float   _soft_nms_min_score_thres;
    bool    _suppress_size;
    float   _min_size;
    float   _im_width;
    float   _im_height;
};

/** Padding and stride information class */
/** Padding information for 2D operations like Conv2d */
struct Padding2D
{
    Padding2D() = default;
    Padding2D(size_t left, size_t right, size_t top, size_t bottom)
        : left(left), right(right), top(top), bottom(bottom)
    {
    }
    size_t left   = { 0 }; /**<  Padding across the width dimension on the left, in elements. */
    size_t right  = { 0 }; /**<  Padding across the width dimension on the right, in elements. */
    size_t top    = { 0 }; /**<  Padding across the height dimension on the top, in elements. */
    size_t bottom = { 0 }; /**<  Padding across the height dimension on the bottom, in elements. */
};

/** Padding information for 3D operations like Conv3d */
struct Padding3D
{
    Padding3D() noexcept
    {
    }

    Padding3D(size_t pad_x, size_t pad_y, size_t pad_z)
        : left(pad_x), right(pad_x), top(pad_y), bottom(pad_y), front(pad_z), back(pad_z)
    {
    }

    Padding3D(size_t left, size_t right, size_t top, size_t bottom, size_t front, size_t back)
        : left(left), right(right), top(top), bottom(bottom), front(front), back(back)
    {
    }

    size_t left   = { 0 }; /**<  Padding across the width dimenstion on the left, in elements. */
    size_t right  = { 0 }; /**<  Padding across the width dimenstion on the right, in elements. */
    size_t top    = { 0 }; /**<  Padding across the height dimenstion  on the top, in elements. */
    size_t bottom = { 0 }; /**<  Padding across the height dimenstion on the bottom, in elements. */
    size_t front  = { 0 }; /**<  Padding across the depth dimenstion on the front, in elements. */
    size_t back   = { 0 }; /**<  Padding across the depth dimenstion on the back, in elements. */
};

/** PriorBox layer info */
class PriorBoxLayerInfo final
{
public:
    /** Default Constructor */
    PriorBoxLayerInfo()
        : _min_sizes(),
          _variances(),
          _offset(),
          _flip(true),
          _clip(false),
          _max_sizes(),
          _aspect_ratios(),
          _img_size(),
          _steps()
    {
    }
    /** Constructor
     *
     * @param[in] min_sizes     Min sizes vector.
     * @param[in] variances     Variances vector.
     * @param[in] offset        Offset value.
     * @param[in] flip          (Optional) Flip the aspect ratios.
     * @param[in] clip          (Optional) Clip coordinates so that they're within [0,1].
     * @param[in] max_sizes     (Optional) Max sizes vector.
     * @param[in] aspect_ratios (Optional) Aspect ratios of the boxes.
     * @param[in] img_size      (Optional) Image size.
     * @param[in] steps         (Optional) Step values.
     */
    PriorBoxLayerInfo(const std::vector<float> &min_sizes, const std::vector<float> &variances, float offset, bool flip = true, bool clip = false,
                      const std::vector<float> &max_sizes = {}, const std::vector<float> &aspect_ratios = {},
    const Coordinates2D &img_size = Coordinates2D{ 0, 0 }, const std::array<float, 2> &steps = { { 0.f, 0.f } })
        : _min_sizes(min_sizes),
          _variances(variances),
          _offset(offset),
          _flip(flip),
          _clip(clip),
          _max_sizes(max_sizes),
          _aspect_ratios(),
          _img_size(img_size),
          _steps(steps)
    {
        _aspect_ratios.push_back(1.);
        for(unsigned int i = 0; i < aspect_ratios.size(); ++i)
        {
            float ar            = aspect_ratios[i];
            bool  already_exist = false;
            for(auto ar_new : _aspect_ratios)
            {
                if(fabs(ar - ar_new) < 1e-6)
                {
                    already_exist = true;
                    break;
                }
            }
            if(!already_exist)
            {
                _aspect_ratios.push_back(ar);
                if(flip)
                {
                    _aspect_ratios.push_back(1.f / ar);
                }
            }
        }
    }
    /** Get min sizes. */
    std::vector<float> min_sizes() const
    {
        return _min_sizes;
    }
    /** Get min variances. */
    std::vector<float> variances() const
    {
        return _variances;
    }
    /** Get the step coordinates */
    std::array<float, 2> steps() const
    {
        return _steps;
    }
    /** Get the image size coordinates */
    Coordinates2D img_size() const
    {
        return _img_size;
    }
    /** Get the offset */
    float offset() const
    {
        return _offset;
    }
    /** Get the flip value */
    bool flip() const
    {
        return _flip;
    }
    /** Get the clip value */
    bool clip() const
    {
        return _clip;
    }
    /** Get max sizes. */
    std::vector<float> max_sizes() const
    {
        return _max_sizes;
    }
    /** Get aspect ratios. */
    std::vector<float> aspect_ratios() const
    {
        return _aspect_ratios;
    }

private:
    std::vector<float> _min_sizes;
    std::vector<float> _variances;
    float              _offset;
    bool               _flip;
    bool               _clip;
    std::vector<float> _max_sizes;
    std::vector<float> _aspect_ratios;
    Coordinates2D      _img_size;
    std::array<float, 2> _steps;
};

// Bounding Box [xmin, ymin, xmax, ymax]
using BBox = std::array<float, 4>;
// LabelBBox used for map label and bounding box
using LabelBBox = std::map<int, std::vector<BBox>>;

/** Available Detection Output code types */
enum class DetectionOutputLayerCodeType
{
    CORNER,      /**< Use box corners */
    CENTER_SIZE, /**< Use box centers and size */
    CORNER_SIZE, /**< Use box centers and size */
    TF_CENTER    /**< Use box centers and size but flip x and y co-ordinates */
};

/** Detection Output layer info */
class DetectionOutputLayerInfo final
{
public:
    /** Default Constructor */
    DetectionOutputLayerInfo()
        : _num_classes(),
          _share_location(),
          _code_type(DetectionOutputLayerCodeType::CORNER),
          _keep_top_k(),
          _nms_threshold(),
          _top_k(),
          _background_label_id(),
          _confidence_threshold(),
          _variance_encoded_in_target(false),
          _eta(),
          _num_loc_classes()
    {
        _num_loc_classes = _share_location ? 1 : _num_classes;
    }
    /** Constructor
     *
     * @param[in] num_classes                Number of classes to be predicted.
     * @param[in] share_location             If true, bounding box are shared among different classes.
     * @param[in] code_type                  Type of coding method for bbox.
     * @param[in] keep_top_k                 Number of total bounding boxes to be kept per image after NMS step.
     * @param[in] nms_threshold              Threshold to be used in NMS.
     * @param[in] top_k                      (Optional) Number of boxes per image with top confidence scores that are fed into the NMS algorithm. Default set to -1.
     * @param[in] background_label_id        (Optional) Background label ID. If there is no background class, set it as -1.
     * @param[in] confidence_threshold       (Optional) Only consider detections whose confidences are larger than a threshold. Default set to -FLT_MAX.
     * @param[in] variance_encoded_in_target (Optional) If true, variance is encoded in target. Otherwise we need to adjust the predicted offset accordingly.Default set to false.
     * @param[in] eta                        (Optional) Eta.
     */
    DetectionOutputLayerInfo(int num_classes, bool share_location, DetectionOutputLayerCodeType code_type, int keep_top_k, float nms_threshold, int top_k = -1, int background_label_id = -1,
                             float confidence_threshold = std::numeric_limits<float>::lowest(), bool variance_encoded_in_target = false, float eta = 1)
        : _num_classes(num_classes),
          _share_location(share_location),
          _code_type(code_type),
          _keep_top_k(keep_top_k),
          _nms_threshold(nms_threshold),
          _top_k(top_k),
          _background_label_id(background_label_id),
          _confidence_threshold(confidence_threshold),
          _variance_encoded_in_target(variance_encoded_in_target),
          _eta(eta),
          _num_loc_classes()
    {
        _num_loc_classes = _share_location ? 1 : _num_classes;
    }
    /** Get num classes. */
    int num_classes() const
    {
        return _num_classes;
    }
    /** Get share location. */
    bool share_location() const
    {
        return _share_location;
    }
    /** Get detection output code type. */
    DetectionOutputLayerCodeType code_type() const
    {
        return _code_type;
    }
    /** Get if variance encoded in target. */
    bool variance_encoded_in_target() const
    {
        return _variance_encoded_in_target;
    }
    /** Get the number of total bounding boxes to be kept per image. */
    int keep_top_k() const
    {
        return _keep_top_k;
    }
    /** Get nms threshold. */
    float nms_threshold() const
    {
        return _nms_threshold;
    }
    /** Get eta. */
    float eta() const
    {
        return _eta;
    }
    /** Get background label ID. */
    int background_label_id() const
    {
        return _background_label_id;
    }
    /** Get confidence threshold. */
    float confidence_threshold() const
    {
        return _confidence_threshold;
    }
    /** Get top K. */
    int top_k() const
    {
        return _top_k;
    }
    /** Get number of location classes. */
    int num_loc_classes() const
    {
        return _num_loc_classes;
    }

private:
    int                          _num_classes;
    bool                         _share_location;
    DetectionOutputLayerCodeType _code_type;
    int                          _keep_top_k;
    float                        _nms_threshold;
    int                          _top_k;
    int                          _background_label_id;
    float                        _confidence_threshold;
    bool                         _variance_encoded_in_target;
    float                        _eta;
    int                          _num_loc_classes;
};

/** Detection Output layer info */
class DetectionPostProcessLayerInfo final
{
public:
    /** Default Constructor */
    DetectionPostProcessLayerInfo()
        : _max_detections(),
          _max_classes_per_detection(),
          _nms_score_threshold(),
          _iou_threshold(),
          _num_classes(),
          _scales_values(),
          _use_regular_nms(),
          _detection_per_class(),
          _dequantize_scores()
    {
    }
    /** Constructor
     *
     * @param[in] max_detections            Number of total detection.
     * @param[in] max_classes_per_detection Number of total classes to be kept after NMS step. Used in the Fast Non-Max-Suppression
     * @param[in] nms_score_threshold       Threshold to be used in NMS
     * @param[in] iou_threshold             Threshold to be used during the intersection over union.
     * @param[in] num_classes               Number of classes.
     * @param[in] scales_values             Scales values used for decode center size boxes.
     * @param[in] use_regular_nms           (Optional) Boolean to determinate if use regular or fast nms. Defaults to false.
     * @param[in] detection_per_class       (Optional) Number of detection per class. Used in the Regular Non-Max-Suppression. Defaults to 100.
     * @param[in] dequantize_scores         (Optional) If the scores need to be dequantized. Defaults to true.
     */
    DetectionPostProcessLayerInfo(unsigned int max_detections, unsigned int max_classes_per_detection, float nms_score_threshold, float iou_threshold, unsigned int num_classes,
                                  std::array<float, 4> scales_values, bool use_regular_nms = false, unsigned int detection_per_class = 100, bool dequantize_scores = true)
        : _max_detections(max_detections),
          _max_classes_per_detection(max_classes_per_detection),
          _nms_score_threshold(nms_score_threshold),
          _iou_threshold(iou_threshold),
          _num_classes(num_classes),
          _scales_values(scales_values),
          _use_regular_nms(use_regular_nms),
          _detection_per_class(detection_per_class),
          _dequantize_scores(dequantize_scores)
    {
    }
    /** Get max detections. */
    unsigned int max_detections() const
    {
        return _max_detections;
    }
    /** Get max_classes per detection. Used in the Fast Non-Max-Suppression.*/
    unsigned int max_classes_per_detection() const
    {
        return _max_classes_per_detection;
    }
    /** Get detection per class. Used in the Regular Non-Max-Suppression */
    unsigned int detection_per_class() const
    {
        return _detection_per_class;
    }
    /** Get nms threshold. */
    float nms_score_threshold() const
    {
        return _nms_score_threshold;
    }
    /** Get intersection over union threshold. */
    float iou_threshold() const
    {
        return _iou_threshold;
    }
    /** Get num classes. */
    unsigned int num_classes() const
    {
        return _num_classes;
    }
    /** Get if use regular nms. */
    bool use_regular_nms() const
    {
        return _use_regular_nms;
    }
    /** Get y scale value. */
    float scale_value_y() const
    {
        // Saved as [y,x,h,w]
        return _scales_values[0];
    }
    /** Get x scale value. */
    float scale_value_x() const
    {
        // Saved as [y,x,h,w]
        return _scales_values[1];
    }
    /** Get h scale value. */
    float scale_value_h() const
    {
        // Saved as [y,x,h,w]
        return _scales_values[2];
    }
    /** Get w scale value. */
    float scale_value_w() const
    {
        // Saved as [y,x,h,w]
        return _scales_values[3];
    }
    /** Get dequantize_scores value. */
    bool dequantize_scores() const
    {
        return _dequantize_scores;
    }

private:
    unsigned int _max_detections;
    unsigned int _max_classes_per_detection;
    float        _nms_score_threshold;
    float        _iou_threshold;
    unsigned int _num_classes;
    std::array<float, 4> _scales_values;
    bool         _use_regular_nms;
    unsigned int _detection_per_class;
    bool         _dequantize_scores;
};

/** Pooling Layer Information struct*/
struct PoolingLayerInfo
{
    /** Default Constructor */
    PoolingLayerInfo()
        : pool_type(PoolingType::MAX),
          pool_size(Size2D()),
          data_layout(DataLayout::UNKNOWN),
          pad_stride_info(PadStrideInfo()),
          exclude_padding(false),
          is_global_pooling(false),
          fp_mixed_precision(false),
          use_inf_as_limit(true),
          use_kernel_indices(false)
    {
    }
    /** Constructor
     *
     * @param[in] pool_type          Pooling type @ref PoolingType.
     * @param[in] pool_size          Pooling size, in elements, across  x and y.
     * @param[in] data_layout        Data layout used by the layer @ref DataLayout
     * @param[in] pad_stride_info    (Optional) Padding and stride information @ref PadStrideInfo
     * @param[in] exclude_padding    (Optional) Strategy when accounting padding in calculations.
     *                               True will exclude padding while false will not (Used in AVG/L2 pooling to determine the pooling area).
     *                               Defaults to false;
     * @param[in] fp_mixed_precision (Optional) Use wider accumulators (32 bit instead of 16 for FP16) to improve accuracy.
     * @param[in] use_inf_as_limit   (Optional) Use inf to represent the limits of datatypes range, instead of  using "lowest" property of the data type.
     * @param[in] use_kernel_indices (Optional) Use kernel indices instead of using source indices while computing indices tensor.
     */
    explicit PoolingLayerInfo(PoolingType   pool_type,
                              unsigned int  pool_size,
                              DataLayout    data_layout,
                              PadStrideInfo pad_stride_info    = PadStrideInfo(),
                              bool          exclude_padding    = false,
                              bool          fp_mixed_precision = false,
                              bool          use_inf_as_limit   = true,
                              bool          use_kernel_indices = false)
        : pool_type(pool_type),
          pool_size(Size2D(pool_size, pool_size)),
          data_layout(data_layout),
          pad_stride_info(pad_stride_info),
          exclude_padding(exclude_padding),
          is_global_pooling(false),
          fp_mixed_precision(fp_mixed_precision),
          use_inf_as_limit(use_inf_as_limit),
          use_kernel_indices(use_kernel_indices)
    {
    }

    /** Constructor
     *
     * @param[in] pool_type          Pooling type @ref PoolingType.
     * @param[in] pool_size          Pooling size, in elements, across  x and y.
     * @param[in] data_layout        Data layout used by the layer @ref DataLayout
     * @param[in] pad_stride_info    (Optional) Padding and stride information @ref PadStrideInfo
     * @param[in] exclude_padding    (Optional) Strategy when accounting padding in calculations.
     *                               True will exclude padding while false will not (Used in AVG/L2 pooling to determine the pooling area).
     *                               Defaults to false;
     * @param[in] fp_mixed_precision (Optional) Use wider accumulators (32 bit instead of 16 for FP16) to improve accuracy.
     * @param[in] use_inf_as_limit   (Optional) Use inf to represent the limits of datatypes range, instead of  using "lowest" property of the data type.
     * @param[in] use_kernel_indices (Optional) Use kernel indices instead of using source indices while computing indices tensor.
     */
    explicit PoolingLayerInfo(PoolingType   pool_type,
                              Size2D        pool_size,
                              DataLayout    data_layout,
                              PadStrideInfo pad_stride_info    = PadStrideInfo(),
                              bool          exclude_padding    = false,
                              bool          fp_mixed_precision = false,
                              bool          use_inf_as_limit   = true,
                              bool          use_kernel_indices = false)
        : pool_type(pool_type),
          pool_size(pool_size),
          data_layout(data_layout),
          pad_stride_info(pad_stride_info),
          exclude_padding(exclude_padding),
          is_global_pooling(false),
          fp_mixed_precision(fp_mixed_precision),
          use_inf_as_limit(use_inf_as_limit),
          use_kernel_indices(use_kernel_indices)
    {
    }

    /** Constructor
     *
     * @note This constructor is used for global pooling
     *
     * @param[in] pool_type   Pooling type @ref PoolingType.
     * @param[in] data_layout Data layout used by the layer @ref DataLayout
     */
    explicit PoolingLayerInfo(PoolingType pool_type, DataLayout data_layout)
        : pool_type(pool_type),
          pool_size(Size2D()),
          data_layout(data_layout),
          pad_stride_info(PadStrideInfo(1, 1, 0, 0)),
          exclude_padding(false),
          is_global_pooling(true),
          fp_mixed_precision(false),
          use_inf_as_limit(true),
          use_kernel_indices(false)
    {
    }

    PoolingType   pool_type;
    Size2D        pool_size;
    DataLayout    data_layout;
    PadStrideInfo pad_stride_info;
    bool          exclude_padding;
    bool          is_global_pooling;
    bool          fp_mixed_precision;
    bool          use_inf_as_limit;
    bool          use_kernel_indices;
};

/** Pooling Layer Information struct*/
struct Pooling3dLayerInfo
{
    /** Default Constructor */
    Pooling3dLayerInfo() noexcept
        : pool_type(PoolingType::MAX),
          pool_size(Size3D()),
          stride(Size3D()),
          padding(Padding3D()),
          exclude_padding(false),
          is_global_pooling(false),
          fp_mixed_precision(false),
          round_type(DimensionRoundingType::FLOOR)
    {
    }
    /** Constructor
     *
     * @param[in] pool_type          Pooling type @ref PoolingType.
     * @param[in] pool_size          Pooling size, in elements, across x, y and z.
     * @param[in] stride             (Optional) stride information @ref Size3D
     * @param[in] padding            (Optional) padding information @ref Padding3D
     * @param[in] exclude_padding    (Optional) Strategy when accounting padding in calculations.
     *                               True will exclude padding while false will not (Used in AVG/L2 pooling to determine the pooling area).
     *                               Defaults to false;
     * @param[in] fp_mixed_precision (Optional) Use wider accumulators (32 bit instead of 16 for FP16) to improve accuracy.
     * @param[in] round_type         (Optional) Dimensions rounding. Defaults to @ref DimensionRoundingType::FLOOR
     */
    explicit Pooling3dLayerInfo(PoolingType           pool_type,
                                unsigned int          pool_size,
                                Size3D                stride             = Size3D(1U, 1U, 1U),
                                Padding3D             padding            = Padding3D(),
                                bool                  exclude_padding    = false,
                                bool                  fp_mixed_precision = false,
                                DimensionRoundingType round_type         = DimensionRoundingType::FLOOR)
        : pool_type(pool_type),
          pool_size(Size3D(pool_size, pool_size, pool_size)),
          stride(stride),
          padding(padding),
          exclude_padding(exclude_padding),
          is_global_pooling(false),
          fp_mixed_precision(fp_mixed_precision),
          round_type(round_type)
    {
    }

    /** Constructor
     *
     * @param[in] pool_type          Pooling type @ref PoolingType.
     * @param[in] pool_size          Pooling size, in elements, across  x, y and z.
     * @param[in] stride             (Optional) stride information @ref Size3D
     * @param[in] padding            (Optional) padding information @ref Padding3D
     * @param[in] exclude_padding    (Optional) Strategy when accounting padding in calculations.
     *                               True will exclude padding while false will not (Used in AVG/L2 pooling to determine the pooling area).
     *                               Defaults to false;
     * @param[in] fp_mixed_precision (Optional) Use wider accumulators (32 bit instead of 16 for FP16) to improve accuracy.
     * @param[in] round_type         (Optional) Dimensions rounding. Defaults to @ref DimensionRoundingType::FLOOR
     */
    explicit Pooling3dLayerInfo(PoolingType           pool_type,
                                Size3D                pool_size,
                                Size3D                stride             = Size3D(1U, 1U, 1U),
                                Padding3D             padding            = Padding3D(),
                                bool                  exclude_padding    = false,
                                bool                  fp_mixed_precision = false,
                                DimensionRoundingType round_type         = DimensionRoundingType::FLOOR)
        : pool_type(pool_type),
          pool_size(pool_size),
          stride(stride),
          padding(padding),
          exclude_padding(exclude_padding),
          is_global_pooling(false),
          fp_mixed_precision(fp_mixed_precision),
          round_type(round_type)
    {
    }

    /** Constructor
     *
     * @note This constructor is used for global pooling
     *
     * @param[in] pool_type Pooling type @ref PoolingType.
     */
    explicit Pooling3dLayerInfo(PoolingType pool_type)
        : pool_type(pool_type),
          pool_size(Size3D()),
          stride(Size3D(1U, 1U, 1U)),
          padding(Padding3D(0, 0, 0)),
          exclude_padding(false),
          is_global_pooling(true),
          fp_mixed_precision(false),
          round_type(DimensionRoundingType::FLOOR)
    {
    }

    PoolingType           pool_type;
    Size3D                pool_size;
    Size3D                stride;
    Padding3D             padding;
    bool                  exclude_padding;
    bool                  is_global_pooling;
    bool                  fp_mixed_precision;
    DimensionRoundingType round_type;
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

/** Generate Proposals Information class */
class GenerateProposalsInfo
{
public:
    /** Constructor
     *
     * @param[in] im_width       Width of the original image
     * @param[in] im_height      Height of the original image
     * @param[in] im_scale       Scale applied to the original image
     * @param[in] spatial_scale  (Optional)Scale applied to the feature map. Defaults to 1.0
     * @param[in] pre_nms_topN   (Optional)Number of the best scores to be selected from the transformations. Defaults to 6000.
     * @param[in] post_nms_topN  (Optional)Number of the best scores to be selected from the NMS operation. Defaults to 300.
     * @param[in] nms_thres      (Optional)NMS overlap threshold. Defaults to 0.7.
     * @param[in] min_size       (Optional)Size used to validate the anchors produced. Defaults to 16.
     * @param[in] values_per_roi (Optional)Values used to represent a ROI(Region of interest). Defaults to 4.
     */
    GenerateProposalsInfo(float im_width, float im_height, float im_scale, float spatial_scale = 1.0, int pre_nms_topN = 6000, int post_nms_topN = 300, float nms_thres = 0.7, float min_size = 16.0,
                          size_t values_per_roi = 4)
        : _im_height(im_height), _im_width(im_width), _im_scale(im_scale), _spatial_scale(spatial_scale), _pre_nms_topN(pre_nms_topN), _post_nms_topN(post_nms_topN), _nms_thres(nms_thres),
          _min_size(min_size), _values_per_roi(values_per_roi)
    {
    }

    /* Get the original height */
    float im_height() const
    {
        return _im_height;
    }
    /* Get the original width */
    float im_width() const
    {
        return _im_width;
    }
    /* Get the image scale */
    float im_scale() const
    {
        return _im_scale;
    }
    /* Get the value of how many best scores to select (before NMS) */
    int pre_nms_topN() const
    {
        return _pre_nms_topN;
    }
    /* Get the value of how many best scores to select (after NMS) */
    int post_nms_topN() const
    {
        return _post_nms_topN;
    }
    /* Get the NMS overlap threshold */
    float nms_thres() const
    {
        return _nms_thres;
    }
    /* Get the minimal size */
    float min_size() const
    {
        return _min_size;
    }
    /* Get the spatial scale to be applied to the feature maps */
    float spatial_scale() const
    {
        return _spatial_scale;
    }
    /* Get the values used to represent a ROI(Region of interest)*/
    size_t values_per_roi() const
    {
        return _values_per_roi;
    }

private:
    float  _im_height;
    float  _im_width;
    float  _im_scale;
    float  _spatial_scale;
    int    _pre_nms_topN;
    int    _post_nms_topN;
    float  _nms_thres;
    float  _min_size;
    size_t _values_per_roi;
};

/** ComputeAnchors information class */
class ComputeAnchorsInfo
{
public:
    /** Constructor
     *
     * @param[in] feat_width     Feature map width
     * @param[in] feat_height    Feature map height
     * @param[in] spatial_scale  Feature map scale
     * @param[in] values_per_roi (Optional)Values used to represent a ROI(Region Of Interest). Defaults to 4
     */
    ComputeAnchorsInfo(float feat_width, float feat_height, float spatial_scale, size_t values_per_roi = 4)
        : _feat_height(feat_height),
          _feat_width(feat_width),
          _spatial_scale(spatial_scale),
          _values_per_roi(values_per_roi)
    {
    }

    /* Get the height of the feature map */
    float feat_height() const
    {
        return _feat_height;
    }

    /* Get the width of the feature map */
    float feat_width() const
    {
        return _feat_width;
    }

    /* Get the scale of the feature map */
    float spatial_scale() const
    {
        return _spatial_scale;
    }

    /* Get the values used to represent a ROI(Region Of Interest)*/
    size_t values_per_roi() const
    {
        return _values_per_roi;
    }

private:
    float  _feat_height;
    float  _feat_width;
    float  _spatial_scale;
    size_t _values_per_roi;
};

/** Bounding Box Transform information class */
class BoundingBoxTransformInfo final
{
public:
    /** Constructor
     *
     * @param[in] img_width                Width of the original image
     * @param[in] img_height               Height, of the original image
     * @param[in] scale                    Scale of the original image
     * @param[in] apply_scale              (Optional)Re-apply scaling after transforming the boxes. Defaults to false
     * @param[in] weights                  (Optional)Weights [wx, wy, ww, wh] for the deltas. Defaults to all ones
     * @param[in] correct_transform_coords (Optional)Correct bounding box transform coordinates. Defaults to false
     * @param[in] bbox_xform_clip          (Optional)Minimum bounding box width and height after bounding box transformation in log-space. Defaults to log(1000/16)
     */
    BoundingBoxTransformInfo(float img_width, float img_height, float scale, bool apply_scale = false, const std::array<float, 4> weights = { { 1.f, 1.f, 1.f, 1.f } }, bool correct_transform_coords =
    false,
    float bbox_xform_clip =
        4.135166556742356f)
        : _img_width(img_width), _img_height(img_height), _scale(scale), _apply_scale(apply_scale), _correct_transform_coords(correct_transform_coords), _weights(weights), _bbox_xform_clip(bbox_xform_clip)
    {
    }

    std::array<float, 4> weights() const
    {
        return _weights;
    }

    float bbox_xform_clip() const
    {
        return _bbox_xform_clip;
    }

    float img_height() const
    {
        return _img_height;
    }

    float img_width() const
    {
        return _img_width;
    }

    float scale() const
    {
        return _scale;
    }

    bool apply_scale() const
    {
        return _apply_scale;
    }

    bool correct_transform_coords() const
    {
        return _correct_transform_coords;
    }

private:
    float _img_width;
    float _img_height;
    float _scale;
    bool  _apply_scale;
    bool  _correct_transform_coords;
    std::array<float, 4> _weights;
    float _bbox_xform_clip;
};

/** Normalization Layer Information class */
class NormalizationLayerInfo
{
public:
    /** Default Constructor
     *
     * @param[in] type      The normalization type. Can be @ref NormType::IN_MAP_1D, @ref NormType::IN_MAP_2D or @ref NormType::CROSS_MAP
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
    /** Get the is_scaled value */
    bool is_scaled() const
    {
        return _is_scaled;
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

class StridedSliceLayerInfo
{
public:
    /** Default Constructor
     *
     * @param[in] begin_mask       (Optional) If the ith bit of begin_mask is set, starts[i] is ignored and the fullest possible range in that dimension is used instead.
     * @param[in] end_mask         (Optional) If the ith bit of end_mask is set, ends[i] is ignored and the fullest possible range in that dimension is used instead.
     * @param[in] shrink_axis_mask (Optional) If the ith bit of shrink_axis_mask is set, it implies that the ith specification shrinks the dimensionality by 1.
     */
    StridedSliceLayerInfo(int32_t begin_mask = 0, int32_t end_mask = 0, int32_t shrink_axis_mask = 0)
        : _begin_mask(begin_mask), _end_mask(end_mask), _shrink_axis_mask(shrink_axis_mask)
    {
    }

    /* Get the begin mask value */
    int32_t begin_mask() const
    {
        return _begin_mask;
    }

    /* Get the end mask value */
    int32_t end_mask() const
    {
        return _end_mask;
    }

    /* Get the shrink axis mask value */
    int32_t shrink_axis_mask() const
    {
        return _shrink_axis_mask;
    }

private:
    int32_t _begin_mask;
    int32_t _end_mask;
    int32_t _shrink_axis_mask;
};

// OHWIo<interleave_by>i<block_by>
inline int interleave_by(const WeightFormat wf)
{
    return (static_cast<int>(wf) >> 8) & 0xFFF;
}
inline int block_by(const WeightFormat wf)
{
    return (static_cast<int>(wf) >> 20) & 0xF;
}
inline bool is_fixed_format(const WeightFormat &wf)
{
    return wf != WeightFormat::UNSPECIFIED && wf != WeightFormat::ANY;
}
inline bool is_fixed_format_fast_math(const WeightFormat &wf)
{
    return (static_cast<int>(wf) >> 4) & 0x1;
}

/** Convolution Layer Weights Information class. This class stores the necessary information to compute convolution layer when the weights are already reshaped */
class WeightsInfo
{
public:
    /** Default constructor */
    WeightsInfo()
        : _are_reshaped(false), _kernel_width(0), _kernel_height(0), _num_kernels(0), _retain_internal_weights(false), _weight_format(arm_compute::WeightFormat::UNSPECIFIED)
    {
    }
    /** Constructor
     *
     * @param[in] are_reshaped            True if the weights have been reshaped
     * @param[in] kernel_width            Kernel width.
     * @param[in] kernel_height           Kernel height.
     * @param[in] num_kernels             Number of convolution kernels.
     * @param[in] retain_internal_weights (Optional) True if internal reshaped weights must be retained. Used for reconfiguration purposes. Default is false.
     * @param[in] weight_format           (Optional) arm_gemm:WeightFormat enumeration requested by the user. Default is arm_compute::WeightFormat::UNSPECIFIED.
     */
    WeightsInfo(bool are_reshaped, unsigned int kernel_width, unsigned int kernel_height, unsigned int num_kernels, bool retain_internal_weights = false,
                arm_compute::WeightFormat weight_format = arm_compute::WeightFormat::UNSPECIFIED)
        : _are_reshaped(are_reshaped), _kernel_width(kernel_width), _kernel_height(kernel_height), _num_kernels(num_kernels), _retain_internal_weights(retain_internal_weights), _weight_format(weight_format)
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
    arm_compute::WeightFormat weight_format() const
    {
        return _weight_format;
    }
    void set_weight_format(arm_compute::WeightFormat weight_format)
    {
        _weight_format = weight_format;
    }

    unsigned int kernel_width() const
    {
        return _kernel_width;
    }
    unsigned int kernel_height() const
    {
        return _kernel_height;
    }

private:
    bool                      _are_reshaped;
    unsigned int              _kernel_width;
    unsigned int              _kernel_height;
    unsigned int              _num_kernels;
    bool                      _retain_internal_weights;
    arm_compute::WeightFormat _weight_format;
};

/** GEMM reshape information class. This class stores the necessary information about matrix A and matrix B reshape.
 *
 * The matrix A can only be reshaped through @ref opencl::kernels::ClGemmReshapeLhsMatrixKernel or  @ref cpu::kernels::CpuGemmInterleave4x4Kernel
 * Note: Optionally just for @ref opencl::kernels::ClGemmReshapeLhsMatrixKernel is it possible to set mult_interleave4x4_height, the multiplication factor for the height of the 4x4 interleaved block
 *
 * The matrix B can only be reshaped through @ref opencl::kernels::ClGemmReshapeRhsMatrixKernel or  @ref cpu::kernels::CpuGemmTranspose1xWKernel
 * Note: Optionally just for @ref opencl::kernels::ClGemmReshapeRhsMatrixKernel is it possible to set mult_transpose1xW_width, the multiplication factor for the width of the 1xW transposed block
 *
 */
class GEMMReshapeInfo final
{
public:
    /** Default constructor */
    GEMMReshapeInfo()
        : _m(1), _n(1), _k(1), _mult_transpose1xW_width(1), _mult_interleave4x4_height(1), _depth_output_gemm3d(0), _reinterpret_input_as_3d(false), _broadcast_bias(false)
    {
    }
    /** Constructor
     *
     * @param[in] m                         Number of matrix A rows
     * @param[in] n                         Number of matrix B columns
     * @param[in] k                         Number of matrix A columns or matrix B rows
     * @param[in] mult_transpose1xW_width   (Optional) Multiplication factor for the width of the 1xW transposed block
     * @param[in] mult_interleave4x4_height (Optional) Multiplication factor for the height of the 4x4 interleaved block
     * @param[in] depth_output_gemm3d       (Optional) Depth (third dimension) of the output tensor to be used with the GEMM3D kernel.
     *                                      If 0 the output will not be reinterpreted as 3D. Default 0
     * @param[in] reinterpret_input_as_3d   (Optional) Reinterpret the input as 3D tensor. (i.e. this flag should be set to true when GEMM is used
     *                                      to perform 1x1 convolutions with the NHWC data layout)
     * @param[in] broadcast_bias            (Optional) Broadcast the shape of the bias tensor from a vector to a matrix.
     */
    GEMMReshapeInfo(int m, int n, int k, int mult_transpose1xW_width = 1, int mult_interleave4x4_height = 1, int depth_output_gemm3d = 0, bool reinterpret_input_as_3d = false, bool broadcast_bias = false)
        : _m(m), _n(n), _k(k), _mult_transpose1xW_width(mult_transpose1xW_width), _mult_interleave4x4_height(mult_interleave4x4_height), _depth_output_gemm3d(depth_output_gemm3d),
          _reinterpret_input_as_3d(reinterpret_input_as_3d), _broadcast_bias(broadcast_bias)
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
    /** Flag which specifies whether to broadcast the shape of the bias tensor.
     *
     * @return True if the shape of the bias tensor is to be broadcasted.
     */
    bool broadcast_bias() const
    {
        return _broadcast_bias;
    };

private:
    int  _m;
    int  _n;
    int  _k;
    int  _mult_transpose1xW_width;
    int  _mult_interleave4x4_height;
    int  _depth_output_gemm3d;
    bool _reinterpret_input_as_3d;
    bool _broadcast_bias;
};

/** GEMM LHS (Left Hand Side) matrix information */
struct GEMMLHSMatrixInfo
{
    GEMMLHSMatrixInfo() = default;
    GEMMLHSMatrixInfo(unsigned int m, unsigned int k, unsigned int v, bool trans, bool inter)
        : m0(m), k0(k), v0(v), transpose(trans), interleave(inter)
    {
    }
    unsigned int m0{ 1 };            /**< Number of rows processed by the matrix multiplication */
    unsigned int k0{ 1 };            /**< Number of partial accumulations performed by the matrix multiplication */
    unsigned int v0{ 1 };            /**< Number of vertical blocks of size (m0xk0) stored on the same output row */
    bool         transpose{ true };  /**< True if the (m0xk0) block has to be transposed before been stored */
    bool         interleave{ true }; /**< True if the v0 (m0xk0) blocks have to be interleaved in the output row */
};

/** GEMM RHS (Right Hand Side) matrix information */
struct GEMMRHSMatrixInfo
{
    GEMMRHSMatrixInfo() = default;
    GEMMRHSMatrixInfo(unsigned int n, unsigned int k, unsigned int h, bool trans, bool inter, bool export_to_cl_img)
        : n0(n), k0(k), h0(h), transpose(trans), interleave(inter), export_to_cl_image(export_to_cl_img)
    {
    }
    unsigned int n0{ 1 };                     /**< Number of columns processed by the matrix multiplication */
    unsigned int k0{ 1 };                     /**< Number of partial accumulations performed by the matrix multiplication */
    unsigned int h0{ 1 };                     /**< Number of horizontal blocks of size (k0xn0) stored on the same output row */
    bool         transpose{ true };           /**< True if the (k0xn0) block has to be transposed before been stored */
    bool         interleave{ true };          /**< True if the h0 (k0xn0) blocks have to be interleaved in the output row */
    bool         export_to_cl_image{ false }; /**< True if the reshaped rhs has to be exported to cl_image. n0 must be equal to 4 */
};

class ITensorInfo;

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

/** Class for holding information related to cropping */
using CropInfo = Padding2D;
} // namespace arm_compute
#endif /* ACL_ARM_COMPUTE_CORE_TYPES */
