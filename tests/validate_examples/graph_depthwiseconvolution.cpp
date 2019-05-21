/*
 * Copyright (c) 2019 ARM Limited.
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
#include "arm_compute/graph.h"

#include "support/ToolchainSupport.h"

#include "tests/NEON/Accessor.h"
#include "tests/validation/Validation.h"
#include "tests/validation/reference/DepthwiseConvolutionLayer.h"
#include "tests/validation/reference/Permute.h"

#include "utils/CommonGraphOptions.h"
#include "utils/GraphUtils.h"
#include "utils/Utils.h"

#include "ValidateExample.h"
#include "graph_validate_utils.h"

#include <utility>

using namespace arm_compute::utils;
using namespace arm_compute::graph::frontend;
using namespace arm_compute::graph_utils;
using namespace arm_compute::graph;
using namespace arm_compute;
using namespace arm_compute::test;
using namespace arm_compute::test::validation;

namespace
{
/** Depthwise Convolution command line options used to configure the graph examples
 *
 * (Similar to common options)
 * The options in this object get populated when "parse()" is called on the parser used to construct it.
 * The expected workflow is:
 *
 * CommandLineParser parser;
 * CommonOptions options( parser );
 * parser.parse(argc, argv);
 */
class DepthConvolutionOptions final : public CommonGraphValidateOptions
{
public:
    explicit DepthConvolutionOptions(CommandLineParser &parser) noexcept
        : CommonGraphValidateOptions(parser),
          width(parser.add_option<SimpleOption<int>>("width", 9)),
          height(parser.add_option<SimpleOption<int>>("height", 9)),
          channels(parser.add_option<SimpleOption<int>>("channels", 1)),
          batch(parser.add_option<SimpleOption<int>>("batch", 1)),
          weights_width(parser.add_option<SimpleOption<int>>("weights_width", 3)),
          weights_height(parser.add_option<SimpleOption<int>>("weights_height", 3)),
          padding_top(parser.add_option<SimpleOption<int>>("padding_top", 0)),
          padding_left(parser.add_option<SimpleOption<int>>("padding_left", 0)),
          padding_bottom(parser.add_option<SimpleOption<int>>("padding_bottom", 0)),
          padding_right(parser.add_option<SimpleOption<int>>("padding_right", 0)),
          stride_x(parser.add_option<SimpleOption<int>>("stride_x", 1)),
          stride_y(parser.add_option<SimpleOption<int>>("stride_y", 1)),
          padding_mode(),
          conv_mode(),
          depth_multiplier(parser.add_option<SimpleOption<int>>("depth_multiplier", 1)),
          data_layout(),
          scale(parser.add_option<SimpleOption<float>>("scale", 1.0f)),
          offset(parser.add_option<SimpleOption<int>>("offset", 0)),
          weights_scale(parser.add_option<SimpleOption<float>>("weights_scale", 1.0f)),
          weights_offset(parser.add_option<SimpleOption<int>>("weights_offset", 0)),
          output_scale(parser.add_option<SimpleOption<float>>("output_scale", 1.0f)),
          output_offset(parser.add_option<SimpleOption<int>>("output_offset", 0)),
          input_range_low(parser.add_option<SimpleOption<uint64_t>>("input_range_low")),
          input_range_high(parser.add_option<SimpleOption<uint64_t>>("input_range_high")),
          weights_range_low(parser.add_option<SimpleOption<uint64_t>>("weights_range_low")),
          weights_range_high(parser.add_option<SimpleOption<uint64_t>>("weights_range_high")),
          input_npy(parser.add_option<SimpleOption<std::string>>("input_image")),
          output_npy(parser.add_option<SimpleOption<std::string>>("reference_image")),
          weights_npy(parser.add_option<SimpleOption<std::string>>("weights_npy")),
          bias_npy(parser.add_option<SimpleOption<std::string>>("bias_image"))
    {
        const std::set<ConvolutionPaddingMode> available_padding_modes
        {
            ConvolutionPaddingMode::Valid,
            ConvolutionPaddingMode::Same
        };

        const std::set<arm_compute::graph::DepthwiseConvolutionMethod> supported_convolution_methods
        {
            arm_compute::graph::DepthwiseConvolutionMethod::Default,
            arm_compute::graph::DepthwiseConvolutionMethod::GEMV,
            arm_compute::graph::DepthwiseConvolutionMethod::Optimized3x3,
        };

        const std::set<DataLayout> supported_data_layouts
        {
            DataLayout::NHWC,
            DataLayout::NCHW,
        };

        padding_mode = parser.add_option<EnumOption<ConvolutionPaddingMode>>("padding_mode", available_padding_modes, ConvolutionPaddingMode::Valid);
        conv_mode    = parser.add_option<EnumOption<arm_compute::graph::DepthwiseConvolutionMethod>>("convolution_method", supported_convolution_methods,
                                                                                                     arm_compute::graph::DepthwiseConvolutionMethod::Default);
        data_layout = parser.add_option<EnumOption<DataLayout>>("layout", supported_data_layouts, DataLayout::NHWC);

        padding_mode->set_help("Set padding mode");
        width->set_help("Set Input dimension width");
        height->set_help("Set Input dimension height");
        channels->set_help("Set Input dimension channels");
        batch->set_help("Set Input dimension batch");
        weights_width->set_help("Set weights_dimensions width");
        weights_height->set_help("Set weights_dimensions height");
        padding_top->set_help("Set padding top");
        padding_bottom->set_help("Set padding bottom");
        padding_left->set_help("Set padding left");
        padding_right->set_help("Set padding right");
        stride_x->set_help("Set padding stride x");
        stride_y->set_help("Set padding stride y");
        conv_mode->set_help("Set convolution method");
        data_layout->set_help("Data layout to use");
        scale->set_help("Quantization scale from QASYMM8");
        offset->set_help("Quantization offset from QASYMM8");
        output_scale->set_help("Quantization scale from QASYMM8");
        output_offset->set_help("Quantization offset from QASYMM8");
        input_npy->set_help("Use input .npy instead");
        output_npy->set_help("Use .npy as a reference");
        input_range_low->set_help("Lower bound for input randomization range");
        input_range_high->set_help("Lower bound for input randomization range");
        weights_scale->set_help("Quantization scale from QASYMM8");
        weights_offset->set_help("Quantization offset from QASYMM8");
        weights_range_low->set_help("Lower bound for input randomization range");
        weights_range_high->set_help("Lower bound for input randomization range");
        depth_multiplier->set_help("Depth multiplier");
    }

    /** Fill out the supplied parameters with user supplied parameters
     *
     * @param[out] os            Output stream.
     * @param[in]  common_params Example parameters to output
     *
     * @return None.
     */
    void consume_parameters(ExampleParams &common_params)
    {
        common_params.input.width      = width->value();
        common_params.input.height     = height->value();
        common_params.input.fm         = channels->value();
        common_params.input.batch      = batch->value();
        common_params.input.quant_info = QuantizationInfo(scale->value(), offset->value());
        common_params.input.npy        = input_npy->value();
        common_params.input.range_low  = input_range_low->value();
        common_params.input.range_high = input_range_high->value();

        common_params.weights.width      = weights_width->value();
        common_params.weights.height     = weights_height->value();
        common_params.weights.npy        = weights_npy->value();
        common_params.weights.range_low  = weights_range_low->value();
        common_params.weights.range_high = weights_range_high->value();
        common_params.weights.quant_info = QuantizationInfo(weights_scale->value(), weights_offset->value());

        common_params.bias.npy = bias_npy->value();

        common_params.output.quant_info = QuantizationInfo(output_scale->value(), output_offset->value());
        common_params.output.npy        = output_npy->value();

        common_params.convolution.padding_mode     = padding_mode->value();
        common_params.convolution.padding_top      = padding_top->value();
        common_params.convolution.padding_bottom   = padding_bottom->value();
        common_params.convolution.padding_left     = padding_left->value();
        common_params.convolution.padding_right    = padding_right->value();
        common_params.convolution.padding_stride_x = stride_x->value();
        common_params.convolution.padding_stride_y = stride_y->value();
        common_params.convolution.depth_multiplier = depth_multiplier->value();

        common_params.data_type                = data_type->value();
        common_params.data_layout              = data_layout->value();
        common_params.depth_convolution_method = conv_mode->value();
    }

    void print_parameters(::std::ostream &os, const ExampleParams &common_params) override
    {
        os << "Threads : " << common_params.common_params.threads << std::endl;
        os << "Target : " << common_params.common_params.target << std::endl;
        os << "Data type : " << common_params.data_type << std::endl;
        os << "Input dimensions(X,Y, Channels, Batch) : (" << common_params.input.width << "," << common_params.input.height << "," << common_params.input.fm << "," << common_params.input.batch << ")"
           << std::endl;
        os << "Weight dimensions(X,Y, Channels(same as input)) : (" << common_params.weights.width << "," << common_params.weights.height << "," << common_params.input.fm << ","
           << ")" << std::endl;
        os << "Padding(top, bottom, left, right) (stride x, stride y) : (" << common_params.convolution.padding_top << "," << common_params.convolution.padding_bottom << "," <<
           common_params.convolution.padding_left << "," << common_params.convolution.padding_right << ") (" << common_params.convolution.padding_stride_x << "," << common_params.convolution.padding_stride_y <<
           ")" << std::endl;
        os << "Padding Mode: " << common_params.convolution.padding_mode << std::endl;
        os << "Convolution Method: " << common_params.depth_convolution_method << std::endl;
        os << "Depth multiplier: " << common_params.convolution.depth_multiplier;
    }

    /** Prevent instances of this class from being copied (As this class contains pointers) */
    DepthConvolutionOptions(const DepthConvolutionOptions &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    DepthConvolutionOptions &operator=(const DepthConvolutionOptions &) = delete;
    /** Allow instances of this class to be moved */
    DepthConvolutionOptions(DepthConvolutionOptions &&) noexcept(true) = default;
    /** Allow instances of this class to be moved */
    DepthConvolutionOptions &operator=(DepthConvolutionOptions &&) noexcept(true) = default;
    /** Default destructor */
    ~DepthConvolutionOptions() override = default;

private:
    SimpleOption<int>                                          *width;              /**< Input width */
    SimpleOption<int>                                          *height;             /**< Input height */
    SimpleOption<int>                                          *channels;           /**< Input channels */
    SimpleOption<int>                                          *batch;              /**< Input batch */
    SimpleOption<int>                                          *weights_width;      /**< weights width */
    SimpleOption<int>                                          *weights_height;     /**< weights height */
    SimpleOption<int>                                          *padding_top;        /**< Padding top */
    SimpleOption<int>                                          *padding_left;       /**< Padding left */
    SimpleOption<int>                                          *padding_bottom;     /**< Padding bottom */
    SimpleOption<int>                                          *padding_right;      /**< Padding right */
    SimpleOption<int>                                          *stride_x;           /**< Padding stride x */
    SimpleOption<int>                                          *stride_y;           /**< Padding stride y */
    EnumOption<ConvolutionPaddingMode>                         *padding_mode;       /**< Padding mode */
    EnumOption<arm_compute::graph::DepthwiseConvolutionMethod> *conv_mode;          /**< Convolution method */
    SimpleOption<int>                                          *depth_multiplier;   /**< Depth multiplier */
    EnumOption<arm_compute::DataLayout>                        *data_layout;        /**< Graph data layout */
    SimpleOption<float>                                        *scale;              /**< Input Quantization scale from QASYMM8 */
    SimpleOption<int>                                          *offset;             /**< Input Quantization offset from QASYMM8 */
    SimpleOption<float>                                        *weights_scale;      /**< Weights Quantization scale from QASYMM8 */
    SimpleOption<int>                                          *weights_offset;     /**< Weights Quantization offset from QASYMM8 */
    SimpleOption<float>                                        *output_scale;       /**< Output Quantization scale from QASYMM8 */
    SimpleOption<int>                                          *output_offset;      /**< Output Quantization offset from QASYMM8 */
    SimpleOption<uint64_t>                                     *input_range_low;    /**< Lower bound for input randomization range */
    SimpleOption<uint64_t>                                     *input_range_high;   /**< Upper bound for input randomization range */
    SimpleOption<uint64_t>                                     *weights_range_low;  /**< Lower bound for weights randomization range */
    SimpleOption<uint64_t>                                     *weights_range_high; /**< Upper bound for weights randomization range */

    SimpleOption<std::string> *input_npy;   /**< Use input .npy image */
    SimpleOption<std::string> *output_npy;  /**< Use output .npy image to verify*/
    SimpleOption<std::string> *weights_npy; /**< Use weights .npy image */
    SimpleOption<std::string> *bias_npy;    /**< Use bias .npy image */
};

/** DepthwiseConvolutionLayer Graph example validation accessor class */
template <typename D>
class DepthConvolutionVerifyAccessor final : public VerifyAccessor<D>
{
public:
    using BaseClassType = VerifyAccessor<D>;
    using BaseClassType::BaseClassType;
    using BaseClassType::_params;
    using TBias = typename std::conditional<std::is_same<typename std::decay<D>::type, uint8_t>::value, int32_t, D>::type;

public:
    SimpleTensor<D> reference(SimpleTensor<D> &src, SimpleTensor<D> &weights, SimpleTensor<TBias> &bias, const TensorShape &output_shape) override
    {
        // Calculate padding information
        const PadStrideInfo padding_info = calculate_convolution_padding(_params);

        //Calculate reference
        return reference::depthwise_convolution<D>(src, weights, bias, output_shape, padding_info,
                                                   _params.convolution.depth_multiplier,
                                                   Size2D(1U, 1U),
                                                   _params.output.quant_info);
    }

    float relative_tolerance() override
    {
        const std::map<arm_compute::graph::Target, const std::map<DataType, float>> relative_tolerance
        {
            {
                arm_compute::graph::Target::CL,
                {   { DataType::F16, 0.01f },
                    { DataType::F32, 0.01f },
                    { DataType::QASYMM8, 0.0f }
                }
            },
            {
                arm_compute::graph::Target::NEON,
                {   { DataType::F16, 0.01f },
                    { DataType::F32, 0.01f },
                    { DataType::QASYMM8, 1.0f }
                }
            }
        };

        return relative_tolerance.at(_params.common_params.target).at(_params.data_type);
    }

    float absolute_tolerance() override
    {
        const std::map<Target, const std::map<DataType, float>> absolute_tolerance
        {
            {
                Target::CL,
                {   { DataType::F16, 0.0f },
                    { DataType::F32, 0.0000f },
                    { DataType::QASYMM8, 0.0f }
                }
            },
            {
                Target::NEON,
                {   { DataType::F16, 0.2f },
                    { DataType::F32, 0.002f },
                    { DataType::QASYMM8, 0.0f }
                }
            }
        };

        return absolute_tolerance.at(_params.common_params.target).at(_params.data_type);
    }

    float tolerance_number() override
    {
        const std::map<Target, const std::map<DataType, float>> absolute_tolerance
        {
            {
                Target::CL,
                {   { DataType::F16, 0.05f },
                    { DataType::F32, 0.00f },
                    { DataType::QASYMM8, 0.0f }
                }
            },
            {
                Target::NEON,
                {   { DataType::F16, 0.05f },
                    { DataType::F32, 0.0f },
                    { DataType::QASYMM8, 0.0f }
                }
            }
        };

        return absolute_tolerance.at(_params.common_params.target).at(_params.data_type);
    }
};

} // namespace

class GraphDepthwiseConvolutionValidateExample final : public GraphValidateExample<DepthwiseConvolutionLayer, DepthConvolutionOptions, DepthConvolutionVerifyAccessor>
{
    using GraphValidateExample::graph;

public:
    GraphDepthwiseConvolutionValidateExample()
        : GraphValidateExample("DepthWiseConvolution Graph example")
    {
    }

    DepthwiseConvolutionLayer GraphFunctionLayer(ExampleParams &params) override
    {
        const PixelValue lower = PixelValue(params.input.range_low, params.data_type, params.input.quant_info);
        const PixelValue upper = PixelValue(params.input.range_high, params.data_type, params.input.quant_info);

        const PixelValue weights_lower = PixelValue(params.weights.range_low, params.data_type, params.weights.quant_info);
        const PixelValue weights_upper = PixelValue(params.weights.range_high, params.data_type, params.weights.quant_info);

        // Calculate padding information
        const PadStrideInfo padding_info = calculate_convolution_padding(params);

        return DepthwiseConvolutionLayer(params.weights.width, params.weights.height,
                                         get_accessor(params.weights, weights_lower, weights_upper, 1),
                                         get_accessor(params.bias, lower, upper, 2),
                                         padding_info, params.convolution.depth_multiplier, params.weights.quant_info, params.output.quant_info);
    }
};

/** Main program for Graph Depthwise Convolution test
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments ( Input dimensions [width, height, channels, batch]
 *                             Weights dimensions [width, height, channels]
 *                             Padding [top,bottom,left,right, Stride x, Stride y, mode [Valid / Same / Manual] )
 *                             Convolution Method[ Default/GEMV/Optimized3x3]
 *                             Verification[tolerance_number,absolute_tolerance,relative_tolerance] )
 *
 */
int main(int argc, char **argv)
{
    return arm_compute::utils::run_example<GraphDepthwiseConvolutionValidateExample>(argc, argv);
}
