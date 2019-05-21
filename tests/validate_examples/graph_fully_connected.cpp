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
#include "tests/validation/reference/FullyConnectedLayer.h"
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
/** Fully connected command line options used to configure the graph examples
 *
 * (Similar to common options)
 * The options in this object get populated when "parse()" is called on the parser used to construct it.
 * The expected workflow is:
 *
 * CommandLineParser parser;
 * CommonOptions options( parser );
 * parser.parse(argc, argv);
 */
class FullyConnectedOptions final : public CommonGraphValidateOptions
{
public:
    explicit FullyConnectedOptions(CommandLineParser &parser) noexcept
        : CommonGraphValidateOptions(parser),
          width(parser.add_option<SimpleOption<int>>("width", 3)),
          batch(parser.add_option<SimpleOption<int>>("batch", 1)),
          input_scale(parser.add_option<SimpleOption<float>>("input_scale", 1.0f)),
          input_offset(parser.add_option<SimpleOption<int>>("input_offset", 0)),
          weights_scale(parser.add_option<SimpleOption<float>>("weights_scale", 1.0f)),
          weights_offset(parser.add_option<SimpleOption<int>>("weights_offset", 0)),
          output_scale(parser.add_option<SimpleOption<float>>("output_scale", 1.0f)),
          output_offset(parser.add_option<SimpleOption<int>>("output_offset", 0)),
          num_outputs(parser.add_option<SimpleOption<int>>("num_outputs", 1)),
          input_range_low(parser.add_option<SimpleOption<uint64_t>>("input_range_low")),
          input_range_high(parser.add_option<SimpleOption<uint64_t>>("input_range_high")),
          weights_range_low(parser.add_option<SimpleOption<uint64_t>>("weights_range_low")),
          weights_range_high(parser.add_option<SimpleOption<uint64_t>>("weights_range_high"))
    {
        width->set_help("Set Input dimension width");
        batch->set_help("Set Input dimension batch");
        input_scale->set_help("Quantization scale from QASYMM8");
        input_offset->set_help("Quantization offset from QASYMM8");
        weights_scale->set_help("Quantization scale from QASYMM8");
        weights_offset->set_help("Quantization offset from QASYMM8");
        output_scale->set_help("Quantization scale from QASYMM8");
        output_offset->set_help("Quantization offset from QASYMM8");
        num_outputs->set_help("Number of outputs.");
        input_range_low->set_help("Lower bound for input randomization range");
        input_range_high->set_help("Lower bound for input randomization range");
        weights_range_low->set_help("Lower bound for input randomization range");
        weights_range_high->set_help("Lower bound for input randomization range");
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
        common_params.input.batch      = batch->value();
        common_params.input.quant_info = QuantizationInfo(input_scale->value(), input_offset->value());
        common_params.input.range_low  = input_range_low->value();
        common_params.input.range_high = input_range_high->value();

        common_params.weights.quant_info = QuantizationInfo(weights_scale->value(), weights_offset->value());
        common_params.weights.range_low  = weights_range_low->value();
        common_params.weights.range_high = weights_range_high->value();

        common_params.output.quant_info = QuantizationInfo(output_scale->value(), output_offset->value());

        common_params.data_type                   = data_type->value();
        common_params.fully_connected.num_outputs = num_outputs->value();
    }

    void print_parameters(::std::ostream &os, const ExampleParams &common_params) override
    {
        os << "Threads : " << common_params.common_params.threads << std::endl;
        os << "Target : " << common_params.common_params.target << std::endl;
        os << "Data type : " << common_params.data_type << std::endl;
        os << "Input dimensions(X,Y, Channels, Batch) : (" << common_params.input.width << "," << common_params.input.height << "," << common_params.input.fm << "," << common_params.input.batch << ")"
           << std::endl;
        os << "Number of outputs : " << common_params.fully_connected.num_outputs << std::endl;
    }

    /** Prevent instances of this class from being copied (As this class contains pointers) */
    FullyConnectedOptions(const FullyConnectedOptions &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    FullyConnectedOptions &operator=(const FullyConnectedOptions &) = delete;
    /** Allow instances of this class to be moved */
    FullyConnectedOptions(FullyConnectedOptions &&) noexcept(true) = default;
    /** Allow instances of this class to be moved */
    FullyConnectedOptions &operator=(FullyConnectedOptions &&) noexcept(true) = default;
    /** Default destructor */
    ~FullyConnectedOptions() override = default;

private:
    SimpleOption<int>      *width;              /**< Input width */
    SimpleOption<int>      *batch;              /**< Input batch */
    SimpleOption<float>    *input_scale;        /**< Input Quantization scale from QASSYMM8 */
    SimpleOption<int>      *input_offset;       /**< Input Quantization offset from QASSYMM8 */
    SimpleOption<float>    *weights_scale;      /**< Weights Quantization scale from QASSYMM8 */
    SimpleOption<int>      *weights_offset;     /**< Weights Quantization offset from QASSYMM8 */
    SimpleOption<float>    *output_scale;       /**< Output Quantization scale from QASSYMM8 */
    SimpleOption<int>      *output_offset;      /**< Output Quantization offset from QASSYMM8 */
    SimpleOption<int>      *num_outputs;        /**< Number of outputs. */
    SimpleOption<uint64_t> *input_range_low;    /**< Lower bound for input randomization range */
    SimpleOption<uint64_t> *input_range_high;   /**< Upper bound for input randomization range */
    SimpleOption<uint64_t> *weights_range_low;  /**< Lower bound for weights randomization range */
    SimpleOption<uint64_t> *weights_range_high; /**< Upper bound for weights randomization range */
};

/** Fully Connected Layer Graph example validation accessor class */
template <typename D>
class FullyConnectedVerifyAccessor final : public VerifyAccessor<D>
{
    using BaseClassType = VerifyAccessor<D>;
    using BaseClassType::BaseClassType;
    using BaseClassType::_params;
    using TBias = typename std::conditional<std::is_same<typename std::decay<D>::type, uint8_t>::value, int32_t, D>::type;

    // Inherited methods overriden:
    void create_tensors(arm_compute::test::SimpleTensor<D>     &src,
                        arm_compute::test::SimpleTensor<D>     &weights,
                        arm_compute::test::SimpleTensor<TBias> &bias,
                        ITensor                                &tensor) override
    {
        // Calculate Tensor shapes for verification
        const TensorShape      input_shape        = TensorShape(_params.input.width, _params.input.height, _params.input.fm, _params.input.batch);
        const TensorDescriptor input_descriptor   = TensorDescriptor(input_shape, _params.data_type, _params.input.quant_info);
        const TensorDescriptor weights_descriptor = FullyConnectedLayerNode::compute_weights_descriptor(input_descriptor,
                                                                                                        _params.fully_connected.num_outputs,
                                                                                                        _params.fully_connected.info,
                                                                                                        _params.weights.quant_info);
        const TensorDescriptor output_desciptor = FullyConnectedLayerNode::compute_output_descriptor(input_descriptor, _params.fully_connected.num_outputs, _params.output.quant_info);

        //Create Input tensors
        src     = SimpleTensor<D> { input_descriptor.shape, _params.data_type, 1, input_descriptor.quant_info };
        weights = SimpleTensor<D> { weights_descriptor.shape, _params.data_type, 1, weights_descriptor.quant_info };
        bias    = SimpleTensor<TBias> { TensorShape(tensor.info()->tensor_shape().x()), _params.data_type, 1, _params.input.quant_info };
    }

    TensorShape output_shape(ITensor &tensor) override
    {
        ARM_COMPUTE_UNUSED(tensor);

        const TensorShape      input_shape      = TensorShape(_params.input.width, _params.input.height, _params.input.fm, _params.input.batch);
        const TensorDescriptor input_descriptor = TensorDescriptor(input_shape, _params.data_type, _params.input.quant_info);
        const TensorDescriptor output_desciptor = FullyConnectedLayerNode::compute_output_descriptor(input_descriptor, _params.fully_connected.num_outputs, _params.output.quant_info);

        return output_desciptor.shape;
    }

    arm_compute::test::SimpleTensor<D> reference(arm_compute::test::SimpleTensor<D>     &src,
                                                 arm_compute::test::SimpleTensor<D>     &weights,
                                                 arm_compute::test::SimpleTensor<TBias> &bias,
                                                 const arm_compute::TensorShape         &output_shape) override
    {
        return reference::fully_connected_layer<D>(src, weights, bias, output_shape, _params.output.quant_info);
    }

    float relative_tolerance() override
    {
        const std::map<arm_compute::graph::Target, const std::map<DataType, float>> relative_tolerance
        {
            {
                arm_compute::graph::Target::CL,
                {   { DataType::F16, 0.2f },
                    { DataType::F32, 0.05f },
                    { DataType::QASYMM8, 1.0f }
                }
            },
            {
                arm_compute::graph::Target::NEON,
                {   { DataType::F16, 0.2f },
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
                    { DataType::F32, 0.0001f },
                    { DataType::QASYMM8, 1.0f }
                }
            },
            {
                Target::NEON,
                {   { DataType::F16, 0.3f },
                    { DataType::F32, 0.1f },
                    { DataType::QASYMM8, 1.0f }
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
                {   { DataType::F16, 0.07f },
                    { DataType::F32, 0.07f },
                    { DataType::QASYMM8, 0.0f }
                }
            },
            {
                Target::NEON,
                {   { DataType::F16, 0.07f },
                    { DataType::F32, 0.0f },
                    { DataType::QASYMM8, 0.0f }
                }
            }
        };

        return absolute_tolerance.at(_params.common_params.target).at(_params.data_type);
    }
};

} // namespace

class GraphFullyConnectedValidateExample final : public GraphValidateExample<FullyConnectedLayer, FullyConnectedOptions, FullyConnectedVerifyAccessor>
{
    using GraphValidateExample::graph;

public:
    GraphFullyConnectedValidateExample()
        : GraphValidateExample("Fully_connected Graph example")
    {
    }

    FullyConnectedLayer GraphFunctionLayer(ExampleParams &params) override
    {
        const PixelValue lower = PixelValue(params.input.range_low, params.data_type, params.input.quant_info);
        const PixelValue upper = PixelValue(params.input.range_high, params.data_type, params.input.quant_info);

        const PixelValue weights_lower = PixelValue(params.weights.range_low, params.data_type, params.weights.quant_info);
        const PixelValue weights_upper = PixelValue(params.weights.range_high, params.data_type, params.weights.quant_info);

        return FullyConnectedLayer(params.fully_connected.num_outputs,
                                   get_random_accessor(weights_lower, weights_upper, 1),
                                   get_random_accessor(lower, upper, 2),
                                   params.fully_connected.info, params.weights.quant_info, params.output.quant_info);
    }
};

/** Main program for Graph fully_connected test
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments ( Input dimensions [width, batch]
 *                             Fully connected  [num_outputs,type]
 *                             Verification[tolerance_number,absolute_tolerance,relative_tolerance] )
 *
 */
int main(int argc, char **argv)
{
    return arm_compute::utils::run_example<GraphFullyConnectedValidateExample>(argc, argv);
}
