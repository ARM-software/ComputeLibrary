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
/** Structure holding all the input tensor graph parameters */
struct TensorParams
{
    int              width{ 1 };
    int              height{ 1 };
    int              fm{ 1 };
    int              batch{ 1 };
    QuantizationInfo quant_info{ 1.0f, 0 };
    uint64_t         range_low{ 0 };
    uint64_t         range_high{ 16 };
};
/** Structure holding all the verification graph parameters */
struct VerificationParams
{
    float absolute_tolerance{ -1.f };
    float relative_tolerance{ -1.f };
    float tolerance_number{ -1.f };
};

/** Structure holding all the common graph parameters */
struct FrameworkParams
{
    bool                       help{ false };
    int                        threads{ 0 };
    arm_compute::graph::Target target{ arm_compute::graph::Target::NEON };
};

/** Structure holding all the fully_connected layer graph parameters */
struct FullyConnectedParams
{
    arm_compute::DataType   data_type{ DataType::F32 };
    arm_compute::DataLayout data_layout{ DataLayout::NCHW };
    FullyConnectedLayerInfo info{};
    int                     num_outputs{ 1 };
};

/** Structure holding all the graph Example parameters */
struct ExampleParams
{
    FrameworkParams      common_params{};
    TensorParams         input{};
    TensorParams         weights{};
    TensorParams         output{};
    VerificationParams   verification{};
    FullyConnectedParams fully_connected{};
};

/** Formatted output of the fully_connectedParams type
 *
 * @param[out] os            Output stream.
 * @param[in]  common_params fully_connected parameters to output
 *
 * @return Modified output stream.
 */
::std::ostream &operator<<(::std::ostream &os, const ExampleParams &common_params)
{
    std::string false_str = std::string("false");
    std::string true_str  = std::string("true");

    os << "Threads : " << common_params.common_params.threads << std::endl;
    os << "Target : " << common_params.common_params.target << std::endl;
    os << "Data type : " << common_params.fully_connected.data_type << std::endl;
    os << "Input dimensions(X,Y, Channels, Batch) : (" << common_params.input.width << "," << common_params.input.height << "," << common_params.input.fm << "," << common_params.input.batch << ")"
       << std::endl;
    os << "Number of outputs : " << common_params.fully_connected.num_outputs << std::endl;
    return os;
}

/** fully_connected command line options used to configure the graph examples
 *
 * (Similar to common options)
 * The options in this object get populated when "parse()" is called on the parser used to construct it.
 * The expected workflow is:
 *
 * CommandLineParser parser;
 * CommonOptions options( parser );
 * parser.parse(argc, argv);
 */
class FullyConnectedOptions final
{
public:
    explicit FullyConnectedOptions(CommandLineParser &parser) noexcept
        : width(parser.add_option<SimpleOption<int>>("width", 3)),
          batch(parser.add_option<SimpleOption<int>>("batch", 1)),
          help(parser.add_option<ToggleOption>("help")),
          threads(parser.add_option<SimpleOption<int>>("threads")),
          target(),
          data_type(),
          absolute_tolerance(parser.add_option<SimpleOption<float>>("abs_tolerance", -1.0f)),
          relative_tolerance(parser.add_option<SimpleOption<float>>("rel_tolerance", -1.0f)),
          tolerance_number(parser.add_option<SimpleOption<float>>("tolerance_num", -1.0f)),
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
        const std::set<arm_compute::graph::Target> supported_targets
        {
            Target::NEON,
            Target::CL,
            Target::GC,
        };

        const std::set<arm_compute::DataType> supported_data_types
        {
            DataType::F16,
            DataType::F32,
            DataType::QASYMM8,
        };

        target    = parser.add_option<EnumOption<Target>>("target", supported_targets, Target::NEON);
        data_type = parser.add_option<EnumOption<DataType>>("type", supported_data_types, DataType::F32);

        target->set_help("Target to execute on");
        data_type->set_help("Data type to use");
        help->set_help("Show this help message");
        width->set_help("Set Input dimension width");
        batch->set_help("Set Input dimension batch");
        absolute_tolerance->set_help("Absolute tolerance used for verification");
        relative_tolerance->set_help("Absolute tolerance used for verification");
        tolerance_number->set_help("Absolute tolerance used for verification");
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

    /** Prevent instances of this class from being copied (As this class contains pointers) */
    FullyConnectedOptions(const FullyConnectedOptions &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    FullyConnectedOptions &operator=(const FullyConnectedOptions &) = delete;
    /** Allow instances of this class to be moved */
    FullyConnectedOptions(FullyConnectedOptions &&) noexcept(true) = default;
    /** Allow instances of this class to be moved */
    FullyConnectedOptions &operator=(FullyConnectedOptions &&) noexcept(true) = default;
    /** Default destructor */
    ~FullyConnectedOptions() = default;

    SimpleOption<int>                      *width;              /**< Input width */
    SimpleOption<int>                      *batch;              /**< Input batch */
    ToggleOption                           *help;               /**< show help message */
    SimpleOption<int>                      *threads;            /**< Number of threads option */
    EnumOption<arm_compute::graph::Target> *target;             /**< Graph execution target */
    EnumOption<arm_compute::DataType>      *data_type;          /**< Graph data type */
    SimpleOption<float>                    *absolute_tolerance; /**< Absolute tolerance used in verification */
    SimpleOption<float>                    *relative_tolerance; /**< Relative tolerance used in verification */
    SimpleOption<float>                    *tolerance_number;   /**< Tolerance number used in verification */
    SimpleOption<float>                    *input_scale;        /**< Input Quantization scale from QASSYMM8 */
    SimpleOption<int>                      *input_offset;       /**< Input Quantization offset from QASSYMM8 */
    SimpleOption<float>                    *weights_scale;      /**< Weights Quantization scale from QASSYMM8 */
    SimpleOption<int>                      *weights_offset;     /**< Weights Quantization offset from QASSYMM8 */
    SimpleOption<float>                    *output_scale;       /**< Output Quantization scale from QASSYMM8 */
    SimpleOption<int>                      *output_offset;      /**< Output Quantization offset from QASSYMM8 */
    SimpleOption<int>                      *num_outputs;        /**< Number of outputs. */
    SimpleOption<uint64_t>                 *input_range_low;    /**< Lower bound for input randomization range */
    SimpleOption<uint64_t>                 *input_range_high;   /**< Upper bound for input randomization range */
    SimpleOption<uint64_t>                 *weights_range_low;  /**< Lower bound for weights randomization range */
    SimpleOption<uint64_t>                 *weights_range_high; /**< Upper bound for weights randomization range */
};

/** Consumes the fully_connected graph options and creates a structure containing any information
 *
 * @param[in] options Options to consume
 *
 * @return fully_connectedparams structure containing the common graph parameters
 */
ExampleParams consume_fully_connected_graph_parameters(FullyConnectedOptions &options)
{
    ExampleParams common_params;

    common_params.common_params.help    = options.help->is_set() ? options.help->value() : false;
    common_params.common_params.threads = options.threads->value();
    common_params.common_params.target  = options.target->value();

    common_params.input.width             = options.width->value();
    common_params.input.batch             = options.batch->value();
    common_params.input.quant_info.scale  = options.input_scale->value();
    common_params.input.quant_info.offset = options.input_offset->value();
    common_params.input.range_low         = options.input_range_low->value();
    common_params.input.range_high        = options.input_range_high->value();

    common_params.weights.quant_info.scale  = options.weights_scale->value();
    common_params.weights.quant_info.offset = options.weights_offset->value();
    common_params.weights.range_low         = options.weights_range_low->value();
    common_params.weights.range_high        = options.weights_range_high->value();

    common_params.output.quant_info.scale  = options.output_scale->value();
    common_params.output.quant_info.offset = options.output_offset->value();

    common_params.fully_connected.data_type   = options.data_type->value();
    common_params.fully_connected.num_outputs = options.num_outputs->value();

    common_params.verification.absolute_tolerance = options.absolute_tolerance->value();
    common_params.verification.relative_tolerance = options.relative_tolerance->value();
    common_params.verification.tolerance_number   = options.tolerance_number->value();

    return common_params;
}

/** fully_connectedLayer Graph example validation accessor class */
template <typename D>
class FullyConnectedVerifyAccessor final : public graph::ITensorAccessor
{
public:
    using TBias = typename std::conditional<std::is_same<typename std::decay<D>::type, uint8_t>::value, int32_t, D>::type;

    /** Constructor
     *
     * @param[in] params fully_connected parameters
     */
    explicit FullyConnectedVerifyAccessor(ExampleParams &params)
        : _params(params)
    {
    }

    // Inherited methods overridden:
    bool access_tensor(ITensor &tensor) override
    {
        const RelativeTolerance<float> rel_tolerance(relative_tolenace(_params.verification.relative_tolerance));  /**< Relative tolerance */
        const AbsoluteTolerance<float> abs_tolerance(absolute_tolerance(_params.verification.absolute_tolerance)); /**< Absolute tolerance */
        const float                    tolerance_num(tolerance_number(_params.verification.tolerance_number));     /**< Tolerance number */

        // Calculate Tensor shapes for verification
        const TensorShape      input_shape        = TensorShape(_params.input.width, _params.input.height, _params.input.fm, _params.input.batch);
        const TensorDescriptor input_descriptor   = TensorDescriptor(input_shape, _params.fully_connected.data_type, _params.input.quant_info);
        const TensorDescriptor weights_descriptor = FullyConnectedLayerNode::compute_weights_descriptor(input_descriptor,
                                                                                                        _params.fully_connected.num_outputs,
                                                                                                        _params.fully_connected.info,
                                                                                                        _params.weights.quant_info);
        const TensorDescriptor output_desciptor = FullyConnectedLayerNode::compute_output_descriptor(input_descriptor, _params.fully_connected.num_outputs, _params.output.quant_info);

        //Create Input tensors
        SimpleTensor<D>     src{ input_descriptor.shape, _params.fully_connected.data_type, 1, input_descriptor.quant_info };
        SimpleTensor<D>     weights{ weights_descriptor.shape, _params.fully_connected.data_type, 1, weights_descriptor.quant_info };
        SimpleTensor<TBias> bias{ TensorShape(tensor.info()->tensor_shape().x()), _params.fully_connected.data_type, 1, _params.input.quant_info };

        //Fill the tensors with random values
        fill_tensor<D>(src, 0, static_cast<D>(_params.input.range_low), static_cast<D>(_params.input.range_high));
        fill_tensor<D>(weights, 1, static_cast<D>(_params.weights.range_low), static_cast<D>(_params.weights.range_high));
        fill_tensor<TBias>(bias, 2, static_cast<TBias>(_params.input.range_low), static_cast<TBias>(_params.input.range_high));

        //Calculate reference
        SimpleTensor<D> output = reference::fully_connected_layer<D>(src, weights, bias, output_desciptor.shape, _params.output.quant_info);

        arm_compute::test::validation::validate(Accessor(tensor), output, rel_tolerance, tolerance_num, abs_tolerance);

        return false;
    }

private:
    /** Fill tensor with Random values.
     *
     * Validate the given tensor against the reference result.
     *
     * @param[out] tensor The tensor we want to file
     * @param[in]  seed   seed for the randomization function
     * @param[in]  low    lower bound for random values
     * @param[in]  high   upper bound for random values
     *
     * @return None.
     */
    template <typename T>
    void fill_tensor(arm_compute::test::SimpleTensor<T> &tensor, std::random_device::result_type seed, T low, T high)
    {
        std::mt19937 gen(seed);
        switch(tensor.data_type())
        {
            case arm_compute::DataType::QASYMM8:
            {
                const uint8_t qasymm8_low  = tensor.quantization_info().quantize(low, RoundingPolicy::TO_NEAREST_UP);
                const uint8_t qasymm8_high = tensor.quantization_info().quantize(high, RoundingPolicy::TO_NEAREST_UP);

                std::uniform_int_distribution<uint8_t> distribution(qasymm8_low, qasymm8_high);

                for(int i = 0; i < tensor.num_elements(); ++i)
                {
                    tensor[i] = tensor.quantization_info().quantize(distribution(gen), RoundingPolicy::TO_NEAREST_UP);
                }

                break;
            }
            case arm_compute::DataType::S32:
            {
                std::uniform_int_distribution<int32_t> distribution(static_cast<int32_t>(low), static_cast<uint32_t>(high));

                for(int i = 0; i < tensor.num_elements(); ++i)
                {
                    tensor[i] = distribution(gen);
                }

                break;
            }

            case arm_compute::DataType::F16:
            {
                std::uniform_real_distribution<float> distribution(static_cast<half>(low), static_cast<half>(high));

                for(int i = 0; i < tensor.num_elements(); ++i)
                {
                    tensor[i] = static_cast<half>(distribution(gen));
                }
                break;
            }
            case arm_compute::DataType::F32:
            {
                std::uniform_real_distribution<float> distribution(static_cast<float>(low), static_cast<float>(high));

                for(int i = 0; i < tensor.num_elements(); ++i)
                {
                    tensor[i] = distribution(gen);
                }

                break;
            }
            default:
                ARM_COMPUTE_ERROR("NOT SUPPORTED!");
        }
    }
    /** Select relative tolerance.
     *
     * Select relative tolerance if not supplied by user.
     *
     * @param[in] user_value supplied relative tolerance. -1 designates no user input
     *
     * @return Appropriate relative tolerance.
     */
    float relative_tolenace(float user_value)
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
        if(user_value == -1)
        {
            return relative_tolerance.at(_params.common_params.target).at(_params.fully_connected.data_type);
        }

        return user_value;
    }

    /** Select absolute tolerance.
     *
     * Select absolute tolerance if not supplied by user.
     *
     * @param[in] user_value supplied absolute tolerance. -1 designates no user input
     *
     * @return Appropriate absolute tolerance.
     */
    float absolute_tolerance(float user_value)
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

        if(user_value == -1)
        {
            return absolute_tolerance.at(_params.common_params.target).at(_params.fully_connected.data_type);
        }
        return user_value;
    }
    /** Select tolerance number.
     *
     * Select tolerance number if not supplied by user.
     *
     * @param[in] user_value supplied tolerance number. -1 designates no user input
     *
     * @return Appropriate tolerance number.
     */
    float tolerance_number(float user_value)
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

        if(user_value == -1)
        {
            return absolute_tolerance.at(_params.common_params.target).at(_params.fully_connected.data_type);
        }
        return user_value;
    }

    ExampleParams _params;
};

/** Generates appropriate fully_connected verify accessor
 *
 * @param[in] params User supplied parameters for fully_connected.
 *
 * @return A fully_connected verify accessor for the requested datatype.
 */
inline std::unique_ptr<graph::ITensorAccessor> get_fully_connected_verify_accessor(ExampleParams params)
{
    switch(params.fully_connected.data_type)
    {
        case DataType::QASYMM8:
        {
            return arm_compute::support::cpp14::make_unique<FullyConnectedVerifyAccessor<uint8_t>>(
                       params);
        }
        case DataType::F16:
        {
            return arm_compute::support::cpp14::make_unique<FullyConnectedVerifyAccessor<half>>(
                       params);
        }
        case DataType::F32:
        {
            return arm_compute::support::cpp14::make_unique<FullyConnectedVerifyAccessor<float>>(
                       params);
        }
        default:
            ARM_COMPUTE_ERROR("NOT SUPPORTED!");
    }
}

} // namespace

class Graphfully_connectedValidateExample final : public ValidateExample
{
public:
    Graphfully_connectedValidateExample()
        : graph(0, "fully_connected Graph example")
    {
    }
    bool do_setup(int argc, char **argv) override
    {
        CommandLineParser parser;

        FullyConnectedOptions Options(parser);

        parser.parse(argc, argv);

        ExampleParams params = consume_fully_connected_graph_parameters(Options);

        if(params.common_params.help)
        {
            parser.print_help(argv[0]);
            return false;
        }

        std::cout << params << std::endl;

        // Create input descriptor
        const TensorShape      input_shape      = TensorShape(params.input.width, params.input.height, params.input.fm, params.input.batch);
        const TensorDescriptor input_descriptor = TensorDescriptor(input_shape, params.fully_connected.data_type, params.input.quant_info, params.fully_connected.data_layout);

        const PixelValue lower = PixelValue(params.input.range_low, params.fully_connected.data_type, params.input.quant_info);
        const PixelValue upper = PixelValue(params.input.range_high, params.fully_connected.data_type, params.input.quant_info);

        const PixelValue weights_lower = PixelValue(params.weights.range_low, params.fully_connected.data_type, params.weights.quant_info);
        const PixelValue weights_upper = PixelValue(params.weights.range_high, params.fully_connected.data_type, params.weights.quant_info);

        graph << params.common_params.target
              << InputLayer(input_descriptor, get_random_accessor(lower, upper, 0))
              << FullyConnectedLayer(params.fully_connected.num_outputs,
                                     get_random_accessor(weights_lower, weights_upper, 1),
                                     get_random_accessor(lower, upper, 2),
                                     params.fully_connected.info, params.weights.quant_info, params.output.quant_info)
              << OutputLayer(get_fully_connected_verify_accessor(params));

        GraphConfig config;
        config.num_threads = params.common_params.threads;

        graph.finalize(params.common_params.target, config);

        return true;
    }

    void do_run() override
    {
        graph.run();
    }

    void do_teardown() override
    {
    }

private:
    Stream graph;
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
    return arm_compute::utils::run_example<Graphfully_connectedValidateExample>(argc, argv);
}
