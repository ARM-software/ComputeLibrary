/*
 * Copyright (c) 2019-2020 ARM Limited.
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

#ifndef GRAPH_VALIDATE_UTILS_H
#define GRAPH_VALIDATE_UTILS_H

#include "arm_compute/graph.h"

#include "ValidateExample.h"
#include "utils/command_line/CommandLineParser.h"

namespace arm_compute
{
namespace utils
{
/*Available Padding modes */
enum class ConvolutionPaddingMode
{
    Valid,
    Same,
    Manual
};

/** Stream Input operator for the ConvolutionPaddingMode type
 *
 * @param[in]  stream Input stream.
 * @param[out] Mode   Convolution parameters to output
 *
 * @return input stream.
 */
inline ::std::istream &operator>>(::std::istream &stream, ConvolutionPaddingMode &Mode)
{
    static const std::map<std::string, ConvolutionPaddingMode> modes =
    {
        { "valid", ConvolutionPaddingMode::Valid },
        { "same", ConvolutionPaddingMode::Same },
        { "manual", ConvolutionPaddingMode::Manual }
    };
    std::string value;
    stream >> value;
#ifndef ARM_COMPUTE_EXCEPTIONS_DISABLED
    try
    {
#endif /* ARM_COMPUTE_EXCEPTIONS_DISABLED */
        Mode = modes.at(arm_compute::utility::tolower(value));
#ifndef ARM_COMPUTE_EXCEPTIONS_DISABLED
    }
    catch(const std::out_of_range &)
    {
        throw std::invalid_argument(value);
    }
#endif /* ARM_COMPUTE_EXCEPTIONS_DISABLED */

    return stream;
}

/** Formatted output of the ConvolutionPaddingMode type
 *
 * @param[out] os   Output stream.
 * @param[in]  Mode ConvolutionPaddingMode to output
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, ConvolutionPaddingMode Mode)
{
    switch(Mode)
    {
        case ConvolutionPaddingMode::Valid:
            os << "Valid";
            break;
        case ConvolutionPaddingMode::Same:
            os << "Same";
            break;
        case ConvolutionPaddingMode::Manual:
            os << "Manual";
            break;
        default:
            throw std::invalid_argument("Unsupported padding mode format");
    }

    return os;
}

/** Structure holding all the input tensor graph parameters */
struct TensorParams
{
    int              width{ 1 };
    int              height{ 1 };
    int              fm{ 1 };
    int              batch{ 1 };
    QuantizationInfo quant_info{ 1.0f, 0 };
    std::string      npy{};
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

/** Structure holding all the graph Example parameters */
struct CommonParams
{
    FrameworkParams       common_params{};
    TensorParams          input{};
    TensorParams          weights{};
    TensorParams          bias{};
    TensorParams          output{};
    VerificationParams    verification{};
    arm_compute::DataType data_type{ DataType::F32 };
};

/** Structure holding all the Convolution layer graph parameters */
struct ConvolutionParams
{
    int depth_multiplier{ 1 };
    /** Padding graph parameters */
    int                    padding_top{ 0 };
    int                    padding_bottom{ 0 };
    int                    padding_left{ 0 };
    int                    padding_right{ 0 };
    int                    padding_stride_x{ 0 };
    int                    padding_stride_y{ 0 };
    ConvolutionPaddingMode padding_mode{ ConvolutionPaddingMode::Valid };
    struct
    {
        struct
        {
            int X{ 0 };
            int Y{ 0 };
        } stride{};
        ConvolutionPaddingMode mode{ ConvolutionPaddingMode::Valid };
    } padding{};
};

/** Structure holding all the fully_connected layer graph parameters */
struct FullyConnectedParams
{
    FullyConnectedLayerInfo info{};
    int                     num_outputs{ 1 };
};

/** Structure holding all the graph Example parameters */
struct ExampleParams : public CommonParams
{
    FullyConnectedParams                           fully_connected{};
    ConvolutionParams                              convolution{};
    arm_compute::graph::DepthwiseConvolutionMethod depth_convolution_method{ arm_compute::graph::DepthwiseConvolutionMethod::Default };
    arm_compute::graph::ConvolutionMethod          convolution_method{ arm_compute::graph::ConvolutionMethod::Default };
    arm_compute::DataLayout                        data_layout{ DataLayout::NCHW };
};

/** Calculate stride information.
 *
 * Depending on the selected padding mode create the desired PadStrideInfo
 *
 * @param[in] params Convolution parameters supplied by the user.
 *
 * @return PadStrideInfo with the correct padding mode.
 */
inline PadStrideInfo calculate_convolution_padding(ExampleParams params)
{
    switch(params.convolution.padding_mode)
    {
        case ConvolutionPaddingMode::Manual:
        {
            return PadStrideInfo(params.convolution.padding_stride_x, params.convolution.padding_stride_y, params.convolution.padding_left, params.convolution.padding_right, params.convolution.padding_top,
                                 params.convolution.padding_bottom, DimensionRoundingType::FLOOR);
        }
        case ConvolutionPaddingMode::Valid:
        {
            return PadStrideInfo();
        }
        case ConvolutionPaddingMode::Same:
        {
            return arm_compute::calculate_same_pad(TensorShape(params.input.width, params.input.height), TensorShape(params.weights.width, params.weights.height),
                                                   PadStrideInfo(params.convolution.padding_stride_x,
                                                                 params.convolution.padding_stride_y));
        }
        default:
            ARM_COMPUTE_ERROR("NOT SUPPORTED!");
    }
}
/** CommonGraphValidateOptions command line options used to configure the graph examples
 *
 * (Similar to common options)
 * The options in this object get populated when "parse()" is called on the parser used to construct it.
 * The expected workflow is:
 *
 * CommandLineParser parser;
 * CommonOptions options( parser );
 * parser.parse(argc, argv);
 */
class CommonGraphValidateOptions
{
public:
    explicit CommonGraphValidateOptions(CommandLineParser &parser) noexcept
        : help(parser.add_option<ToggleOption>("help")),
          threads(parser.add_option<SimpleOption<int>>("threads")),
          target(),
          data_type(),
          absolute_tolerance(parser.add_option<SimpleOption<float>>("abs_tolerance", -1.0f)),
          relative_tolerance(parser.add_option<SimpleOption<float>>("rel_tolerance", -1.0f)),
          tolerance_number(parser.add_option<SimpleOption<float>>("tolerance_num", -1.0f))
    {
        const std::set<arm_compute::graph::Target> supported_targets
        {
            arm_compute::graph::Target::NEON,
            arm_compute::graph::Target::CL,
            arm_compute::graph::Target::GC,
        };

        const std::set<arm_compute::DataType> supported_data_types
        {
            DataType::F16,
            DataType::F32,
            DataType::QASYMM8,
        };

        target    = parser.add_option<EnumOption<arm_compute::graph::Target>>("target", supported_targets, arm_compute::graph::Target::NEON);
        data_type = parser.add_option<EnumOption<DataType>>("type", supported_data_types, DataType::F32);

        target->set_help("Target to execute on");
        data_type->set_help("Data type to use");
        help->set_help("Show this help message");
        absolute_tolerance->set_help("Absolute tolerance used for verification");
        relative_tolerance->set_help("Absolute tolerance used for verification");
        tolerance_number->set_help("Absolute tolerance used for verification");
    }

    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CommonGraphValidateOptions(const CommonGraphValidateOptions &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CommonGraphValidateOptions &operator=(const CommonGraphValidateOptions &) = delete;
    /** Allow instances of this class to be moved */
    CommonGraphValidateOptions(CommonGraphValidateOptions &&) noexcept(true) = default;
    /** Allow instances of this class to be moved */
    CommonGraphValidateOptions &operator=(CommonGraphValidateOptions &&) noexcept(true) = default;
    /** Default destructor */
    virtual ~CommonGraphValidateOptions() = default;

    void consume_common_parameters(CommonParams &common_params)
    {
        common_params.common_params.help    = help->is_set() ? help->value() : false;
        common_params.common_params.threads = threads->value();
        common_params.common_params.target  = target->value();

        common_params.verification.absolute_tolerance = absolute_tolerance->value();
        common_params.verification.relative_tolerance = relative_tolerance->value();
        common_params.verification.tolerance_number   = tolerance_number->value();
    }

    /** Formatted output of the ExampleParams type
     *
     * @param[out] os            Output stream.
     * @param[in]  common_params Example parameters to output
     *
     * @return None.
     */
    virtual void print_parameters(::std::ostream &os, const ExampleParams &common_params)
    {
        os << "Threads : " << common_params.common_params.threads << std::endl;
        os << "Target : " << common_params.common_params.target << std::endl;
        os << "Data type : " << common_params.data_type << std::endl;
    }

    ToggleOption                           *help;               /**< show help message */
    SimpleOption<int>                      *threads;            /**< Number of threads option */
    EnumOption<arm_compute::graph::Target> *target;             /**< Graph execution target */
    EnumOption<arm_compute::DataType>      *data_type;          /**< Graph data type */
    SimpleOption<float>                    *absolute_tolerance; /**< Absolute tolerance used in verification */
    SimpleOption<float>                    *relative_tolerance; /**< Relative tolerance used in verification */
    SimpleOption<float>                    *tolerance_number;   /**< Tolerance number used in verification */
};

/** Consumes the consume_common_graph_parameters graph options and creates a structure containing any information
 *
 * @param[in]  options       Options to consume
 * @param[out] common_params params structure to consume.
 *
 * @return consume_common_graph_parameters structure containing the common graph parameters
 */
void consume_common_graph_parameters(CommonGraphValidateOptions &options, CommonParams &common_params)
{
    common_params.common_params.help    = options.help->is_set() ? options.help->value() : false;
    common_params.common_params.threads = options.threads->value();
    common_params.common_params.target  = options.target->value();

    common_params.verification.absolute_tolerance = options.absolute_tolerance->value();
    common_params.verification.relative_tolerance = options.relative_tolerance->value();
    common_params.verification.tolerance_number   = options.tolerance_number->value();
}

/** Generates appropriate accessor according to the specified graph parameters
 *
 * @param[in] tensor Tensor parameters
 * @param[in] lower  Lower random values bound
 * @param[in] upper  Upper random values bound
 * @param[in] seed   Random generator seed
 *
 * @return An appropriate tensor accessor
 */
inline std::unique_ptr<graph::ITensorAccessor> get_accessor(const TensorParams &tensor, PixelValue lower, PixelValue upper, const std::random_device::result_type seed = 0)
{
    if(!tensor.npy.empty())
    {
        return arm_compute::support::cpp14::make_unique<arm_compute::graph_utils::NumPyBinLoader>(tensor.npy);
    }
    else
    {
        return arm_compute::support::cpp14::make_unique<arm_compute::graph_utils::RandomAccessor>(lower, upper, seed);
    }
}

/** Graph example validation accessor class */
template <typename D>
class VerifyAccessor : public graph::ITensorAccessor
{
public:
    using TBias = typename std::conditional<std::is_same<typename std::decay<D>::type, uint8_t>::value, int32_t, D>::type;
    /** Constructor
     *
     * @param[in] params Convolution parameters
     */
    explicit VerifyAccessor(ExampleParams &params)
        : _params(std::move(params))
    {
    }
    // Inherited methods overriden:
    bool access_tensor(ITensor &tensor) override
    {
        if(_params.output.npy.empty())
        {
            arm_compute::test::SimpleTensor<D>     src;
            arm_compute::test::SimpleTensor<D>     weights;
            arm_compute::test::SimpleTensor<TBias> bias;

            //Create Input tensors
            create_tensors(src, weights, bias, tensor);

            //Fill the tensors with random values
            fill_tensor(src, 0, static_cast<D>(_params.input.range_low), static_cast<D>(_params.input.range_high));
            fill_tensor(weights, 1, static_cast<D>(_params.weights.range_low), static_cast<D>(_params.weights.range_high));
            fill_tensor(bias, 2, static_cast<TBias>(_params.input.range_low), static_cast<TBias>(_params.input.range_high));

            arm_compute::test::SimpleTensor<D> output = reference(src, weights, bias, output_shape(tensor));

            validate(tensor, output);
        }
        else
        {
            //The user provided a reference file use an npy accessor to validate
            arm_compute::graph_utils::NumPyAccessor(_params.output.npy, tensor.info()->tensor_shape(), tensor.info()->data_type()).access_tensor(tensor);
        }
        return false;
    }

    /** Create reference tensors.
     *
     * Validate the given tensor against the reference result.
     *
     * @param[out] src     The tensor with the source data.
     * @param[out] weights The tensor with the weigths data.
     * @param[out] bias    The tensor with the bias data.
     * @param[in]  tensor  Tensor result of the actual operation passed into the Accessor.
     *
     * @return None.
     */
    virtual void create_tensors(arm_compute::test::SimpleTensor<D>     &src,
                                arm_compute::test::SimpleTensor<D>     &weights,
                                arm_compute::test::SimpleTensor<TBias> &bias,
                                ITensor                                &tensor)
    {
        ARM_COMPUTE_UNUSED(tensor);
        //Create Input tensors
        src     = arm_compute::test::SimpleTensor<D> { TensorShape(_params.input.width, _params.input.height, _params.input.fm, _params.input.batch), _params.data_type, 1, _params.input.quant_info };
        weights = arm_compute::test::SimpleTensor<D> { TensorShape(_params.weights.width, _params.weights.height, _params.weights.fm), _params.data_type, 1, _params.weights.quant_info };
        bias    = arm_compute::test::SimpleTensor<TBias> { TensorShape(_params.input.height), _params.data_type, 1, _params.input.quant_info };
    }

    /** Calculate reference output tensor shape.
     *
     * @param[in] tensor Tensor result of the actual operation passed into the Accessor.
     *
     * @return output tensor shape.
     */
    virtual TensorShape output_shape(ITensor &tensor)
    {
        return arm_compute::graph_utils::permute_shape(tensor.info()->tensor_shape(), _params.data_layout, DataLayout::NCHW);
    }

    /** Calculate reference tensor.
     *
     * Validate the given tensor against the reference result.
     *
     * @param[in] src          The tensor with the source data.
     * @param[in] weights      The tensor with the weigths data.
     * @param[in] bias         The tensor with the bias data.
     * @param[in] output_shape Shape of the output tensor.
     *
     * @return Tensor with the reference output.
     */
    virtual arm_compute::test::SimpleTensor<D> reference(arm_compute::test::SimpleTensor<D>     &src,
                                                         arm_compute::test::SimpleTensor<D>     &weights,
                                                         arm_compute::test::SimpleTensor<TBias> &bias,
                                                         const arm_compute::TensorShape         &output_shape) = 0;

    /** Fill QASYMM tensor with Random values.
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
    void fill_tensor(arm_compute::test::SimpleTensor<uint8_t> &tensor, std::random_device::result_type seed, uint8_t low, uint8_t high)
    {
        ARM_COMPUTE_ERROR_ON(tensor.data_type() != arm_compute::DataType::QASYMM8);

        const UniformQuantizationInfo qinfo = tensor.quantization_info().uniform();

        uint8_t qasymm8_low  = quantize_qasymm8(low, qinfo);
        uint8_t qasymm8_high = quantize_qasymm8(high, qinfo);

        std::mt19937                           gen(seed);
        std::uniform_int_distribution<uint8_t> distribution(qasymm8_low, qasymm8_high);

        for(int i = 0; i < tensor.num_elements(); ++i)
        {
            tensor[i] = quantize_qasymm8(distribution(gen), qinfo);
        }
    }
    /** Fill S32 tensor with Random values.
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
    void fill_tensor(arm_compute::test::SimpleTensor<int32_t> &tensor, std::random_device::result_type seed, int32_t low, int32_t high)
    {
        std::mt19937                           gen(seed);
        std::uniform_int_distribution<int32_t> distribution(static_cast<int32_t>(low), static_cast<uint32_t>(high));

        for(int i = 0; i < tensor.num_elements(); ++i)
        {
            tensor[i] = distribution(gen);
        }
    }
    /** Fill F32 tensor with Random values.
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
    void fill_tensor(arm_compute::test::SimpleTensor<float> &tensor, std::random_device::result_type seed, float low, float high)
    {
        ARM_COMPUTE_ERROR_ON(tensor.data_type() != arm_compute::DataType::F32);
        std::mt19937                          gen(seed);
        std::uniform_real_distribution<float> distribution(low, high);

        for(int i = 0; i < tensor.num_elements(); ++i)
        {
            tensor[i] = distribution(gen);
        }
    }
    /** Fill F16 tensor with Random values.
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
    void fill_tensor(arm_compute::test::SimpleTensor<half> &tensor, std::random_device::result_type seed, half low, half high)
    {
        ARM_COMPUTE_ERROR_ON(tensor.data_type() != arm_compute::DataType::F16);
        std::mt19937                          gen(seed);
        std::uniform_real_distribution<float> distribution(static_cast<half>(low), static_cast<half>(high));

        for(int i = 0; i < tensor.num_elements(); ++i)
        {
            tensor[i] = static_cast<half>(distribution(gen));
        }
    }

    /** Select relative tolerance.
     *
     * Select relative tolerance if not supplied by user.
     *
     * @return Appropriate relative tolerance.
     */
    virtual float relative_tolerance() = 0;

    /** Select absolute tolerance.
     *
     * Select absolute tolerance if not supplied by user.
     *
     * @return Appropriate absolute tolerance.
     */
    virtual float absolute_tolerance() = 0;

    /** Select tolerance number.
     *
     * Select tolerance number if not supplied by user.
     *
     * @return Appropriate tolerance number.
     */
    virtual float tolerance_number() = 0;

    /** Validate the output versus the reference.
     *
     * @param[in] tensor Tensor result of the actual operation passed into the Accessor.
     * @param[in] output Tensor result of the reference implementation.
     *
     * @return None.
     */
    void validate(ITensor &tensor, arm_compute::test::SimpleTensor<D> output)
    {
        float user_relative_tolerance = _params.verification.relative_tolerance;
        float user_absolute_tolerance = _params.verification.absolute_tolerance;
        float user_tolerance_num      = _params.verification.tolerance_number;
        /* If no user input was provided override with defaults. */
        if(user_relative_tolerance == -1)
        {
            user_relative_tolerance = relative_tolerance();
        }

        if(user_absolute_tolerance == -1)
        {
            user_absolute_tolerance = absolute_tolerance();
        }

        if(user_tolerance_num == -1)
        {
            user_tolerance_num = tolerance_number();
        }

        const arm_compute::test::validation::RelativeTolerance<float> rel_tolerance(user_relative_tolerance); /**< Relative tolerance */
        const arm_compute::test::validation::AbsoluteTolerance<float> abs_tolerance(user_absolute_tolerance); /**< Absolute tolerance */
        const float                                                   tolerance_num(user_tolerance_num);      /**< Tolerance number */

        arm_compute::test::validation::validate(arm_compute::test::Accessor(tensor), output, rel_tolerance, tolerance_num, abs_tolerance);
    }

    ExampleParams _params;
};

/** Generates appropriate convolution verify accessor
 *
 * @param[in] params User supplied parameters for convolution.
 *
 * @return A convolution verify accessor for the requested datatype.
 */
template <template <typename D> class VerifyAccessorT>
inline std::unique_ptr<graph::ITensorAccessor> get_verify_accessor(ExampleParams params)
{
    switch(params.data_type)
    {
        case DataType::QASYMM8:
        {
            return arm_compute::support::cpp14::make_unique<VerifyAccessorT<uint8_t>>(
                       params);
        }
        case DataType::F16:
        {
            return arm_compute::support::cpp14::make_unique<VerifyAccessorT<half>>(
                       params);
        }
        case DataType::F32:
        {
            return arm_compute::support::cpp14::make_unique<VerifyAccessorT<float>>(
                       params);
        }
        default:
            ARM_COMPUTE_ERROR("NOT SUPPORTED!");
    }
}

template <typename LayerT, typename OptionsT, template <typename D> class VerifyAccessorT>
class GraphValidateExample : public ValidateExample
{
public:
    GraphValidateExample(std::string name)
        : graph(0, name)
    {
    }

    virtual LayerT GraphFunctionLayer(ExampleParams &params) = 0;

    bool do_setup(int argc, char **argv) override
    {
        CommandLineParser parser;

        OptionsT Options(parser);

        parser.parse(argc, argv);

        ExampleParams params;

        Options.consume_common_parameters(params);
        Options.consume_parameters(params);

        if(params.common_params.help)
        {
            parser.print_help(argv[0]);
            return false;
        }

        Options.print_parameters(std::cout, params);
        // Create input descriptor
        const TensorShape input_shape = arm_compute::graph_utils::permute_shape(TensorShape(params.input.width, params.input.height, params.input.fm, params.input.batch),
                                                                                DataLayout::NCHW, params.data_layout);
        arm_compute::graph::TensorDescriptor input_descriptor = arm_compute::graph::TensorDescriptor(input_shape, params.data_type, params.input.quant_info, params.data_layout);

        const PixelValue lower = PixelValue(params.input.range_low, params.data_type, params.input.quant_info);
        const PixelValue upper = PixelValue(params.input.range_high, params.data_type, params.input.quant_info);

        graph << params.common_params.target
              << params.convolution_method
              << params.depth_convolution_method
              << arm_compute::graph::frontend::InputLayer(input_descriptor, get_accessor(params.input, lower, upper, 0))
              << GraphFunctionLayer(params)
              << arm_compute::graph::frontend::OutputLayer(get_verify_accessor<VerifyAccessorT>(params));

        arm_compute::graph::GraphConfig config;
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

    arm_compute::graph::frontend::Stream graph;
};

} // graph_validate_utils
} // arm_compute
#endif //GRAPH_VALIDATE_UTILS_H
