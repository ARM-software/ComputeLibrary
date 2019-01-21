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
#include "tests/validation/reference/ConvolutionLayer.h"
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
/*Available Padding modes */
enum class PaddingMode
{
    Valid,
    Same,
    Manual
};

/** Stream Input operator for the PaddingMode type
 *
 * @param[in]  stream Input stream.
 * @param[out] Mode   Convolution parameters to output
 *
 * @return input stream.
 */
inline ::std::istream &operator>>(::std::istream &stream, PaddingMode &Mode)
{
    static const std::map<std::string, PaddingMode> modes =
    {
        { "valid", PaddingMode::Valid },
        { "same", PaddingMode::Same },
        { "manual", PaddingMode::Manual }
    };
    std::string value;
    stream >> value;
    try
    {
        Mode = modes.at(arm_compute::utility::tolower(value));
    }
    catch(const std::out_of_range &)
    {
        throw std::invalid_argument(value);
    }

    return stream;
}

/** Formatted output of the PaddingMode type
 *
 * @param[out] os   Output stream.
 * @param[in]  Mode PaddingMode to output
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, PaddingMode Mode)
{
    switch(Mode)
    {
        case PaddingMode::Valid:
            os << "Valid";
            break;
        case PaddingMode::Same:
            os << "Same";
            break;
        case PaddingMode::Manual:
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
    int              width{ 0 };
    int              height{ 0 };
    int              fm{ 0 };
    int              batch{ 0 };
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

/** Structure holding all the Convolution layer graph parameters */
struct ConvolutionParams
{
    arm_compute::DataType                 data_type{ DataType::F32 };
    arm_compute::DataLayout               data_layout{ DataLayout::NCHW };
    arm_compute::graph::ConvolutionMethod convolution_method{ arm_compute::graph::ConvolutionMethod::Default };

    /** Padding graph parameters */
    int         padding_top{ 0 };
    int         padding_bottom{ 0 };
    int         padding_left{ 0 };
    int         padding_right{ 0 };
    int         padding_stride_x{ 0 };
    int         padding_stride_y{ 0 };
    PaddingMode padding_mode{ PaddingMode::Valid };
    struct
    {
        struct
        {
            int X{ 0 };
            int Y{ 0 };
        } stride{};
        PaddingMode mode{ PaddingMode::Valid };
    } padding{};
};

/** Structure holding all the graph Example parameters */
struct ExampleParams
{
    FrameworkParams    common_params{};
    TensorParams       input{};
    TensorParams       weights{};
    TensorParams       bias{};
    TensorParams       output{};
    VerificationParams verification{};
    ConvolutionParams  convolution{};
};

/** Formatted output of the ConvolutionParams type
 *
 * @param[out] os            Output stream.
 * @param[in]  common_params Convolution parameters to output
 *
 * @return Modified output stream.
 */
::std::ostream &operator<<(::std::ostream &os, const ExampleParams &common_params)
{
    os << "Threads : " << common_params.common_params.threads << std::endl;
    os << "Target : " << common_params.common_params.target << std::endl;
    os << "Data type : " << common_params.convolution.data_type << std::endl;
    os << "Input dimensions(X,Y, Channels, Batch) : (" << common_params.input.width << "," << common_params.input.height << "," << common_params.input.fm << "," << common_params.input.batch << ")"
       << std::endl;
    os << "Weight dimensions(X,Y, Channels(same as input), OFM) : (" << common_params.weights.width << "," << common_params.weights.height << "," << common_params.input.fm << "," <<
       common_params.weights.fm << ")" << std::endl;
    os << "Padding(top, bottom, left, right) (stride x, stride y) : (" << common_params.convolution.padding_top << "," << common_params.convolution.padding_bottom << "," <<
       common_params.convolution.padding_left << "," << common_params.convolution.padding_right << ") (" << common_params.convolution.padding_stride_x << "," << common_params.convolution.padding_stride_y <<
       ")" << std::endl;
    os << "Padding Mode: " << common_params.convolution.padding_mode << std::endl;
    os << "Convolution Method: " << common_params.convolution.convolution_method << std::endl;
    return os;
}

/** Convolution command line options used to configure the graph examples
 *
 * (Similar to common options)
 * The options in this object get populated when "parse()" is called on the parser used to construct it.
 * The expected workflow is:
 *
 * CommandLineParser parser;
 * CommonOptions options( parser );
 * parser.parse(argc, argv);
 */
class ConvolutionOptions final
{
public:
    explicit ConvolutionOptions(CommandLineParser &parser) noexcept
        : width(parser.add_option<SimpleOption<int>>("width", 9)),
          height(parser.add_option<SimpleOption<int>>("height", 9)),
          channels(parser.add_option<SimpleOption<int>>("channels", 1)),
          batch(parser.add_option<SimpleOption<int>>("batch", 1)),
          weights_width(parser.add_option<SimpleOption<int>>("weights_width", 3)),
          weights_height(parser.add_option<SimpleOption<int>>("weights_height", 3)),
          OFM(parser.add_option<SimpleOption<int>>("OFM", 1)),
          padding_top(parser.add_option<SimpleOption<int>>("padding_top", 0)),
          padding_left(parser.add_option<SimpleOption<int>>("padding_left", 0)),
          padding_bottom(parser.add_option<SimpleOption<int>>("padding_bottom", 0)),
          padding_right(parser.add_option<SimpleOption<int>>("padding_right", 0)),
          stride_x(parser.add_option<SimpleOption<int>>("stride_x", 1)),
          stride_y(parser.add_option<SimpleOption<int>>("stride_y", 1)),
          help(parser.add_option<ToggleOption>("help")),
          threads(parser.add_option<SimpleOption<int>>("threads")),
          target(),
          data_type(),
          padding_mode(),
          conv_mode(),
          data_layout(),
          absolute_tolerance(parser.add_option<SimpleOption<float>>("abs_tolerance", -1.0f)),
          relative_tolerance(parser.add_option<SimpleOption<float>>("rel_tolerance", -1.0f)),
          tolerance_number(parser.add_option<SimpleOption<float>>("tolerance_num", -1.0f)),
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
        const std::set<PaddingMode> available_padding_modes
        {
            PaddingMode::Valid,
            PaddingMode::Same
        };

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

        const std::set<arm_compute::graph::ConvolutionMethod> supported_convolution_methods
        {
            arm_compute::graph::ConvolutionMethod::Default,
            arm_compute::graph::ConvolutionMethod::GEMM,
            arm_compute::graph::ConvolutionMethod::Winograd,
            arm_compute::graph::ConvolutionMethod::Direct
        };

        const std::set<DataLayout> supported_data_layouts
        {
            DataLayout::NHWC,
            DataLayout::NCHW,
        };

        padding_mode = parser.add_option<EnumOption<PaddingMode>>("padding_mode", available_padding_modes, PaddingMode::Valid);
        target       = parser.add_option<EnumOption<Target>>("target", supported_targets, Target::NEON);
        data_type    = parser.add_option<EnumOption<DataType>>("type", supported_data_types, DataType::F32);
        conv_mode    = parser.add_option<EnumOption<arm_compute::graph::ConvolutionMethod>>("convolution_method", supported_convolution_methods, arm_compute::graph::ConvolutionMethod::Default);
        data_layout  = parser.add_option<EnumOption<DataLayout>>("layout", supported_data_layouts, DataLayout::NHWC);

        target->set_help("Target to execute on");
        data_type->set_help("Data type to use");
        padding_mode->set_help("Set padding mode");
        help->set_help("Show this help message");
        width->set_help("Set Input dimension width");
        height->set_help("Set Input dimension height");
        channels->set_help("Set Input dimension channels");
        batch->set_help("Set Input dimension batch");
        weights_width->set_help("Set weights_dimensions width");
        weights_height->set_help("Set weights_dimensions height");
        OFM->set_help("Set OFM");
        padding_top->set_help("Set padding top");
        padding_bottom->set_help("Set padding bottom");
        padding_left->set_help("Set padding left");
        padding_right->set_help("Set padding right");
        stride_x->set_help("Set padding stride x");
        stride_y->set_help("Set padding stride y");
        conv_mode->set_help("Set convolution method");
        data_layout->set_help("Data layout to use");
        absolute_tolerance->set_help("Absolute tolerance used for verification");
        relative_tolerance->set_help("Absolute tolerance used for verification");
        tolerance_number->set_help("Absolute tolerance used for verification");
        scale->set_help("Quantization scale from QASYMM8");
        offset->set_help("Quantization offset from QASYMM8");
        weights_scale->set_help("Quantization scale from QASYMM8");
        weights_offset->set_help("Quantization offset from QASYMM8");
        output_scale->set_help("Quantization scale from QASYMM8");
        output_offset->set_help("Quantization offset from QASYMM8");
        input_npy->set_help("Use input .npy instead");
        output_npy->set_help("Use .npy as a reference");
        input_range_low->set_help("Lower bound for input randomization range");
        input_range_high->set_help("Lower bound for input randomization range");
        weights_range_low->set_help("Lower bound for input randomization range");
        weights_range_high->set_help("Lower bound for input randomization range");
    }

    /** Prevent instances of this class from being copied (As this class contains pointers) */
    ConvolutionOptions(const ConvolutionOptions &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    ConvolutionOptions &operator=(const ConvolutionOptions &) = delete;
    /** Allow instances of this class to be moved */
    ConvolutionOptions(ConvolutionOptions &&) noexcept(true) = default;
    /** Allow instances of this class to be moved */
    ConvolutionOptions &operator=(ConvolutionOptions &&) noexcept(true) = default;
    /** Default destructor */
    ~ConvolutionOptions() = default;

    SimpleOption<int>                                 *width;              /**< Input width */
    SimpleOption<int>                                 *height;             /**< Input height */
    SimpleOption<int>                                 *channels;           /**< Input channels */
    SimpleOption<int>                                 *batch;              /**< Input batch */
    SimpleOption<int>                                 *weights_width;      /**< weights width */
    SimpleOption<int>                                 *weights_height;     /**< weights height */
    SimpleOption<int>                                 *OFM;                /**< Output Feature Map */
    SimpleOption<int>                                 *padding_top;        /**< Padding top */
    SimpleOption<int>                                 *padding_left;       /**< Padding left */
    SimpleOption<int>                                 *padding_bottom;     /**< Padding bottom */
    SimpleOption<int>                                 *padding_right;      /**< Padding right */
    SimpleOption<int>                                 *stride_x;           /**< Padding stride x */
    SimpleOption<int>                                 *stride_y;           /**< Padding stride y */
    ToggleOption                                      *help;               /**< show help message */
    SimpleOption<int>                                 *threads;            /**< Number of threads option */
    EnumOption<arm_compute::graph::Target>            *target;             /**< Graph execution target */
    EnumOption<arm_compute::DataType>                 *data_type;          /**< Graph data type */
    EnumOption<PaddingMode>                           *padding_mode;       /**< Padding mode */
    EnumOption<arm_compute::graph::ConvolutionMethod> *conv_mode;          /**< Convolution method */
    EnumOption<arm_compute::DataLayout>               *data_layout;        /**< Graph data layout */
    SimpleOption<float>                               *absolute_tolerance; /**< Absolute tolerance used in verification */
    SimpleOption<float>                               *relative_tolerance; /**< Relative tolerance used in verification */
    SimpleOption<float>                               *tolerance_number;   /**< Tolerance number used in verification */
    SimpleOption<float>                               *scale;              /**< Input Quantization scale from QASYMM8 */
    SimpleOption<int>                                 *offset;             /**< Input Quantization offset from QASYMM8 */
    SimpleOption<float>                               *weights_scale;      /**< Weights Quantization scale from QASYMM8 */
    SimpleOption<int>                                 *weights_offset;     /**< Weights Quantization offset from QASYMM8 */
    SimpleOption<float>                               *output_scale;       /**< Output Quantization scale from QASYMM8 */
    SimpleOption<int>                                 *output_offset;      /**< Output Quantization offset from QASYMM8 */
    SimpleOption<uint64_t>                            *input_range_low;    /**< Lower bound for input randomization range */
    SimpleOption<uint64_t>                            *input_range_high;   /**< Upper bound for input randomization range */
    SimpleOption<uint64_t>                            *weights_range_low;  /**< Lower bound for weights randomization range */
    SimpleOption<uint64_t>                            *weights_range_high; /**< Upper bound for weights randomization range */

    SimpleOption<std::string> *input_npy;   /**< Use input .npy image */
    SimpleOption<std::string> *output_npy;  /**< Use output .npy image to verify*/
    SimpleOption<std::string> *weights_npy; /**< Use weights .npy image */
    SimpleOption<std::string> *bias_npy;    /**< Use bias .npy image */
};

/** Consumes the convolution graph options and creates a structure containing any information
 *
 * @param[in] options Options to consume
 *
 * @return Convolutionparams structure containing the common graph parameters
 */
ExampleParams consume_covolution_graph_parameters(ConvolutionOptions &options)
{
    ExampleParams common_params;

    common_params.common_params.help    = options.help->is_set() ? options.help->value() : false;
    common_params.common_params.threads = options.threads->value();
    common_params.common_params.target  = options.target->value();

    common_params.input.width             = options.width->value();
    common_params.input.height            = options.height->value();
    common_params.input.fm                = options.channels->value();
    common_params.input.batch             = options.batch->value();
    common_params.input.quant_info.scale  = options.scale->value();
    common_params.input.quant_info.offset = options.offset->value();
    common_params.input.npy               = options.input_npy->value();
    common_params.input.range_low         = options.input_range_low->value();
    common_params.input.range_high        = options.input_range_high->value();

    common_params.weights.width             = options.weights_width->value();
    common_params.weights.height            = options.weights_height->value();
    common_params.weights.fm                = options.OFM->value();
    common_params.weights.npy               = options.weights_npy->value();
    common_params.weights.quant_info.scale  = options.weights_scale->value();
    common_params.weights.quant_info.offset = options.weights_offset->value();
    common_params.weights.range_low         = options.weights_range_low->value();
    common_params.weights.range_high        = options.weights_range_high->value();

    common_params.bias.npy = options.bias_npy->value();

    common_params.output.quant_info.scale  = options.output_scale->value();
    common_params.output.quant_info.offset = options.output_offset->value();
    common_params.output.npy               = options.output_npy->value();

    common_params.convolution.padding_mode       = options.padding_mode->value();
    common_params.convolution.padding_top        = options.padding_top->value();
    common_params.convolution.padding_bottom     = options.padding_bottom->value();
    common_params.convolution.padding_left       = options.padding_left->value();
    common_params.convolution.padding_right      = options.padding_right->value();
    common_params.convolution.padding_stride_x   = options.stride_x->value();
    common_params.convolution.padding_stride_y   = options.stride_y->value();
    common_params.convolution.convolution_method = options.conv_mode->value();
    common_params.convolution.data_type          = options.data_type->value();
    common_params.convolution.data_layout        = options.data_layout->value();

    common_params.verification.absolute_tolerance = options.absolute_tolerance->value();
    common_params.verification.relative_tolerance = options.relative_tolerance->value();
    common_params.verification.tolerance_number   = options.tolerance_number->value();

    return common_params;
}

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
        case PaddingMode::Manual:
        {
            return PadStrideInfo(params.convolution.padding_stride_x, params.convolution.padding_stride_y, params.convolution.padding_left, params.convolution.padding_right, params.convolution.padding_top,
                                 params.convolution.padding_bottom, DimensionRoundingType::FLOOR);
        }
        case PaddingMode::Valid:
        {
            return PadStrideInfo();
        }
        case PaddingMode::Same:
        {
            return arm_compute::calculate_same_pad(TensorShape(params.input.width, params.input.height), TensorShape(params.weights.width, params.weights.height),
                                                   PadStrideInfo(params.convolution.padding_stride_x,
                                                                 params.convolution.padding_stride_y));
        }
        default:
            ARM_COMPUTE_ERROR("NOT SUPPORTED!");
    }
}

/** ConvolutionLayer Graph example validation accessor class */
template <typename D>
class ConvolutionVerifyAccessor final : public graph::ITensorAccessor
{
public:
    using TBias = typename std::conditional<std::is_same<typename std::decay<D>::type, uint8_t>::value, int32_t, D>::type;

    /** Constructor
     *
     * @param[in] params Convolution parameters
     */
    explicit ConvolutionVerifyAccessor(ExampleParams &params)
        : _params(std::move(params))
    {
    }

    // Inherited methods overriden:
    bool access_tensor(ITensor &tensor) override
    {
        if(_params.output.npy.empty())
        {
            const RelativeTolerance<float> rel_tolerance(relative_tolenace(_params.verification.relative_tolerance));  /**< Relative tolerance */
            const AbsoluteTolerance<float> abs_tolerance(absolute_tolerance(_params.verification.absolute_tolerance)); /**< Absolute tolerance */
            const float                    tolerance_num(tolerance_number(_params.verification.tolerance_number));     /**< Tolerance number */

            //Create Input tensors
            SimpleTensor<D>     src{ TensorShape(_params.input.width, _params.input.height, _params.input.fm, _params.input.batch), _params.convolution.data_type, 1, _params.input.quant_info };
            SimpleTensor<D>     weights{ TensorShape(_params.weights.width, _params.weights.height, _params.weights.fm), _params.convolution.data_type, 1, _params.weights.quant_info };
            SimpleTensor<TBias> bias{ TensorShape(_params.input.height), _params.convolution.data_type, 1, _params.input.quant_info };

            //Fill the tenors with random values
            fill_tensor<D>(src, 0, static_cast<D>(_params.input.range_low), static_cast<D>(_params.input.range_high));
            fill_tensor<D>(weights, 1, static_cast<D>(_params.weights.range_low), static_cast<D>(_params.weights.range_high));
            fill_tensor<TBias>(bias, 2, static_cast<TBias>(_params.input.range_low), static_cast<TBias>(_params.input.range_high));

            // Calculate padding information
            const PadStrideInfo padding_info = calculate_convolution_padding(_params);

            //Calculate reference
            SimpleTensor<D> output = reference::convolution_layer<D>(src, weights, bias, permute_shape(tensor.info()->tensor_shape(), _params.convolution.data_layout, DataLayout::NCHW), padding_info, Size2D(1,
                                                                     1),
                                                                     1,
                                                                     _params.output.quant_info);

            arm_compute::test::validation::validate(Accessor(tensor), output, rel_tolerance, tolerance_num, abs_tolerance);
        }
        else
        {
            //The user provided a reference file use an npy accessor to validate
            NumPyAccessor(_params.output.npy, tensor.info()->tensor_shape(), tensor.info()->data_type()).access_tensor(tensor);
        }
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
                uint8_t qasymm8_low  = tensor.quantization_info().quantize(low, RoundingPolicy::TO_NEAREST_UP);
                uint8_t qasymm8_high = tensor.quantization_info().quantize(high, RoundingPolicy::TO_NEAREST_UP);

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
                    { DataType::F32, 0.5f },
                    { DataType::QASYMM8, 1.0f }
                }
            },
            {
                arm_compute::graph::Target::NEON,
                {   { DataType::F16, 0.2f },
                    { DataType::F32, 0.01f },
                    { DataType::QASYMM8, 0.0f }
                }
            }
        };
        if(user_value == -1)
        {
            if(_params.convolution.convolution_method == arm_compute::graph::ConvolutionMethod::Winograd
               && _params.convolution.data_type == DataType::F32
               && _params.common_params.target == arm_compute::graph::Target::NEON)
            {
                return 0.05f;
            }
            else
            {
                return relative_tolerance.at(_params.common_params.target).at(_params.convolution.data_type);
            }
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

        if(user_value == -1)
        {
            return absolute_tolerance.at(_params.common_params.target).at(_params.convolution.data_type);
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
            return absolute_tolerance.at(_params.common_params.target).at(_params.convolution.data_type);
        }
        return user_value;
    }

    ExampleParams _params;
};

/** Generates appropriate convolution verify accessor
 *
 * @param[in] params User supplied parameters for convolution.
 *
 * @return A convolution verify accessor for the requested datatype.
 */
inline std::unique_ptr<graph::ITensorAccessor> get_convolution_verify_accessor(ExampleParams params)
{
    switch(params.convolution.data_type)
    {
        case DataType::QASYMM8:
        {
            return arm_compute::support::cpp14::make_unique<ConvolutionVerifyAccessor<uint8_t>>(
                       params);
        }
        case DataType::F16:
        {
            return arm_compute::support::cpp14::make_unique<ConvolutionVerifyAccessor<half>>(
                       params);
        }
        case DataType::F32:
        {
            return arm_compute::support::cpp14::make_unique<ConvolutionVerifyAccessor<float>>(
                       params);
        }
        default:
            ARM_COMPUTE_ERROR("NOT SUPPORTED!");
    }
}
/** Generates appropriate accessor according to the specified graph parameters
 *
 * @param[in] graph_parameters Graph parameters
 * @param[in] lower            Lower random values bound
 * @param[in] upper            Upper random values bound
 * @param[in] seed             Random generator seed
 *
 * @return An appropriate tensor accessor
 */
inline std::unique_ptr<graph::ITensorAccessor> get_accessor(const TensorParams &tensor, PixelValue lower, PixelValue upper, const std::random_device::result_type seed = 0)
{
    if(!tensor.npy.empty())
    {
        return arm_compute::support::cpp14::make_unique<NumPyBinLoader>(tensor.npy);
    }
    else
    {
        return arm_compute::support::cpp14::make_unique<RandomAccessor>(lower, upper, seed);
    }
}
} // namespace

class GraphConvolutionValidateExample final : public ValidateExample
{
public:
    GraphConvolutionValidateExample()
        : graph(0, "Convolution Graph example")
    {
    }
    bool do_setup(int argc, char **argv) override
    {
        CommandLineParser parser;

        ConvolutionOptions Options(parser);

        parser.parse(argc, argv);

        ExampleParams params = consume_covolution_graph_parameters(Options);

        if(params.common_params.help)
        {
            parser.print_help(argv[0]);
            return false;
        }

        std::cout << params << std::endl;

        // Calculate padding information
        const PadStrideInfo padding_info = calculate_convolution_padding(params);

        // Create input descriptor
        const TensorShape input_shape      = permute_shape(TensorShape(params.input.width, params.input.height, params.input.fm, params.input.batch), DataLayout::NCHW, params.convolution.data_layout);
        TensorDescriptor  input_descriptor = TensorDescriptor(input_shape, params.convolution.data_type, params.input.quant_info, params.convolution.data_layout);

        const PixelValue lower = PixelValue(params.input.range_low, params.convolution.data_type, params.input.quant_info);
        const PixelValue upper = PixelValue(params.input.range_high, params.convolution.data_type, params.input.quant_info);

        const PixelValue weights_lower = PixelValue(params.weights.range_low, params.convolution.data_type, params.weights.quant_info);
        const PixelValue weights_upper = PixelValue(params.weights.range_high, params.convolution.data_type, params.weights.quant_info);

        graph << params.common_params.target
              << params.convolution.convolution_method
              << InputLayer(input_descriptor, get_accessor(params.input, lower, upper, 0))
              << ConvolutionLayer(params.weights.width, params.weights.height, params.weights.fm,
                                  get_accessor(params.weights, weights_lower, weights_upper, 1),
                                  get_accessor(params.bias, lower, upper, 2),
                                  padding_info, 1, params.weights.quant_info, params.output.quant_info)
              << OutputLayer(get_convolution_verify_accessor(params));

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

/** Main program for Graph Convolution test
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments ( Input dimensions [width, height, channels, batch]
 *                             Weights dimensions [width, height, OFM]
 *                             Padding [top,bottom,left,right, Stride x, Stride y, mode [Valid / Same / Manual] )
 *                             Convolution Method[ Auto/GEMM/Winograd/Direct]
 *                             Verification[tolerance_number,absolute_tolerance,relative_tolerance] )
 *
 */
int main(int argc, char **argv)
{
    return arm_compute::utils::run_example<GraphConvolutionValidateExample>(argc, argv);
}
