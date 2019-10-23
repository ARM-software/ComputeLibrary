/*
 * Copyright (c) 2017-2019 ARM Limited.
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
#ifndef ARM_COMPUTE_CL /* Needed by Utils.cpp to handle OpenCL exceptions properly */
#error "This example needs to be built with -DARM_COMPUTE_CL"
#endif /* ARM_COMPUTE_CL */

#include "arm_compute/core/Types.h"
#include "arm_compute/core/utils/quantization/AsymmHelpers.h"
#include "arm_compute/runtime/CL/CLFunctions.h"
#include "arm_compute/runtime/CL/CLScheduler.h"

#include "tests/AssetsLibrary.h"
#include "tests/CL/CLAccessor.h"
#include "tests/Globals.h"
#include "tests/IAccessor.h"
#include "tests/SimpleTensor.h"
#include "tests/validation/Validation.h"
#include "tests/validation/reference/GEMM.h"
#include "tests/validation/reference/GEMMLowp.h"

#include "utils/TypePrinter.h"
#include "utils/Utils.h"
#include "utils/command_line/CommandLineOptions.h"
#include "utils/command_line/CommandLineParser.h"

#include "ValidateExample.h"

#include <cstdlib>

using namespace arm_compute;
using namespace utils;
using namespace arm_compute::test;
using namespace arm_compute::test::validation;

constexpr float                     abs_tolerance_f32(0.0001f); /**< F32 Absolute tolerance value for comparing reference's output against implementation's output for
                                                               * floating point data types in case using relative tolerance fails because of small values */
RelativeTolerance<float>            tolerance_f32(0.001f);      /**< F32 Tolerance value for comparing reference's output against implementation's output for floating point data types */
RelativeTolerance<half_float::half> tolerance_f16(half(0.2));   /**< F16 Tolerance value for comparing reference's output against implementation's output for floating point data types */
constexpr float                     tolerance_num_f16 = 0.02f;  /**< F16 Tolerance number */

namespace arm_compute
{
DataType data_type_from_name(const std::string &name)
{
    static const std::map<std::string, DataType> data_types =
    {
        { "f16", DataType::F16 },
        { "f32", DataType::F32 },
        { "qasymm8", DataType::QASYMM8 },
    };

#ifndef ARM_COMPUTE_EXCEPTIONS_DISABLED
    try
    {
#endif /* ARM_COMPUTE_EXCEPTIONS_DISABLED */
        return data_types.at(utility::tolower(name));

#ifndef ARM_COMPUTE_EXCEPTIONS_DISABLED
    }
    catch(const std::out_of_range &)
    {
        throw std::invalid_argument(name);
    }
#endif /* ARM_COMPUTE_EXCEPTIONS_DISABLED */
}

inline ::std::istream &operator>>(::std::istream &stream, DataType &data_type)
{
    std::string value;
    stream >> value;
    data_type = data_type_from_name(value);
    return stream;
}
} // namespace arm_compute
namespace
{
class GEMMCommandLineOptions final
{
public:
    explicit GEMMCommandLineOptions(CommandLineParser &parser) noexcept
        : help(parser.add_option<ToggleOption>("help")),
          add_bias(parser.add_option<ToggleOption>("add_bias")),
          M(parser.add_option<SimpleOption<int>>("m", 7)),
          N(parser.add_option<SimpleOption<int>>("n", 3)),
          K(parser.add_option<SimpleOption<int>>("k", 5)),
          B(parser.add_option<SimpleOption<int>>("b", 1)),
          alpha(parser.add_option<SimpleOption<float>>("alpha", 1.f)),
          beta(parser.add_option<SimpleOption<float>>("beta", 0.f)),
          offset_src0(parser.add_option<SimpleOption<int>>("offset_i0", 10)),
          offset_src1(parser.add_option<SimpleOption<int>>("offset_i1", 10)),
          offset_dst(parser.add_option<SimpleOption<int>>("offset_o", 10)),
          scale_src0(parser.add_option<SimpleOption<float>>("scale_i0", 1.f / 255)),
          scale_src1(parser.add_option<SimpleOption<float>>("scale_i1", 1.f / 255)),
          scale_dst(parser.add_option<SimpleOption<float>>("scale_o", 1.f / 255)),
          data_type()
    {
        // Setup data type
        const std::set<arm_compute::DataType> supported_data_types
        {
            DataType::F16,
            DataType::F32,
            DataType::QASYMM8,
        };
        data_type = parser.add_option<EnumOption<DataType>>("type", supported_data_types, DataType::F32);

        // Setup help strings
        help->set_help("Show this help message");
        add_bias->set_help("Add bias to the GEMM. Used when running in QASYMM8");
        M->set_help("M value");
        N->set_help("N value");
        K->set_help("K value");
        B->set_help("B value - number of batches");
        alpha->set_help("Alpha value");
        beta->set_help("Beta value");
        offset_src0->set_help("Offset of first input. Used when running in QASYMM8");
        offset_src1->set_help("Offset of second input. Used when running in QASYMM8");
        offset_dst->set_help("Offset of output. Used when running in QASYMM8");
        scale_src0->set_help("Scale of first input. Used when running in QASYMM8");
        scale_src1->set_help("Scale of second input. Used when running in QASYMM8");
        scale_dst->set_help("Scale of output. Used when running in QASYMM8");
        data_type->set_help("Data type to use");
    }
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    GEMMCommandLineOptions(const GEMMCommandLineOptions &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    GEMMCommandLineOptions &operator=(const GEMMCommandLineOptions &) = delete;
    /** Allow instances of this class to be moved */
    GEMMCommandLineOptions(GEMMCommandLineOptions &&) noexcept(true) = default;
    /** Allow instances of this class to be moved */
    GEMMCommandLineOptions &operator=(GEMMCommandLineOptions &&) noexcept(true) = default;
    /** Default destructor */
    ~GEMMCommandLineOptions() = default;

public:
    ToggleOption                      *help;
    ToggleOption                      *add_bias;
    SimpleOption<int>                 *M;
    SimpleOption<int>                 *N;
    SimpleOption<int>                 *K;
    SimpleOption<int>                 *B;
    SimpleOption<float>               *alpha;
    SimpleOption<float>               *beta;
    SimpleOption<int>                 *offset_src0;
    SimpleOption<int>                 *offset_src1;
    SimpleOption<int>                 *offset_dst;
    SimpleOption<float>               *scale_src0;
    SimpleOption<float>               *scale_src1;
    SimpleOption<float>               *scale_dst;
    EnumOption<arm_compute::DataType> *data_type;
};
} // namespace

class CLGEMMValidateExample : public ValidateExample
{
public:
    bool do_setup(int argc, char **argv) override
    {
        CLScheduler::get().default_init();

        // Parse options
        CommandLineParser      parser;
        GEMMCommandLineOptions gemm_options(parser);
        parser.parse(argc, argv);

        // Print help
        const bool print_help = gemm_options.help->is_set() ? gemm_options.help->value() : false;
        if(print_help)
        {
            parser.print_help(argv[0]);
            return false;
        }

        // Consume parameters
        consume_params(gemm_options);
        print_parameters_internal();

        const bool is_quantized = is_data_type_quantized(data_type);

        // Calculate re-quantization parameters
        if(is_quantized)
        {
            float multiplier = scale_src0 * scale_src1 / scale_dst;
            quantization::calculate_quantized_multiplier(multiplier, &dst_multiplier, &dst_shift);
        }

        // Initialize GEMM inputs/outputs
        src0.allocator()->init(TensorInfo(TensorShape(K, M, B), 1, data_type));
        src1.allocator()->init(TensorInfo(TensorShape(N, K, B), 1, data_type));
        src2.allocator()->init(TensorInfo(TensorShape(N, M, B), 1, data_type));
        init_sgemm_output(dst, src0, src1, data_type);

        // Configure function
        if(is_quantized)
        {
            src0.info()->set_quantization_info(QuantizationInfo(scale_src0, offset_src0));
            src1.info()->set_quantization_info(QuantizationInfo(scale_src1, offset_src1));
            dst.info()->set_quantization_info(QuantizationInfo(scale_dst, offset_dst));
            biases.allocator()->init(TensorInfo(TensorShape(N), 1, DataType::S32));
            init_sgemm_output(tmp_dst, src0, src1, DataType::S32);

            // Configure GEMMlowp matrix multiply function
            mm_gemmlowp.configure(&src0, &src1, nullptr, &tmp_dst);

            // Configure GEMMlowp output stage
            mm_gemmlowp_output_stage.configure(&tmp_dst, add_bias ? &biases : nullptr, &dst, dst_multiplier, dst_shift, offset_dst);
            tmp_dst.allocator()->allocate();
            biases.allocator()->allocate();
            fill(CLAccessor(biases), 3);
        }
        else
        {
            // Configure matrix multiply function
            mm_gemm.configure(&src0, &src1, &src2, &dst, alpha, beta);
        }

        // Allocate all the tensors
        src0.allocator()->allocate();
        src1.allocator()->allocate();
        dst.allocator()->allocate();
        src2.allocator()->allocate();

        fill(CLAccessor(src0), 0);
        fill(CLAccessor(src1), 1);
        fill(CLAccessor(src2), 2);

        return true;
    }

    void print_parameters_internal()
    {
        std::cout << "Datatype : " << string_from_data_type(data_type) << "\n";
        std::cout << "M : " << support::cpp11::to_string(M) << "\n";
        std::cout << "N : " << support::cpp11::to_string(N) << "\n";
        std::cout << "K : " << support::cpp11::to_string(K) << "\n";
        std::cout << "B : " << support::cpp11::to_string(B) << "\n";
        if(data_type == DataType::QASYMM8)
        {
            std::cout << "Scale_Src0 : " << support::cpp11::to_string(scale_src0) << "\n";
            std::cout << "Offset_Src0 : " << support::cpp11::to_string(offset_src0) << "\n";
            std::cout << "Scale_Scr1 : " << support::cpp11::to_string(scale_src1) << "\n";
            std::cout << "Offset_Src1 : " << support::cpp11::to_string(offset_src1) << "\n";
            std::cout << "Scale_Dst : " << support::cpp11::to_string(scale_dst) << "\n";
            std::cout << "Offset_Dst : " << support::cpp11::to_string(offset_dst) << "\n";
            std::cout << "Bias : " << support::cpp11::to_string(add_bias) << "\n";
        }
        else
        {
            std::cout << "Alpha : " << support::cpp11::to_string(alpha) << "\n";
            std::cout << "Beta : " << support::cpp11::to_string(beta) << "\n";
        }
    }

    void do_validate() override
    {
        switch(data_type)
        {
            case DataType::F16:
            {
                SimpleTensor<half> ref_src0 = { TensorShape(K, M, B), data_type, 1 };
                SimpleTensor<half> ref_src1 = { TensorShape(N, K, B), data_type, 1 };
                SimpleTensor<half> ref_src2 = { TensorShape(N, M, B), data_type, 1 };

                fill(ref_src0, 0);
                fill(ref_src1, 1);
                fill(ref_src2, 2);

                SimpleTensor<half> ref_dst = reference::gemm<half>(ref_src0, ref_src1, ref_src2, alpha, beta);
                validate(CLAccessor(dst), ref_dst, tolerance_f16, tolerance_num_f16);
                break;
            }
            case DataType::F32:
            {
                SimpleTensor<float> ref_src0 = { TensorShape(K, M, B), data_type, 1 };
                SimpleTensor<float> ref_src1 = { TensorShape(N, K, B), data_type, 1 };
                SimpleTensor<float> ref_src2 = { TensorShape(N, M, B), data_type, 1 };

                fill(ref_src0, 0);
                fill(ref_src1, 1);
                fill(ref_src2, 2);

                SimpleTensor<float> ref_dst = reference::gemm<float>(ref_src0, ref_src1, ref_src2, alpha, beta);
                validate(CLAccessor(dst), ref_dst, tolerance_f32, 0.f, abs_tolerance_f32);
                break;
            }
            case DataType::QASYMM8:
            {
                SimpleTensor<uint8_t> ref_src0{ TensorShape(K, M, B), data_type, 1 };
                SimpleTensor<uint8_t> ref_src1{ TensorShape(N, K, B), data_type, 1 };
                SimpleTensor<uint8_t> ref_dst;

                // Fill reference
                fill(ref_src0, 0);
                fill(ref_src1, 1);

                SimpleTensor<int32_t> ref_tmp_dst = reference::gemmlowp_matrix_multiply_core<int32_t, uint8_t>(ref_src0, ref_src1, TensorShape(N, M, B), offset_src0, offset_src1);

                const std::vector<int32_t> dst_multiplier_vec = { dst_multiplier };
                const std::vector<int32_t> dst_shift_vec      = { dst_shift };

                if(add_bias)
                {
                    SimpleTensor<int32_t> biases{ TensorShape(N), DataType::S32, 1 };
                    // Fill bias
                    fill(biases, 3);
                    ref_dst = reference::gemmlowp_quantize_down_scale_by_fixedpoint<int32_t, uint8_t>(ref_tmp_dst, biases, dst_multiplier_vec, dst_shift_vec, offset_dst);
                }
                else
                {
                    ref_dst = reference::gemmlowp_quantize_down_scale_by_fixedpoint<int32_t, uint8_t>(ref_tmp_dst, dst_multiplier_vec, dst_shift_vec, offset_dst);
                }
                validate(CLAccessor(dst), ref_dst);
                break;
            }
            default:
                break;
        }
    }
    void do_run() override
    {
        // Execute the function
        if(data_type == DataType::QASYMM8)
        {
            // Run gemmlowp
            mm_gemmlowp.run();
            // Run output stage
            mm_gemmlowp_output_stage.run();
        }
        else
        {
            // Run gemm
            mm_gemm.run();
        }

        // Make sure all the OpenCL jobs are done executing:
        CLScheduler::get().sync();
    }

private:
    template <typename U>
    void fill(U &&tensor, int i)
    {
        switch(tensor.data_type())
        {
            case DataType::F16:
            case DataType::F32:
            {
                std::uniform_real_distribution<> distribution(-1.0f, 1.0f);
                library->fill(tensor, distribution, i);
                break;
            }
            case DataType::S32:
            case DataType::QASYMM8:
            {
                std::uniform_int_distribution<> distribution(-6000, 6000);
                library->fill(tensor, distribution, i);
                break;
            }
            default:
                library->fill_tensor_uniform(tensor, i);
        }
    }

    void consume_params(const GEMMCommandLineOptions &opts)
    {
        ARM_COMPUTE_ERROR_ON(opts.M->value() <= 0);
        ARM_COMPUTE_ERROR_ON(opts.N->value() <= 0);
        ARM_COMPUTE_ERROR_ON(opts.K->value() <= 0);
        ARM_COMPUTE_ERROR_ON(opts.B->value() <= 0);
        M           = opts.M->value();
        N           = opts.N->value();
        K           = opts.K->value();
        B           = opts.B->value();
        alpha       = opts.alpha->value();
        beta        = opts.beta->value();
        offset_src0 = opts.offset_src0->value();
        offset_src1 = opts.offset_src1->value();
        offset_dst  = opts.offset_dst->value();
        scale_src0  = opts.scale_src0->value();
        scale_src1  = opts.scale_src1->value();
        scale_dst   = opts.scale_dst->value();
        add_bias    = opts.add_bias->is_set() ? opts.add_bias->value() : true;
        data_type   = opts.data_type->value();
    }

    CLTensor src0{}, src1{}, src2{}, dst{};
    CLTensor tmp_dst{}, biases{};

    CLGEMM                                              mm_gemm{};
    CLGEMMLowpMatrixMultiplyCore                        mm_gemmlowp{};
    CLGEMMLowpQuantizeDownInt32ToUint8ScaleByFixedPoint mm_gemmlowp_output_stage{};

    size_t   M{ 7 }, N{ 3 }, K{ 5 }, B{ 1 };
    DataType data_type{ DataType::F32 };
    float    alpha{ 1.0 }, beta{ 0.0 };
    int      offset_src0{ 10 }, offset_src1{ 10 }, offset_dst{ 10 };
    float    scale_src0{ 1.0f / 255 }, scale_src1{ 1.0f / 255 }, scale_dst{ 1.0f / 255 };
    int32_t  dst_multiplier{ 0 }, dst_shift{ 0 };
    bool     add_bias{ true };
};

/** Main program for gemm test
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments
 *
 */
int main(int argc, char **argv)
{
    return utils::run_example<CLGEMMValidateExample>(argc, argv);
}
