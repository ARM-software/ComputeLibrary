/*
 * Copyright (c) 2019-2021 Arm Limited.
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
#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "arm_compute/runtime/NEON/functions/NEGenerateProposalsLayer.h"
#include "arm_compute/runtime/NEON/functions/NEPermute.h"
#include "arm_compute/runtime/NEON/functions/NESlice.h"
#include "src/core/NEON/kernels/NEGenerateProposalsLayerKernel.h"
#include "tests/Globals.h"
#include "tests/NEON/Accessor.h"
#include "tests/NEON/ArrayAccessor.h"
#include "tests/NEON/Helper.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/ComputeAllAnchorsFixture.h"
#include "utils/TypePrinter.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
using NEComputeAllAnchors = NESynthetizeFunction<NEComputeAllAnchorsKernel>;

template <typename U, typename T>
inline void fill_tensor(U &&tensor, const std::vector<T> &v)
{
    std::memcpy(tensor.data(), v.data(), sizeof(T) * v.size());
}

template <typename T>
inline void fill_tensor(Accessor &&tensor, const std::vector<T> &v)
{
    if(tensor.data_layout() == DataLayout::NCHW)
    {
        std::memcpy(tensor.data(), v.data(), sizeof(T) * v.size());
    }
    else
    {
        const int channels = tensor.shape()[0];
        const int width    = tensor.shape()[1];
        const int height   = tensor.shape()[2];
        for(int x = 0; x < width; ++x)
        {
            for(int y = 0; y < height; ++y)
            {
                for(int c = 0; c < channels; ++c)
                {
                    *(reinterpret_cast<T *>(tensor(Coordinates(c, x, y)))) = *(reinterpret_cast<const T *>(v.data() + x + y * width + c * height * width));
                }
            }
        }
    }
}

const auto ComputeAllInfoDataset = framework::dataset::make("ComputeAllInfo",
{
    ComputeAnchorsInfo(10U, 10U, 1. / 16.f),
    ComputeAnchorsInfo(100U, 1U, 1. / 2.f),
    ComputeAnchorsInfo(100U, 1U, 1. / 4.f),
    ComputeAnchorsInfo(100U, 100U, 1. / 4.f),

});

constexpr AbsoluteTolerance<int16_t> tolerance_qsymm16(1);
} // namespace

TEST_SUITE(NEON)
TEST_SUITE(GenerateProposals)

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(zip(zip(zip(zip(
               framework::dataset::make("scores", { TensorInfo(TensorShape(100U, 100U, 9U), 1, DataType::F32),
                                                    TensorInfo(TensorShape(100U, 100U, 9U), 1, DataType::F16), // Mismatching types
                                                    TensorInfo(TensorShape(100U, 100U, 9U), 1, DataType::F16), // Wrong deltas (number of transformation non multiple of 4)
                                                    TensorInfo(TensorShape(100U, 100U, 9U), 1, DataType::F16), // Wrong anchors (number of values per roi != 5)
                                                    TensorInfo(TensorShape(100U, 100U, 9U), 1, DataType::F16), // Output tensor num_valid_proposals not scalar
                                                    TensorInfo(TensorShape(100U, 100U, 9U), 1, DataType::F16)}), // num_valid_proposals not U32
               framework::dataset::make("deltas",{ TensorInfo(TensorShape(100U, 100U, 36U), 1, DataType::F32),
                                                   TensorInfo(TensorShape(100U, 100U, 36U), 1, DataType::F32),
                                                   TensorInfo(TensorShape(100U, 100U, 38U), 1, DataType::F32),
                                                   TensorInfo(TensorShape(100U, 100U, 38U), 1, DataType::F32),
                                                   TensorInfo(TensorShape(100U, 100U, 38U), 1, DataType::F32),
                                                   TensorInfo(TensorShape(100U, 100U, 38U), 1, DataType::F32)})),
               framework::dataset::make("anchors", { TensorInfo(TensorShape(4U, 9U), 1, DataType::F32),
                                                     TensorInfo(TensorShape(4U, 9U), 1, DataType::F32),
                                                     TensorInfo(TensorShape(4U, 9U), 1, DataType::F32),
                                                     TensorInfo(TensorShape(5U, 9U), 1, DataType::F32),
                                                     TensorInfo(TensorShape(4U, 9U), 1, DataType::F32),
                                                     TensorInfo(TensorShape(4U, 9U), 1, DataType::F32)})),
               framework::dataset::make("proposals", { TensorInfo(TensorShape(5U, 100U*100U*9U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(5U, 100U*100U*9U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(5U, 100U*100U*9U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(5U, 100U*100U*9U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(5U, 100U*100U*9U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(5U, 100U*100U*9U), 1, DataType::F32)})),
               framework::dataset::make("scores_out", { TensorInfo(TensorShape(100U*100U*9U), 1, DataType::F32),
                                                        TensorInfo(TensorShape(100U*100U*9U), 1, DataType::F32),
                                                        TensorInfo(TensorShape(100U*100U*9U), 1, DataType::F32),
                                                        TensorInfo(TensorShape(100U*100U*9U), 1, DataType::F32),
                                                        TensorInfo(TensorShape(100U*100U*9U), 1, DataType::F32),
                                                        TensorInfo(TensorShape(100U*100U*9U), 1, DataType::F32)})),
               framework::dataset::make("num_valid_proposals", { TensorInfo(TensorShape(1U, 1U), 1, DataType::U32),
                                                                 TensorInfo(TensorShape(1U, 1U), 1, DataType::U32),
                                                                 TensorInfo(TensorShape(1U, 1U), 1, DataType::U32),
                                                                 TensorInfo(TensorShape(1U, 1U), 1, DataType::U32),
                                                                 TensorInfo(TensorShape(1U, 10U), 1, DataType::U32),
                                                                 TensorInfo(TensorShape(1U, 1U), 1, DataType::F16)})),
               framework::dataset::make("generate_proposals_info", { GenerateProposalsInfo(10.f, 10.f, 1.f),
                                                                     GenerateProposalsInfo(10.f, 10.f, 1.f),
                                                                     GenerateProposalsInfo(10.f, 10.f, 1.f),
                                                                     GenerateProposalsInfo(10.f, 10.f, 1.f),
                                                                     GenerateProposalsInfo(10.f, 10.f, 1.f),
                                                                     GenerateProposalsInfo(10.f, 10.f, 1.f)})),
               framework::dataset::make("Expected", { true, false, false, false, false, false })),
        scores, deltas, anchors, proposals, scores_out, num_valid_proposals, generate_proposals_info, expected)
{
    ARM_COMPUTE_EXPECT(bool(NEGenerateProposalsLayer::validate(&scores.clone()->set_is_resizable(true),
                                                          &deltas.clone()->set_is_resizable(true),
                                                          &anchors.clone()->set_is_resizable(true),
                                                          &proposals.clone()->set_is_resizable(true),
                                                          &scores_out.clone()->set_is_resizable(true),
                                                          &num_valid_proposals.clone()->set_is_resizable(true),
                                                          generate_proposals_info)) == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

template <typename T>
using NEComputeAllAnchorsFixture = ComputeAllAnchorsFixture<Tensor, Accessor, NEComputeAllAnchors, T>;

TEST_SUITE(Float)
TEST_SUITE(FP32)
DATA_TEST_CASE(IntegrationTestCaseAllAnchors, framework::DatasetMode::ALL, framework::dataset::make("DataType", { DataType::F32 }),
               data_type)
{
    const int values_per_roi = 4;
    const int num_anchors    = 3;
    const int feature_height = 4;
    const int feature_width  = 3;

    SimpleTensor<float> anchors_expected(TensorShape(values_per_roi, feature_width * feature_height * num_anchors), DataType::F32);
    fill_tensor(anchors_expected, std::vector<float> { -26, -19, 87, 86,
                                                       -81, -27, 58, 63,
                                                       -44, -15, 55, 36,
                                                       -10, -19, 103, 86,
                                                       -65, -27, 74, 63,
                                                       -28, -15, 71, 36,
                                                       6, -19, 119, 86,
                                                       -49, -27, 90, 63,
                                                       -12, -15, 87, 36,
                                                       -26, -3, 87, 102,
                                                       -81, -11, 58, 79,
                                                       -44, 1, 55, 52,
                                                       -10, -3, 103, 102,
                                                       -65, -11, 74, 79,
                                                       -28, 1, 71, 52,
                                                       6, -3, 119, 102,
                                                       -49, -11, 90, 79,
                                                       -12, 1, 87, 52,
                                                       -26, 13, 87, 118,
                                                       -81, 5, 58, 95,
                                                       -44, 17, 55, 68,
                                                       -10, 13, 103, 118,
                                                       -65, 5, 74, 95,
                                                       -28, 17, 71, 68,
                                                       6, 13, 119, 118,
                                                       -49, 5, 90, 95,
                                                       -12, 17, 87, 68,
                                                       -26, 29, 87, 134,
                                                       -81, 21, 58, 111,
                                                       -44, 33, 55, 84,
                                                       -10, 29, 103, 134,
                                                       -65, 21, 74, 111,
                                                       -28, 33, 71, 84,
                                                       6, 29, 119, 134,
                                                       -49, 21, 90, 111,
                                                       -12, 33, 87, 84
                                                     });

    Tensor all_anchors;
    Tensor anchors = create_tensor<Tensor>(TensorShape(4, num_anchors), data_type);

    // Create and configure function
    NEComputeAllAnchors compute_anchors;
    compute_anchors.configure(&anchors, &all_anchors, ComputeAnchorsInfo(feature_width, feature_height, 1. / 16.0));
    anchors.allocator()->allocate();
    all_anchors.allocator()->allocate();

    fill_tensor(Accessor(anchors), std::vector<float> { -26, -19, 87, 86,
                                                        -81, -27, 58, 63,
                                                        -44, -15, 55, 36
                                                      });
    // Compute function
    compute_anchors.run();
    validate(Accessor(all_anchors), anchors_expected);
}

DATA_TEST_CASE(IntegrationTestCaseGenerateProposals, framework::DatasetMode::ALL, combine(framework::dataset::make("DataType", { DataType::F32 }),
                                                                                          framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
               data_type, data_layout)
{
    const int values_per_roi = 4;
    const int num_anchors    = 2;
    const int feature_height = 4;
    const int feature_width  = 5;

    std::vector<float> scores_vector
    {
        5.055894435664012e-04f, 1.270304909820112e-03f, 2.492271113912067e-03f, 5.951663827809190e-03f,
        7.846917156877404e-03f, 6.776275276294789e-03f, 6.761571012891965e-03f, 4.898292096237725e-03f,
        6.044472332578605e-04f, 3.203334118759474e-03f, 2.947527908919908e-03f, 6.313238560015770e-03f,
        7.931767757095738e-03f, 8.764345805102866e-03f, 7.325012199914913e-03f, 4.317069470446271e-03f,
        2.372537409795522e-03f, 1.589227460352735e-03f, 7.419477503600818e-03f, 3.157690354133824e-05f,
        1.125915135986472e-03f, 9.865363483872330e-03f, 2.429454743386769e-03f, 2.724460564167563e-03f,
        7.670409838207963e-03f, 5.558891552328172e-03f, 7.876904873099614e-03f, 6.824746047239291e-03f,
        7.023817548067892e-03f, 3.651314909238673e-04f, 6.720443709032501e-03f, 5.935615511606155e-03f,
        2.837349642759774e-03f, 1.787235113610299e-03f, 4.538568889918262e-03f, 3.391510678188818e-03f,
        7.328474239481874e-03f, 6.306967923936016e-03f, 8.102218904895860e-04f, 3.366646521610209e-03f
    };

    std::vector<float> bbx_vector
    {
        5.066650471856862e-03, -7.638671742936328e-03, 2.549596503988635e-03, -8.316416756423296e-03,
        -2.397471917924575e-04, 7.370595187754891e-03, -2.771880178185262e-03, 3.958364873973579e-03,
        4.493661094712284e-03, 2.016487051533088e-03, -5.893883038142033e-03, 7.570636080807809e-03,
        -1.395511229386785e-03, 3.686686052704696e-03, -7.738166245767079e-03, -1.947306329828059e-03,
        -9.299719716045681e-03, -3.476410493413708e-03, -2.390761190919604e-03, 4.359281254364210e-03,
        -2.135251160164030e-04, 9.203299843371962e-03, 4.042322775006053e-03, -9.464271243910754e-03,
        2.566239543229305e-03, -9.691093900220627e-03, -4.019283034310979e-03, 8.145470429508792e-03,
        7.345087308315662e-04, 7.049642787384043e-03, -2.768492313674294e-03, 6.997160053405803e-03,
        6.675346697112969e-03, 2.353293365652274e-03, -3.612002585241749e-04, 1.592076522068768e-03,
        -8.354188900818149e-04, -5.232515333564140e-04, 6.946683728847089e-03, -8.469757407935994e-03,
        -8.985324496496555e-03, 4.885832859017961e-03, -7.662967577576512e-03, 7.284124004335807e-03,
        -5.812167510299458e-03, -5.760336800482398e-03, 6.040416930336549e-03, 5.861508595443691e-03,
        -5.509243096133549e-04, -2.006142470055888e-03, -7.205925340416066e-03, -1.117459082969758e-03,
        4.233247017623154e-03, 8.079257498201178e-03, 2.962639022639513e-03, 7.069474943472751e-03,
        -8.562946284971293e-03, -8.228634642768271e-03, -6.116245322799971e-04, -7.213122000180859e-03,
        1.693094399433209e-03, -4.287504459132290e-03, 8.740365683925144e-03, 3.751788160720638e-03,
        7.006764222862830e-03, 9.676754678358187e-03, -6.458757235812945e-03, -4.486506575589758e-03,
        -4.371087196816259e-03, 3.542166755953152e-03, -2.504808998699504e-03, 5.666601724512010e-03,
        -3.691862724546129e-03, 3.689809719085287e-03, 9.079930264704458e-03, 6.365127787359476e-03,
        2.881681788246101e-06, 9.991866069315165e-03, -1.104757466496565e-03, -2.668455405633477e-03,
        -1.225748887087659e-03, 6.530536159094015e-03, 3.629468917975644e-03, 1.374426066950348e-03,
        -2.404098881570632e-03, -4.791365049441602e-03, -2.970654027009094e-03, 7.807553690294366e-03,
        -1.198321129505323e-03, -3.574885336949881e-03, -5.380848303732298e-03, 9.705151282165116e-03,
        -1.005217683242201e-03, 9.178094036278405e-03, -5.615977269541644e-03, 5.333533158509859e-03,
        -2.817116206168516e-03, 6.672609782000503e-03, 6.575769501651313e-03, 8.987596634989362e-03,
        -1.283530791296188e-03, 1.687717120057778e-03, 3.242391851439037e-03, -7.312060454341677e-03,
        4.735335326324270e-03, -6.832367028817463e-03, -5.414854835884652e-03, -9.352380213755996e-03,
        -3.682662043703889e-03, -6.127508590419776e-04, -7.682256596819467e-03, 9.569532628790246e-03,
        -1.572157284518933e-03, -6.023034366859191e-03, -5.110873282582924e-03, -8.697072236660256e-03,
        -3.235150419663566e-03, -8.286320236471386e-03, -5.229472409112913e-03, 9.920785896115053e-03,
        -2.478413362126123e-03, -9.261324796935007e-03, 1.718512310840434e-04, 3.015875488208480e-03,
        -6.172932549255669e-03, -4.031715551985103e-03, -9.263878005853677e-03, -2.815310738453385e-03,
        7.075307462133643e-03, 1.404611747938669e-03, -1.518548732533266e-03, -9.293430941655778e-03,
        6.382186966633246e-03, 8.256835789169248e-03, 3.196907843506736e-03, 8.821615689753433e-03,
        -7.661543424832439e-03, 1.636273081822326e-03, -8.792373335756125e-03, 2.958775812049877e-03,
        -6.269300278071262e-03, 6.248285790856450e-03, -3.675414624536002e-03, -1.692616700318762e-03,
        4.126007647815893e-03, -9.155291689759584e-03, -8.432616039924004e-03, 4.899980636213323e-03,
        3.511535019681671e-03, -1.582745757177339e-03, -2.703657774917963e-03, 6.738168990840388e-03,
        4.300455303937919e-03, 9.618312854781494e-03, 2.762142918402472e-03, -6.590025003382154e-03,
        -2.071168373801788e-03, 8.613893943683627e-03, 9.411190295341036e-03, -6.129018930548372e-03
    };

    const std::vector<float> anchors_vector{ -26, -19, 87, 86, -81, -27, 58, 63 };
    SimpleTensor<float>      proposals_expected(TensorShape(5, 9), DataType::F32);
    fill_tensor(proposals_expected, std::vector<float>
    {
        0, 0, 0, 75.269, 64.4388,
        0, 21.9579, 13.0535, 119, 99,
        0, 38.303, 0, 119, 87.6447,
        0, 0, 0, 119, 64.619,
        0, 0, 20.7997, 74.0714, 99,
        0, 0, 0, 91.8963, 79.3724,
        0, 0, 4.42377, 58.1405, 95.1781,
        0, 0, 13.4405, 104.799, 99,
        0, 38.9066, 28.2434, 119, 99,

    });

    SimpleTensor<float> scores_expected(TensorShape(9), DataType::F32);
    fill_tensor(scores_expected, std::vector<float>
    {
        0.00986536,
        0.00876435,
        0.00784692,
        0.00767041,
        0.00732847,
        0.00682475,
        0.00672044,
        0.00631324,
        3.15769e-05
    });

    TensorShape scores_shape = TensorShape(feature_width, feature_height, num_anchors);
    TensorShape deltas_shape = TensorShape(feature_width, feature_height, values_per_roi * num_anchors);
    if(data_layout == DataLayout::NHWC)
    {
        permute(scores_shape, PermutationVector(2U, 0U, 1U));
        permute(deltas_shape, PermutationVector(2U, 0U, 1U));
    }
    // Inputs
    Tensor scores      = create_tensor<Tensor>(scores_shape, data_type, 1, QuantizationInfo(), data_layout);
    Tensor bbox_deltas = create_tensor<Tensor>(deltas_shape, data_type, 1, QuantizationInfo(), data_layout);
    Tensor anchors     = create_tensor<Tensor>(TensorShape(values_per_roi, num_anchors), data_type);

    // Outputs
    Tensor proposals;
    Tensor num_valid_proposals;
    Tensor scores_out;
    num_valid_proposals.allocator()->init(TensorInfo(TensorShape(1), 1, DataType::U32));

    NEGenerateProposalsLayer generate_proposals;
    generate_proposals.configure(&scores, &bbox_deltas, &anchors, &proposals, &scores_out, &num_valid_proposals,
                                 GenerateProposalsInfo(120, 100, 0.166667f, 1 / 16.0, 6000, 300, 0.7f, 16.0f));

    // Allocate memory for input/output tensors
    scores.allocator()->allocate();
    bbox_deltas.allocator()->allocate();
    anchors.allocator()->allocate();
    proposals.allocator()->allocate();
    num_valid_proposals.allocator()->allocate();
    scores_out.allocator()->allocate();
    // Fill inputs
    fill_tensor(Accessor(scores), scores_vector);
    fill_tensor(Accessor(bbox_deltas), bbx_vector);
    fill_tensor(Accessor(anchors), anchors_vector);

    // Run operator
    generate_proposals.run();
    // Gather num_valid_proposals
    const uint32_t N = *reinterpret_cast<uint32_t *>(num_valid_proposals.ptr_to_element(Coordinates(0, 0)));

    // Select the first N entries of the proposals
    Tensor  proposals_final;
    NESlice select_proposals;
    select_proposals.configure(&proposals, &proposals_final, Coordinates(0, 0), Coordinates(values_per_roi + 1, N));

    proposals_final.allocator()->allocate();
    select_proposals.run();

    // Select the first N entries of the proposals
    Tensor  scores_final;
    NESlice select_scores;
    select_scores.configure(&scores_out, &scores_final, Coordinates(0), Coordinates(N));
    scores_final.allocator()->allocate();
    select_scores.run();

    const RelativeTolerance<float> tolerance_f32(1e-5f);
    // Validate the output
    validate(Accessor(proposals_final), proposals_expected, tolerance_f32);
    validate(Accessor(scores_final), scores_expected, tolerance_f32);
}

FIXTURE_DATA_TEST_CASE(ComputeAllAnchors, NEComputeAllAnchorsFixture<float>, framework::DatasetMode::ALL,
                       combine(combine(framework::dataset::make("NumAnchors", { 2, 4, 8 }), ComputeAllInfoDataset), framework::dataset::make("DataType", { DataType::F32 })))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // FP32
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(ComputeAllAnchors, NEComputeAllAnchorsFixture<half>, framework::DatasetMode::ALL,
                       combine(combine(framework::dataset::make("NumAnchors", { 2, 4, 8 }), ComputeAllInfoDataset), framework::dataset::make("DataType", { DataType::F16 })))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // FP16
#endif           // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

TEST_SUITE_END() // Float

template <typename T>
using NEComputeAllAnchorsQuantizedFixture = ComputeAllAnchorsQuantizedFixture<Tensor, Accessor, NEComputeAllAnchors, T>;

TEST_SUITE(Quantized)
TEST_SUITE(QASYMM8)
FIXTURE_DATA_TEST_CASE(ComputeAllAnchors, NEComputeAllAnchorsQuantizedFixture<int16_t>, framework::DatasetMode::ALL,
                       combine(combine(combine(framework::dataset::make("NumAnchors", { 2, 4, 8 }), ComputeAllInfoDataset),
                                       framework::dataset::make("DataType", { DataType::QSYMM16 })),
                               framework::dataset::make("QuantInfo", { QuantizationInfo(0.125f, 0) })))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qsymm16);
}
TEST_SUITE_END() // QASYMM8
TEST_SUITE_END() // Quantized

TEST_SUITE_END() // GenerateProposals
TEST_SUITE_END() // Neon
} // namespace validation
} // namespace test
} // namespace arm_compute
