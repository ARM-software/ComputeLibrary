/*
 * Copyright (c) 2018 ARM Limited.
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
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "arm_compute/runtime/CL/functions/CLComputeAllAnchors.h"
#include "arm_compute/runtime/CL/functions/CLGenerateProposalsLayer.h"
#include "arm_compute/runtime/CL/functions/CLSlice.h"
#include "tests/CL/CLAccessor.h"
#include "tests/CL/CLArrayAccessor.h"
#include "tests/Globals.h"
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
template <typename U, typename T>
inline void fill_tensor(U &&tensor, const std::vector<T> &v)
{
    std::memcpy(tensor.data(), v.data(), sizeof(T) * v.size());
}

const auto ComputeAllInfoDataset = framework::dataset::make("ComputeAllInfo",
{
    ComputeAnchorsInfo(10U, 10U, 1. / 16.f),
    ComputeAnchorsInfo(100U, 1U, 1. / 2.f),
    ComputeAnchorsInfo(100U, 1U, 1. / 4.f),
    ComputeAnchorsInfo(100U, 100U, 1. / 4.f),

});
} // namespace

TEST_SUITE(CL)
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
    ARM_COMPUTE_EXPECT(bool(CLGenerateProposalsLayer::validate(&scores.clone()->set_is_resizable(true),
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
using CLComputeAllAnchorsFixture = ComputeAllAnchorsFixture<CLTensor, CLAccessor, CLComputeAllAnchors, T>;

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
    fill_tensor(anchors_expected, std::vector<float> { -38, -16, 53, 31, -84, -40, 99, 55, -176, -88, 191, 103,
                                                       -22, -16, 69, 31, -68, -40, 115, 55, -160, -88, 207, 103,
                                                       -6, -16, 85, 31, -52, -40, 131, 55, -144, -88, 223, 103, -38,
                                                       0, 53, 47, -84, -24, 99, 71,
                                                       -176, -72, 191, 119, -22, 0, 69, 47, -68, -24, 115, 71, -160, -72, 207,
                                                       119, -6, 0, 85, 47, -52, -24, 131, 71, -144, -72, 223, 119, -38, 16, 53,
                                                       63, -84, -8, 99, 87, -176, -56, 191, 135, -22, 16, 69, 63, -68, -8, 115,
                                                       87, -160, -56, 207, 135, -6, 16, 85, 63, -52, -8, 131, 87, -144, -56, 223,
                                                       135, -38, 32, 53, 79, -84, 8, 99, 103, -176, -40, 191, 151, -22, 32, 69,
                                                       79, -68, 8, 115, 103, -160, -40, 207, 151, -6, 32, 85, 79, -52, 8, 131,
                                                       103, -144, -40, 223, 151
                                                     });

    CLTensor all_anchors;
    CLTensor anchors = create_tensor<CLTensor>(TensorShape(4, num_anchors), data_type);

    // Create and configure function
    CLComputeAllAnchors compute_anchors;
    compute_anchors.configure(&anchors, &all_anchors, ComputeAnchorsInfo(feature_width, feature_height, 1. / 16.0));
    anchors.allocator()->allocate();
    all_anchors.allocator()->allocate();

    fill_tensor(CLAccessor(anchors), std::vector<float> { -38, -16, 53, 31,
                                                          -84, -40, 99, 55,
                                                          -176, -88, 191, 103
                                                        });
    // Compute function
    compute_anchors.run();
    validate(CLAccessor(all_anchors), anchors_expected);
}

DATA_TEST_CASE(IntegrationTestCaseGenerateProposals, framework::DatasetMode::ALL, framework::dataset::make("DataType", { DataType::F32 }),
               data_type)
{
    const int values_per_roi = 4;
    const int num_anchors    = 2;
    const int feature_height = 4;
    const int feature_width  = 5;

    std::vector<float> scores_vector
    {
        5.44218998e-03f, 1.19207997e-03f, 1.12379994e-03f, 1.17181998e-03f,
        1.20544003e-03f, 6.17993006e-04f, 1.05261997e-05f, 8.91025957e-06f,
        9.29536981e-09f, 6.09605013e-05f, 4.72735002e-04f, 1.13482002e-10f,
        1.50015003e-05f, 4.45032993e-06f, 3.21612994e-08f, 8.02662980e-04f,
        1.40488002e-04f, 3.12508007e-07f, 3.02616991e-06f, 1.97759000e-08f,
        2.66913995e-02f, 5.26766013e-03f, 5.05053019e-03f, 5.62100019e-03f,
        5.37420018e-03f, 5.26280981e-03f, 2.48894998e-04f, 1.06842002e-04f,
        3.92931997e-06f, 1.79388002e-03f, 4.79440019e-03f, 3.41609990e-07f,
        5.20430971e-04f, 3.34090000e-05f, 2.19159006e-07f, 2.28786003e-03f,
        5.16703985e-05f, 4.04523007e-06f, 1.79227004e-06f, 5.32449000e-08f
    };

    std::vector<float> bbx_vector
    {
        -1.65040009e-02f, -1.84051003e-02f, -1.85930002e-02f, -2.08263006e-02f,
        -1.83814000e-02f, -2.89172009e-02f, -3.89706008e-02f, -7.52277970e-02f,
        -1.54091999e-01f, -2.55433004e-02f, -1.77490003e-02f, -1.10340998e-01f,
        -4.20190990e-02f, -2.71421000e-02f, 6.89801015e-03f, 5.71171008e-02f,
        -1.75665006e-01f, 2.30021998e-02f, 3.08554992e-02f, -1.39333997e-02f,
        3.40579003e-01f, 3.91070992e-01f, 3.91624004e-01f, 3.92527014e-01f,
        3.91445011e-01f, 3.79328012e-01f, 4.26631987e-01f, 3.64892989e-01f,
        2.76894987e-01f, 5.13985991e-01f, 3.79999995e-01f, 1.80457994e-01f,
        4.37402993e-01f, 4.18545991e-01f, 2.51549989e-01f, 4.48318988e-01f,
        1.68564007e-01f, 4.65440989e-01f, 4.21891987e-01f, 4.45928007e-01f,
        3.27155995e-03f, 3.71480011e-03f, 3.60032008e-03f, 4.27092984e-03f,
        3.74579988e-03f, 5.95752988e-03f, -3.14473989e-03f, 3.52022005e-03f,
        -1.88564006e-02f, 1.65188999e-03f, 1.73791999e-03f, -3.56074013e-02f,
        -1.66615995e-04f, 3.14146001e-03f, -1.11830998e-02f, -5.35363983e-03f,
        6.49790000e-03f, -9.27671045e-03f, -2.83346009e-02f, -1.61233004e-02f,
        -2.15505004e-01f, -2.19910994e-01f, -2.20872998e-01f, -2.12831005e-01f,
        -2.19145000e-01f, -2.27687001e-01f, -3.43973994e-01f, -2.75869995e-01f,
        -3.19516987e-01f, -2.50418007e-01f, -2.48537004e-01f, -5.08224010e-01f,
        -2.28724003e-01f, -2.82402009e-01f, -3.75815988e-01f, -2.86352992e-01f,
        -5.28333001e-02f, -4.43836004e-01f, -4.55134988e-01f, -4.34897989e-01f,
        -5.65053988e-03f, -9.25739005e-04f, -1.06790999e-03f, -2.37016007e-03f,
        -9.71166010e-04f, -8.90910998e-03f, -1.17592998e-02f, -2.08992008e-02f,
        -4.94231991e-02f, 6.63906988e-03f, 3.20469006e-03f, -6.44695014e-02f,
        -3.11607006e-03f, 2.02738005e-03f, 1.48096997e-02f, 4.39785011e-02f,
        -8.28424022e-02f, 3.62076014e-02f, 2.71668993e-02f, 1.38250999e-02f,
        6.76669031e-02f, 1.03252999e-01f, 1.03255004e-01f, 9.89722982e-02f,
        1.03646003e-01f, 4.79663983e-02f, 1.11014001e-01f, 9.31736007e-02f,
        1.15768999e-01f, 1.04014002e-01f, -8.90677981e-03f, 1.13103002e-01f,
        1.33085996e-01f, 1.25405997e-01f, 1.50051996e-01f, -1.13038003e-01f,
        7.01059997e-02f, 1.79651007e-01f, 1.41055003e-01f, 1.62841007e-01f,
        -1.00247003e-02f, -8.17587040e-03f, -8.32176022e-03f, -8.90108012e-03f,
        -8.13035015e-03f, -1.77263003e-02f, -3.69572006e-02f, -3.51580009e-02f,
        -5.92143014e-02f, -1.80795006e-02f, -5.46086021e-03f, -4.10550982e-02f,
        -1.83081999e-02f, -2.15411000e-02f, -1.17953997e-02f, 3.33894007e-02f,
        -5.29635996e-02f, -6.97528012e-03f, -3.15250992e-03f, -3.27355005e-02f,
        1.29676998e-01f, 1.16080999e-01f, 1.15947001e-01f, 1.21797003e-01f,
        1.16089001e-01f, 1.44875005e-01f, 1.15617000e-01f, 1.31586999e-01f,
        1.74735002e-02f, 1.21973999e-01f, 1.31596997e-01f, 2.48907991e-02f,
        6.18605018e-02f, 1.12855002e-01f, -6.99798986e-02f, 9.58312973e-02f,
        1.53593004e-01f, -8.75087008e-02f, -4.92327996e-02f, -3.32239009e-02f
    };

    std::vector<float> anchors_vector{ -38, -16, 53, 31,
                                       -120, -120, 135, 135 };

    SimpleTensor<float> proposals_expected(TensorShape(5, 9), DataType::F32);
    fill_tensor(proposals_expected, std::vector<float> { 0, 0, 0, 79, 59,
                                                         0, 0, 5.0005703f, 52.63237f, 43.69501495f,
                                                         0, 24.13628387f, 7.51243401f, 79, 46.06628418f,
                                                         0, 0, 7.50924301f, 68.47792816f, 46.03357315f,
                                                         0, 0, 23.09477997f, 51.61448669f, 59,
                                                         0, 0, 39.52141571f, 52.44710541f, 59,
                                                         0, 23.57396317f, 29.98791885f, 79, 59,
                                                         0, 0, 41.90219116f, 79, 59,
                                                         0, 0, 23.30098343f, 79, 59
                                                       });

    SimpleTensor<float> scores_expected(TensorShape(9), DataType::F32);
    fill_tensor(scores_expected, std::vector<float>
    {
        2.66913995e-02f,
        5.44218998e-03f,
        1.20544003e-03f,
        1.19207997e-03f,
        6.17993006e-04f,
        4.72735002e-04f,
        6.09605013e-05f,
        1.50015003e-05f,
        8.91025957e-06f
    });

    // Inputs
    CLTensor scores      = create_tensor<CLTensor>(TensorShape(feature_width, feature_height, num_anchors), data_type);
    CLTensor bbox_deltas = create_tensor<CLTensor>(TensorShape(feature_width, feature_height, values_per_roi * num_anchors), data_type);
    CLTensor anchors     = create_tensor<CLTensor>(TensorShape(values_per_roi, num_anchors), data_type);

    // Outputs
    CLTensor proposals;
    CLTensor num_valid_proposals;
    CLTensor scores_out;
    num_valid_proposals.allocator()->init(TensorInfo(TensorShape(1), 1, DataType::U32));

    CLGenerateProposalsLayer generate_proposals;
    generate_proposals.configure(&scores, &bbox_deltas, &anchors, &proposals, &scores_out, &num_valid_proposals,
                                 GenerateProposalsInfo(80, 60, 0.166667f, 1 / 16.0, 6000, 300, 0.7f, 16.0f));

    // Allocate memory for input/output tensors
    scores.allocator()->allocate();
    bbox_deltas.allocator()->allocate();
    anchors.allocator()->allocate();
    proposals.allocator()->allocate();
    num_valid_proposals.allocator()->allocate();
    scores_out.allocator()->allocate();

    // Fill inputs
    fill_tensor(CLAccessor(scores), scores_vector);
    fill_tensor(CLAccessor(bbox_deltas), bbx_vector);
    fill_tensor(CLAccessor(anchors), anchors_vector);

    // Run operator
    generate_proposals.run();

    // Gather num_valid_proposals
    num_valid_proposals.map();
    const uint32_t N = *reinterpret_cast<uint32_t *>(num_valid_proposals.ptr_to_element(Coordinates(0, 0)));
    num_valid_proposals.unmap();

    // Select the first N entries of the proposals
    CLTensor proposals_final;
    CLSlice  select_proposals;
    select_proposals.configure(&proposals, &proposals_final, Coordinates(0, 0), Coordinates(values_per_roi + 1, N));
    proposals_final.allocator()->allocate();
    select_proposals.run();

    // Select the first N entries of the proposals
    CLTensor scores_final;
    CLSlice  select_scores;
    select_scores.configure(&scores_out, &scores_final, Coordinates(0), Coordinates(N));
    scores_final.allocator()->allocate();
    select_scores.run();

    const RelativeTolerance<float> tolerance_f32(1e-6f);
    // Validate the output
    validate(CLAccessor(proposals_final), proposals_expected, tolerance_f32);
    validate(CLAccessor(scores_final), scores_expected, tolerance_f32);
}

FIXTURE_DATA_TEST_CASE(ComputeAllAnchors, CLComputeAllAnchorsFixture<float>, framework::DatasetMode::ALL,
                       combine(combine(framework::dataset::make("NumAnchors", { 2, 4, 8 }), ComputeAllInfoDataset), framework::dataset::make("DataType", { DataType::F32 })))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END() // FP32

TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(ComputeAllAnchors, CLComputeAllAnchorsFixture<half>, framework::DatasetMode::ALL,
                       combine(combine(framework::dataset::make("NumAnchors", { 2, 4, 8 }), ComputeAllInfoDataset), framework::dataset::make("DataType", { DataType::F16 })))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END() // FP16
TEST_SUITE_END() // Float

TEST_SUITE_END() // GenerateProposals
TEST_SUITE_END() // CL

} // namespace validation
} // namespace test
} // namespace arm_compute
