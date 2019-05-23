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
#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/SimpleTensor.h"
#include "tests/SimpleTensorAccessor.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"

#include "tests/validation/Validation.h"
#include "tests/validation/reference/ConvolutionLayer.h"
#include "tests/validation/reference/DFT.h"

#include <random>

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
auto shapes_1d_dft = framework::dataset::make("TensorShape", { TensorShape(33U),
                                                               TensorShape(8U),
                                                               TensorShape(23U, 7U),
                                                               TensorShape(16U, 8U, 4U)
                                                             });

auto shapes_2d_dft = framework::dataset::make("TensorShape", { TensorShape(33U, 14U),
                                                               TensorShape(8U, 9U),
                                                               TensorShape(23U, 7U, 3U),
                                                               TensorShape(16U, 8U, 4U)
                                                             });

auto conv_dataset_dft = framework::dataset::zip(framework::dataset::zip(framework::dataset::make("InputShape", { TensorShape(8U, 7U, 3U, 2U),
                                                                                                                 TensorShape(18U, 22U, 4U),
                                                                                                                 TensorShape(32U, 48U, 8U)
                                                                                                               }),
                                                                        framework::dataset::make("WeightShape", { TensorShape(3U, 3U, 3U, 6U),
                                                                                                                  TensorShape(5U, 5U, 4U, 3U),
                                                                                                                  TensorShape(9U, 9U, 8U, 3U)
                                                                                                                })),
                                                framework::dataset::make("ConvInfo", { PadStrideInfo(1, 1, 1, 1),
                                                                                       PadStrideInfo(1, 1, 2, 2),
                                                                                       PadStrideInfo(1, 1, 4, 4)
                                                                                     }));
} // namespace
TEST_SUITE(CPP)
TEST_SUITE(DFT)

TEST_SUITE(DFT1D)
DATA_TEST_CASE(Real, framework::DatasetMode::ALL, shapes_1d_dft,
               shape)
{
    SimpleTensor<float>                   src{ shape, DataType::F32, 1 };
    std::uniform_real_distribution<float> distribution(-5.f, 5.f);
    library->fill(src, distribution, 0);

    const bool is_odd = shape.x() % 2;

    // Forward pass
    auto forward = reference::rdft_1d(src);
    // Backward pass
    auto backward = reference::ridft_1d(forward, is_odd);

    // Validate with input
    validate(SimpleTensorAccessor<float>(src), backward, RelativeTolerance<float>(0.1f));
}

DATA_TEST_CASE(Complex, framework::DatasetMode::ALL, shapes_1d_dft,
               shape)
{
    SimpleTensor<float>                   src{ shape, DataType::F32, 2 };
    std::uniform_real_distribution<float> distribution(-5.f, 5.f);
    library->fill(src, distribution, 0);

    // Forward pass
    auto forward = reference::dft_1d(src, reference::FFTDirection::Forward);
    // Backward pass
    auto backward = reference::dft_1d(forward, reference::FFTDirection::Inverse);

    // Validate with input
    validate(SimpleTensorAccessor<float>(src), backward, RelativeTolerance<float>(0.1f));
}
TEST_SUITE_END() // DFT1D

TEST_SUITE(DFT2D)
DATA_TEST_CASE(Real, framework::DatasetMode::ALL, shapes_2d_dft,
               shape)
{
    SimpleTensor<float>                   src{ shape, DataType::F32, 1 };
    std::uniform_real_distribution<float> distribution(-5.f, 5.f);
    library->fill(src, distribution, 0);

    const bool is_odd = shape.x() % 2;

    // Forward pass
    auto forward = reference::rdft_2d(src);
    // Backward pass
    auto backward = reference::ridft_2d(forward, is_odd);

    // Validate with input
    validate(SimpleTensorAccessor<float>(src), backward, RelativeTolerance<float>(0.1f));
}

DATA_TEST_CASE(Complex, framework::DatasetMode::ALL, shapes_2d_dft,
               shape)
{
    SimpleTensor<float>                   src{ shape, DataType::F32, 2 };
    std::uniform_real_distribution<float> distribution(-5.f, 5.f);
    library->fill(src, distribution, 0);

    // Forward pass
    auto forward = reference::dft_2d(src, reference::FFTDirection::Forward);
    // Backward pass
    auto backward = reference::dft_2d(forward, reference::FFTDirection::Inverse);

    // Validate with input
    validate(SimpleTensorAccessor<float>(src), backward, RelativeTolerance<float>(0.1f));
}
TEST_SUITE_END() // DFT2D

TEST_SUITE(Conv)
DATA_TEST_CASE(Real2Real, framework::DatasetMode::ALL, conv_dataset_dft,
               shape_in, shape_w, conv_info)
{
    std::uniform_real_distribution<float> distribution(-1.f, 1.f);
    std::uniform_real_distribution<float> distribution_b(0.f, 0.f);

    SimpleTensor<float> src{ shape_in, DataType::F32, 1 };
    SimpleTensor<float> w{ shape_w, DataType::F32, 1 };
    SimpleTensor<float> b{ TensorShape(shape_w[3]), DataType::F32, 1 };

    library->fill(src, distribution, 0);
    library->fill(w, distribution, 1);
    library->fill(b, distribution_b, 2);

    const auto  output_wh = arm_compute::scaled_dimensions(shape_in.x(), shape_in.y(), shape_w.x(), shape_w.y(), conv_info);
    TensorShape dst_shape = shape_in;
    dst_shape.set(0, output_wh.first);
    dst_shape.set(1, output_wh.second);
    dst_shape.set(2, shape_w[3]);

    // FFT based convolution
    auto dst = reference::conv2d_dft(src, w, conv_info);
    // Reference convolution
    auto dst_ref = reference::convolution_layer(src, w, b, dst_shape, conv_info);

    // Validate with input
    validate(SimpleTensorAccessor<float>(dst), dst_ref, RelativeTolerance<float>(0.1f), 0.f, AbsoluteTolerance<float>(0.001f));
}
TEST_SUITE_END() // Conv

TEST_SUITE_END() // DFT
TEST_SUITE_END() // CPP
} // namespace validation
} // namespace test
} // namespace arm_compute
