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
#include "arm_compute/core/CL/kernels/CLDirectConvolutionLayerKernel.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/tuners/BifrostTuner.h"
#include "support/ToolchainSupport.h"
#include "tests/Utils.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
TEST_SUITE(CL)
TEST_SUITE(UNIT)
TEST_SUITE(Tuner)

/** Validates static tuning of Bifrost tuner */
TEST_CASE(BifrostTunerSimple, framework::DatasetMode::ALL)
{
    // Create tuner
    tuners::BifrostTuner tuner;

    // Create tensors
    auto src     = create_tensor<CLTensor>(TensorShape(13U, 13U, 16U), DataType::F32);
    auto weights = create_tensor<CLTensor>(TensorShape(3U, 3U, 16U, 3U), DataType::F32);
    auto bias    = create_tensor<CLTensor>(TensorShape(3U), DataType::F32);
    auto dst     = create_tensor<CLTensor>(TensorShape(13U, 13U, 3U), DataType::F32);

    // Create kernel
    cl::NDRange                    fake_lws(2000);
    CLDirectConvolutionLayerKernel conv;
    conv.set_target(GPUTarget::G72);

    // Configure
    conv.configure(&src, &weights, &bias, &dst, PadStrideInfo(1, 1, 1, 1));

    // Hard-wire lws to kernel and validate lws
    conv.set_lws_hint(fake_lws);
    ARM_COMPUTE_EXPECT(conv.lws_hint()[0] == 2000, framework::LogLevel::ERRORS);

    // Tune kernel and validate
    tuner.tune_kernel_static(conv);
    ARM_COMPUTE_EXPECT(conv.lws_hint()[0] != 2000, framework::LogLevel::ERRORS);

    // Clear tuner
    CLScheduler::get().default_init();
}
TEST_SUITE_END()
TEST_SUITE_END()
TEST_SUITE_END()
} // namespace validation
} // namespace test
} // namespace arm_compute
