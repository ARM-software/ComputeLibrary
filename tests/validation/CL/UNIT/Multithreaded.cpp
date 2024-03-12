/*
 * Copyright (c) 2022 Arm Limited.
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
#include "arm_compute/runtime/RuntimeContext.h"

#include "tests/CL/CLAccessor.h"
#include "tests/framework/Macros.h"
#include "tests/framework/ParametersLibrary.h"
#include "tests/validation/Validation.h"
#include "arm_compute/runtime/CL/functions/CLActivationLayer.h"
#include "arm_compute/runtime/CL/functions/CLPixelWiseMultiplication.h"
#include "tests/validation/reference/ActivationLayer.h"
#include "tests/validation/reference/PixelWiseMultiplication.h"
#include <thread>

namespace arm_compute
{
namespace test
{
namespace validation
{
TEST_SUITE(CL)
TEST_SUITE(UNIT)
TEST_SUITE(RuntimeContext)
// This test tries scheduling work concurrently from two independent threads
TEST_CASE(MultipleThreadedScheduller, framework::DatasetMode::ALL)
{
    constexpr auto num_threads(16u);
    std::array<CLActivationLayer, num_threads>         func{};
    std::array<CLPixelWiseMultiplication, num_threads> pmul{};
    std::array<CLTensor, num_threads>                  s0{};
    std::array<CLTensor, num_threads>                  s1{};

    std::array<CLTensor, num_threads> st{};
    std::array<CLTensor, num_threads> dt{};

    const TensorShape         tensor_shape(128u, 4u, 5u);
    const ActivationLayerInfo ainfo(ActivationLayerInfo::ActivationFunction::LOGISTIC, 0.5f, 1.f);
    std::array<std::thread, num_threads> threads;
    auto ctx = parameters->get_ctx<CLTensor>();

    for(auto i = 0u; i < num_threads; ++i)
    {
        s0[i]   = create_tensor<CLTensor>(tensor_shape, DataType::F32, 1);
        s1[i]   = create_tensor<CLTensor>(tensor_shape, DataType::F32, 1);
        st[i]   = create_tensor<CLTensor>(tensor_shape, DataType::F32, 1);
        dt[i]   = create_tensor<CLTensor>(tensor_shape, DataType::F32, 1);
        func[i] = CLActivationLayer(ctx);
        pmul[i] = CLPixelWiseMultiplication();
        threads[i] =
            std::thread([&,i]
        {
            auto &s  = st[i];
            auto &t  = dt[i];
            auto &p0 = s0[i];
            auto &p1 = s1[i];
            pmul[i].configure(&p0, &p1, &s, 1.f, ConvertPolicy::WRAP, RoundingPolicy::TO_NEAREST_UP);
            func[i].configure(&s, &t, ainfo);
            s.allocator()->allocate();
            t.allocator()->allocate();
            p0.allocator()->allocate();
            p1.allocator()->allocate();
            library->fill_tensor_uniform(CLAccessor(p0), 0, -1.f, 1.f);
            library->fill_tensor_uniform(CLAccessor(p1), 0, -1.f, 1.f);
            pmul[i].run();
            func[i].run();
        });
    }

    for(auto &t : threads)
    {
        t.join();
    }

    SimpleTensor<float> rs{ tensor_shape, DataType::F32, 1 };
    SimpleTensor<float> ra{ tensor_shape, DataType::F32, 1 };
    SimpleTensor<float> rb{ tensor_shape, DataType::F32, 1 };
    library->fill_tensor_uniform(ra, 0, -1.f, 1.f);
    library->fill_tensor_uniform(rb, 0, -1.f, 1.f);
    const auto mul    = reference::pixel_wise_multiplication<float, float, float>(ra, rb, 1.f, ConvertPolicy::WRAP, RoundingPolicy::TO_NEAREST_UP, DataType::F32);
    const auto golden = reference::activation_layer<float>(mul, ainfo);
    for(auto &d : dt)
    {
        validate(CLAccessor(d), golden);
    }
}

TEST_SUITE_END() // MultipleThreadedScheduller
TEST_SUITE_END() // UNIT
TEST_SUITE_END() // CL
} // namespace validation
} // namespace test
} // namespace arm_compute
