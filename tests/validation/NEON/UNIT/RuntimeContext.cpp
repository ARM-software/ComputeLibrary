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
#include "arm_compute/runtime/RuntimeContext.h"

#include "arm_compute/runtime/NEON/functions/NEActivationLayer.h"
#include "arm_compute/runtime/SchedulerFactory.h"
#include "arm_compute/runtime/Tensor.h"
#include "tests/Globals.h"
#include "tests/NEON/Accessor.h"
#include "tests/Utils.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/validation/Validation.h"
#include "tests/validation/reference/ActivationLayer.h"

#include <memory>
#include <random>
#if !defined(BARE_METAL)
#include <thread>
#endif // !defined(BARE_METAL)

namespace arm_compute
{
namespace test
{
namespace validation
{
TEST_SUITE(NEON)
TEST_SUITE(UNIT)
TEST_SUITE(RuntimeContext)

TEST_CASE(Scheduler, framework::DatasetMode::ALL)
{
    using namespace arm_compute;
    // Create a runtime context object
    RuntimeContext ctx;

    // Check if it's been initialised properly
    ARM_COMPUTE_EXPECT(ctx.scheduler() != nullptr, framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(ctx.asset_manager() == nullptr, framework::LogLevel::ERRORS);

    // Create a Scheduler
    auto scheduler = SchedulerFactory::create();
    ctx.set_scheduler(scheduler.get());
    // Check if the scheduler has been properly setup
    ARM_COMPUTE_EXPECT(ctx.scheduler() != nullptr, framework::LogLevel::ERRORS);

    // Create a new activation function
    NEActivationLayer act_layer(&ctx);

    Tensor src = create_tensor<Tensor>(TensorShape(32, 32), DataType::F32, 1);
    Tensor dst = create_tensor<Tensor>(TensorShape(32, 32), DataType::F32, 1);

    act_layer.configure(&src, &dst, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LINEAR));

    ARM_COMPUTE_EXPECT(src.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(dst.info()->is_resizable(), framework::LogLevel::ERRORS);

    // Allocate tensors
    src.allocator()->allocate();
    dst.allocator()->allocate();

    ARM_COMPUTE_EXPECT(!src.info()->is_resizable(), framework::LogLevel::ERRORS);

    float min_bound = 0;
    float max_bound = 0;
    std::tie(min_bound, max_bound) = get_activation_layer_test_bounds<float>(ActivationLayerInfo::ActivationFunction::LINEAR, DataType::F32);
    std::uniform_real_distribution<> distribution(min_bound, max_bound);
    library->fill(Accessor(src), distribution, 0);

    // Compute function
    act_layer.run();
}

#if !defined(BARE_METAL)
// This test tries scheduling work concurrently from two independent threads
TEST_CASE(MultipleThreadedScheduller, framework::DatasetMode::ALL)
{
    // Create a runtime context object for thread 1
    RuntimeContext ctx1;

    // Create a runtime context object for thread 2
    RuntimeContext ctx2;

    // Create a new activation function
    NEActivationLayer act_layer_thread0(&ctx1);
    NEActivationLayer act_layer_thread1(&ctx2);

    const TensorShape   tensor_shape(128, 128);
    Tensor              src_t0 = create_tensor<Tensor>(tensor_shape, DataType::F32, 1);
    Tensor              dst_t0 = create_tensor<Tensor>(tensor_shape, DataType::F32, 1);
    Tensor              src_t1 = create_tensor<Tensor>(tensor_shape, DataType::F32, 1);
    Tensor              dst_t1 = create_tensor<Tensor>(tensor_shape, DataType::F32, 1);
    ActivationLayerInfo activation_info(ActivationLayerInfo::ActivationFunction::LINEAR);

    act_layer_thread0.configure(&src_t0, &dst_t0, activation_info);
    act_layer_thread1.configure(&src_t1, &dst_t1, activation_info);

    ARM_COMPUTE_EXPECT(src_t0.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(dst_t0.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(src_t1.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(dst_t1.info()->is_resizable(), framework::LogLevel::ERRORS);

    // Allocate tensors
    src_t0.allocator()->allocate();
    dst_t0.allocator()->allocate();
    src_t1.allocator()->allocate();
    dst_t1.allocator()->allocate();

    ARM_COMPUTE_EXPECT(!src_t0.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(!src_t1.info()->is_resizable(), framework::LogLevel::ERRORS);

    float min_bound = 0;
    float max_bound = 0;
    std::tie(min_bound, max_bound) = get_activation_layer_test_bounds<float>(ActivationLayerInfo::ActivationFunction::LINEAR, DataType::F32);
    std::uniform_real_distribution<> distribution(min_bound, max_bound);
    library->fill(Accessor(src_t0), distribution, 0);
    library->fill(Accessor(src_t1), distribution, 0);

    std::thread neon_thread1([&] { act_layer_thread0.run(); });
    std::thread neon_thread2([&] { act_layer_thread1.run(); });

    neon_thread1.join();
    neon_thread2.join();

    Window window;
    window.use_tensor_dimensions(dst_t0.info()->tensor_shape());
    Iterator t0_it(&dst_t0, window);
    Iterator t1_it(&dst_t1, window);
    execute_window_loop(window, [&](const Coordinates &)
    {
        const bool match = (*reinterpret_cast<float *>(t0_it.ptr()) == *reinterpret_cast<float *>(t1_it.ptr()));
        ARM_COMPUTE_EXPECT(match, framework::LogLevel::ERRORS);
    },
    t0_it, t1_it);
}
#endif // !defined(BARE_METAL)

TEST_SUITE_END() // RuntimeContext
TEST_SUITE_END() // UNIT
TEST_SUITE_END() // Neon
} // namespace validation
} // namespace test
} // namespace arm_compute
