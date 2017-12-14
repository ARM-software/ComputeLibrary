/*
 * Copyright (c) 2017 ARM Limited.
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
#include "arm_compute/runtime/Allocator.h"
#include "arm_compute/runtime/BlobLifetimeManager.h"
#include "arm_compute/runtime/MemoryGroup.h"
#include "arm_compute/runtime/MemoryManagerOnDemand.h"
#include "arm_compute/runtime/NEON/functions/NENormalizationLayer.h"
#include "arm_compute/runtime/PoolManager.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "support/ToolchainSupport.h"
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/NEON/Accessor.h"
#include "tests/Utils.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
TEST_SUITE(NEON)
TEST_SUITE(UNIT)
TEST_SUITE(MemoryManager)

TEST_CASE(BlobMemoryManagerSimpleWithinFunctionLevel, framework::DatasetMode::ALL)
{
    Allocator allocator{};
    auto      lifetime_mgr = std::make_shared<BlobLifetimeManager>();
    auto      pool_mgr     = std::make_shared<PoolManager>();
    auto      mm           = std::make_shared<MemoryManagerOnDemand>(lifetime_mgr, pool_mgr);

    // Create tensors
    Tensor src = create_tensor<Tensor>(TensorShape(27U, 11U, 3U), DataType::F32, 1);
    Tensor dst = create_tensor<Tensor>(TensorShape(27U, 11U, 3U), DataType::F32, 1);

    // Create and configure function
    NENormalizationLayer norm_layer_1(mm);
    NENormalizationLayer norm_layer_2(mm);
    norm_layer_1.configure(&src, &dst, NormalizationLayerInfo(NormType::CROSS_MAP, 3));
    norm_layer_2.configure(&src, &dst, NormalizationLayerInfo(NormType::IN_MAP_1D, 3));

    ARM_COMPUTE_EXPECT(src.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(dst.info()->is_resizable(), framework::LogLevel::ERRORS);

    // Allocate tensors
    src.allocator()->allocate();
    dst.allocator()->allocate();

    ARM_COMPUTE_EXPECT(!src.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(!dst.info()->is_resizable(), framework::LogLevel::ERRORS);

    // Finalize memory manager
    mm->set_allocator(&allocator);
    mm->set_num_pools(1);
    mm->finalize();
    ARM_COMPUTE_EXPECT(mm->is_finalized(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(mm->lifetime_manager()->are_all_finalized(), framework::LogLevel::ERRORS);

    // Fill tensors
    arm_compute::test::library->fill_tensor_uniform(Accessor(src), 0);

    // Compute functions
    norm_layer_1.run();
    norm_layer_2.run();
}

TEST_SUITE_END()
TEST_SUITE_END()
TEST_SUITE_END()
} // namespace validation
} // namespace test
} // namespace arm_compute
