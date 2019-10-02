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
#include "arm_compute/runtime/BlobLifetimeManager.h"
#include "arm_compute/runtime/Memory.h"
#include "arm_compute/runtime/MemoryGroup.h"
#include "arm_compute/runtime/MemoryManagerOnDemand.h"
#include "arm_compute/runtime/PoolManager.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "utils/TypePrinter.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
/** Mock class of memory manageable objects */
class MockMemoryManageable : public IMemoryManageable
{
public:
    void associate_memory_group(IMemoryGroup *memory_group) override
    {
        ARM_COMPUTE_UNUSED(memory_group);
    }
};
/** Creates a lifetime of three objects where the two of them can share the same underlying within the given scope
 *
 * @warning Contents and pointers of the objects registered will be invalid at the end of the function thus do not dereference.
 *
 * @param[in] mg The memory group under which the manageable objects will be registered
 */
void generate_lifetime_info(MemoryGroup &mg)
{
    MockMemoryManageable a{}, b{}, c{};
    Memory               m_a{}, m_b{}, m_c{};

    // Generate a custom lifetime for the objects
    mg.manage(&a);
    mg.manage(&b);
    mg.finalize_memory(&a, m_a, 12U /* size */, 8U /* alignment */);
    mg.manage(&c);
    mg.finalize_memory(&b, m_b, 128U /* size */, 16U /* alignment */);
    mg.finalize_memory(&c, m_c, 32U /* size */, 0U /* alignment */);
}
} // namespace
TEST_SUITE(UNIT)
TEST_SUITE(LifetimeManager)

/** Validate memory group register */
TEST_CASE(MemoryGroupRegister, framework::DatasetMode::ALL)
{
    auto        lft_mgr  = std::make_shared<BlobLifetimeManager>();
    auto        pool_mgr = std::make_shared<PoolManager>();
    auto        mm       = std::make_shared<MemoryManagerOnDemand>(lft_mgr, pool_mgr);
    MemoryGroup mg(mm);

    // Register group
    lft_mgr->register_group(&mg);

    // Generate lifetime information
    generate_lifetime_info(mg);

    // Validate lifetime manager state
    ARM_COMPUTE_EXPECT(lft_mgr->info().size() == 2, framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(lft_mgr->info()[0].size == 128, framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(lft_mgr->info()[0].alignment == 16, framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(lft_mgr->info()[0].owners == 1, framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(lft_mgr->info()[1].size == 32, framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(lft_mgr->info()[1].alignment == 8, framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(lft_mgr->info()[1].owners == 2, framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(mg.mappings().size() == 3, framework::LogLevel::ERRORS);
}

/** Validate memory group release */
TEST_CASE(MemoryGroupRelease, framework::DatasetMode::ALL)
{
    auto        lft_mgr  = std::make_shared<BlobLifetimeManager>();
    auto        pool_mgr = std::make_shared<PoolManager>();
    auto        mm       = std::make_shared<MemoryManagerOnDemand>(lft_mgr, pool_mgr);
    MemoryGroup mg(mm);

    // Register group
    lft_mgr->register_group(&mg);

    // Generate lifetime information
    generate_lifetime_info(mg);

    // Check group mappings
    ARM_COMPUTE_EXPECT(mg.mappings().size() == 3, framework::LogLevel::ERRORS);

    // Release group and validate its mappings
    lft_mgr->release_group(&mg);
    ARM_COMPUTE_EXPECT(mg.mappings().size() == 0, framework::LogLevel::ERRORS);
}

TEST_SUITE_END() // LifetimeManager
TEST_SUITE_END()
} // namespace validation
} // namespace test
} // namespace arm_compute
