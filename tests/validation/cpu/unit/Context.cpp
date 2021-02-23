/*
 * Copyright (c) 2021 Arm Limited.
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
#include "tests/validation/fixtures/UNIT/Context.h"

#include "src/cpu/CpuContext.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
TEST_SUITE(CPU)
TEST_SUITE(UNIT)
TEST_SUITE(Context)

/** Test-case for AclCreateContext
 *
 * Validate that AclCreateContext behaves as expected on invalid target
 *
 * Test Steps:
 *  - Call AclCreateContext with invalid target
 *  - Confirm that AclUnsupportedTarget is reported
 *  - Confirm that context is still nullptr
 */
TEST_CASE(CreateContextWithInvalidTarget, framework::DatasetMode::ALL)
{
    AclTarget  invalid_target = static_cast<AclTarget>(-1);
    AclContext ctx            = nullptr;
    ARM_COMPUTE_ASSERT(AclCreateContext(&ctx, invalid_target, nullptr) == AclStatus::AclUnsupportedTarget);
    ARM_COMPUTE_ASSERT(ctx == nullptr);
}

/** Test-case for AclCreateContext
 *
 * Validate that AclCreateContext behaves as expected on invalid context options
 *
 * Test Steps:
 *  - Call AclCreateContext with valid target but invalid context options
 *  - Confirm that AclInvalidArgument is reported
 *  - Confirm that context is still nullptr
 */
TEST_CASE(CreateContextWithInvalidOptions, framework::DatasetMode::ALL)
{
    AclContextOptions invalid_ctx_opts;
    invalid_ctx_opts.mode               = static_cast<AclExecutionMode>(-1);
    invalid_ctx_opts.capabilities       = AclCpuCapabilitiesAuto;
    invalid_ctx_opts.max_compute_units  = 0;
    invalid_ctx_opts.enable_fast_math   = false;
    invalid_ctx_opts.kernel_config_file = "";
    AclContext ctx                      = nullptr;
    ARM_COMPUTE_ASSERT(AclCreateContext(&ctx, AclCpu, &invalid_ctx_opts) == AclStatus::AclInvalidArgument);
    ARM_COMPUTE_ASSERT(ctx == nullptr);
}

FIXTURE_TEST_CASE(DestroyInvalidContext, DestroyInvalidContextFixture<AclTarget::AclCpu>, framework::DatasetMode::ALL)
{
}
FIXTURE_TEST_CASE(SimpleContextCApi, SimpleContextCApiFixture<AclTarget::AclCpu>, framework::DatasetMode::ALL)
{
}
FIXTURE_TEST_CASE(SimpleContextCppApi, SimpleContextCppApiFixture<acl::Target::Cpu>, framework::DatasetMode::ALL)
{
}
FIXTURE_TEST_CASE(MultipleContexts, MultipleContextsFixture<AclTarget::AclCpu>, framework::DatasetMode::ALL)
{
}

/** Test-case for CpuCapabilities
 *
 * Validate that CpuCapabilities are set correctly
 *
 * Test Steps:
 *  - Create a context with a given list of capabilities
 *  - Confirm that AclSuccess is reported
 *  - Validate that all capabilities are set correctly
 */
TEST_CASE(CpuCapabilities, framework::DatasetMode::ALL)
{
    acl::Context::Options opts;
    opts.copts.capabilities = AclCpuCapabilitiesDot | AclCpuCapabilitiesMmlaInt8 | AclCpuCapabilitiesSve2;
    arm_compute::cpu::CpuContext ctx(&opts.copts);

    ARM_COMPUTE_ASSERT(ctx.capabilities().dot == true);
    ARM_COMPUTE_ASSERT(ctx.capabilities().mmla_int8 == true);
    ARM_COMPUTE_ASSERT(ctx.capabilities().sve2 == true);
    ARM_COMPUTE_ASSERT(ctx.capabilities().fp16 == false);

    arm_compute::cpu::CpuContext ctx_legacy(nullptr);
    ARM_COMPUTE_ASSERT(ctx_legacy.capabilities().neon == true);
}

TEST_SUITE_END() // Context
TEST_SUITE_END() // UNIT
TEST_SUITE_END() // CPU
} // namespace validation
} // namespace test
} // namespace arm_compute
