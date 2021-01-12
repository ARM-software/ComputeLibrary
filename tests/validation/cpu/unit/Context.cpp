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
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/validation/Validation.h"

#include "arm_compute/Acl.hpp"

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

/** Test-case for AclDestroyContext
 *
 * Validate that AclDestroyContext behaves as expected when invalid inputs as context are given
 *
 * Test Steps:
 *  - Call AclDestroyContext with null context
 *  - Confirm that AclInvalidArgument is reported
 *  - Call AclDestroyContext on empty array
 *  - Confirm that AclInvalidArgument is reported
 *  - Call AclDestroyContext on an ACL object other than AclContext
 *  - Confirm that AclInvalidArgument is reported
 *  - Confirm that context is still nullptr
 */
TEST_CASE(DestroyInvalidContext, framework::DatasetMode::ALL)
{
    AclContext ctx = nullptr;
    std::array<char, 256> empty_array{};
    AclContext valid_ctx = nullptr;
    ARM_COMPUTE_ASSERT(AclCreateContext(&valid_ctx, AclCpu, nullptr) == AclStatus::AclSuccess);
    ARM_COMPUTE_ASSERT(AclDestroyContext(ctx) == AclStatus::AclInvalidArgument);
    ARM_COMPUTE_ASSERT(AclDestroyContext(reinterpret_cast<AclContext>(empty_array.data())) == AclStatus::AclInvalidArgument);
    ARM_COMPUTE_ASSERT(ctx == nullptr);
    ARM_COMPUTE_ASSERT(AclDestroyContext(valid_ctx) == AclStatus::AclSuccess);
}

/** Test-case for AclCreateContext and AclDestroy Context
 *
 * Validate that AclCreateContext can create and destroy a context
 *
 * Test Steps:
 *  - Call AclCreateContext with valid target
 *  - Confirm that context is not nullptr and error code is AclSuccess
 *  - Destroy context
 *  - Confirm that AclSuccess is reported
 */
TEST_CASE(SimpleContextCApi, framework::DatasetMode::ALL)
{
    AclContext ctx = nullptr;
    ARM_COMPUTE_ASSERT(AclCreateContext(&ctx, AclCpu, nullptr) == AclStatus::AclSuccess);
    ARM_COMPUTE_ASSERT(ctx != nullptr);
    ARM_COMPUTE_ASSERT(AclDestroyContext(ctx) == AclStatus::AclSuccess);
}

/** Test-case for Context from the C++ interface
 *
 * Test Steps:
 *  - Create a Context obejct
 *  - Confirm that StatusCode::Success is reported
 *  - Confirm that equality operator works
 *  - Confirm that inequality operator works
 */
TEST_CASE(SimpleContextCppApi, framework::DatasetMode::ALL)
{
    acl::StatusCode status = acl::StatusCode::Success;
    acl::Context    ctx(acl::Target::Cpu, &status);
    ARM_COMPUTE_ASSERT(status == acl::StatusCode::Success);

    auto ctx_eq = ctx;
    ARM_COMPUTE_ASSERT(ctx_eq == ctx);

    acl::Context ctx_ienq(acl::Target::Cpu, &status);
    ARM_COMPUTE_ASSERT(status == acl::StatusCode::Success);
    ARM_COMPUTE_ASSERT(ctx_ienq != ctx);
}

/** Test-case for CpuCapabilities
 *
 * Validate that AclCreateContext can create/destroy multiple contexts with different options
 *
 * Test Steps:
 *  - Call AclCreateContext with different targets
 *  - Confirm that AclSuccess is reported
 *  - Destroy all contexts
 *  - Confirm that AclSuccess is reported
 */
TEST_CASE(MultipleContexts, framework::DatasetMode::ALL)
{
    const unsigned int num_tests = 5;
    std::array<AclContext, num_tests> ctxs{};
    for(unsigned int i = 0; i < num_tests; ++i)
    {
        ARM_COMPUTE_ASSERT(AclCreateContext(&ctxs[i], AclTarget::AclCpu, nullptr) == AclStatus::AclSuccess);
        ARM_COMPUTE_ASSERT(ctxs[i] != nullptr);
        ARM_COMPUTE_ASSERT(AclDestroyContext(ctxs[i]) == AclStatus::AclSuccess);
    }
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
    AclContextOptions opts = acl_default_ctx_options;
    opts.capabilities      = AclCpuCapabilitiesDot | AclCpuCapabilitiesMmlaInt8 | AclCpuCapabilitiesSve2;
    arm_compute::cpu::CpuContext ctx(&opts);

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
