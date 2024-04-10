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
#include "tests/validation/fixtures/UNIT/ContextFixture.h"

#include "src/gpu/cl/ClContext.h"

using namespace arm_compute::mlgo;

namespace arm_compute
{
namespace test
{
namespace validation
{
TEST_SUITE(CL)
TEST_SUITE(UNIT)
TEST_SUITE(Context)

EMPTY_BODY_FIXTURE_TEST_CASE(SimpleContextCApi, SimpleContextCApiFixture<AclTarget::AclGpuOcl>, framework::DatasetMode::ALL)
EMPTY_BODY_FIXTURE_TEST_CASE(SimpleContextCppApi, SimpleContextCppApiFixture<acl::Target::GpuOcl>, framework::DatasetMode::ALL)
EMPTY_BODY_FIXTURE_TEST_CASE(MultipleContexts, MultipleContextsFixture<AclTarget::AclGpuOcl>, framework::DatasetMode::ALL)

/** Test-case for MLGO kernel configuration file
 *
 * Validate that CpuCapabilities are set correctly
 *
 * Test Steps:
 *  - Create a file with the MLGO configuration
 *  - Pass the kernel file to the Context during creation
 *  - Validate that the MLGO file has been parsed successfully
 */
TEST_CASE(CheckMLGO, framework::DatasetMode::ALL)
{
    // Create test mlgo file
    std::string       mlgo_str      = R"_(

        <header>

        gemm-version, [1,2,1]
        ip-type,gpu
        </header>
        <heuristics-table>
        0, g76 , 8, f32, best-performance, static, gemm-type, [m,n,k,n]
        1, g76 , 8, f16, best-performance, static, gemm-config-reshaped, [m,n,k,n]
        </heuristics-table>
        <heuristic, 0>
        b , 0, var, m, ==, num, 10., 1, 2
        l , 1, gemm-type, reshaped
        b , 2, var, r_mn, >=, num, 2., 3, 6

        b , 3, var, n, >=, num, 200., 4, 5
        l, 4,                          gemm-type, reshaped-only-rhs
        l , 5, gemm-type, reshaped
        l , 6, gemm-type, reshaped-only-rhs
        </heuristic>

        <heuristic, 1>
        l ,0,gemm-config-reshaped,[4,2,4,2,8,1,0,1,0]
        </heuristic>

    )_";
    const std::string mlgo_filename = "test.mlgo";
    std::ofstream     ofs(mlgo_filename, std::ofstream::trunc);
    ARM_COMPUTE_EXPECT(ofs, framework::LogLevel::ERRORS);
    ofs << mlgo_str;
    ofs.close();

    acl::Context::Options opts;
    opts.copts.kernel_config_file = mlgo_filename.c_str();
    arm_compute::gpu::opencl::ClContext ctx(&opts.copts);

    const MLGOHeuristics &heuristics = ctx.mlgo();

    ARM_COMPUTE_EXPECT(heuristics.query_gemm_type(Query{ "g76", DataType::F32, 10, 1024, 20, 1 }).second == GEMMType::RESHAPED,
                       framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT((heuristics.query_gemm_config_reshaped(Query{ "g76", DataType::F16, 100, 100, 20, 32 }).second == GEMMConfigReshaped{ 4, 2, 4, 2, 8, true, false, true, false }),
                       framework::LogLevel::ERRORS);
}

TEST_SUITE_END() // Context
TEST_SUITE_END() // UNIT
TEST_SUITE_END() // CL
} // namespace validation
} // namespace test
} // namespace arm_compute
