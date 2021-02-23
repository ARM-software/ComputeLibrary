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
#include "src/runtime/CL/mlgo/MLGOHeuristics.h"
#include "src/runtime/CL/mlgo/Utils.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"

using namespace arm_compute::mlgo;

namespace arm_compute
{
namespace test
{
namespace validation
{
TEST_SUITE(CL)
TEST_SUITE(UNIT)
TEST_SUITE(MLGOHeuristics)
TEST_CASE(CorrectDotMLGOShouldLoadCorrectly, framework::DatasetMode::ALL)
{
    std::string       mlgo_str = R"_(

        <header>

        gemm-version, [1,2,1]
        ip-type,gpu
        </header>
        <heuristics-table>
        0, g76 , 8, f32, best-performance, static, gemm-type, [m,n,k,n]

        1, g71 , 8, f16, best-performance, static, gemm-config-reshaped-only-rhs, [m,n,k,n]
        2, g76 , 8, f16, best-performance, static, gemm-config-reshaped, [m,n,k,n]
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
        b ,0,var, n, >, num, 100., 1, 4
        b ,1,var, r_mnk, <=, num, 20., 2, 3


        l ,2,gemm-config-reshaped-only-rhs, [4, 4,4,2,1,0,1]
        l ,3,gemm-config-reshaped-only-rhs,[ 2, 2,4,2,1,1, 1 ]
        b ,4,var, n, >=, num, 199.12, 5, 6
        l ,5,gemm-config-reshaped-only-rhs, [1, 4,3,4,0,0,0]
        l ,6,gemm-config-reshaped-only-rhs, [5, 4,4,5,1,1,0]
        </heuristic>

        <heuristic, 2>
        l ,0,gemm-config-reshaped,[4,2,4,2,8,1,0,1,0]

        </heuristic>

    )_";
    std::stringstream ss(mlgo_str);
    MLGOHeuristics    heuristics;
    heuristics.reload_from_stream(ss);

    ARM_COMPUTE_EXPECT(heuristics.query_gemm_type(Query{ "g76", DataType::F32, 10, 1024, 20, 1 }).second == GEMMType::RESHAPED, framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(heuristics.query_gemm_type(Query{ "g76", DataType::F32, 400, 201, 5, 1 }).second == GEMMType::RESHAPED_ONLY_RHS, framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(heuristics.query_gemm_type(Query{ "g76", DataType::F32, 400, 200, 199, 16 }).second == GEMMType::RESHAPED_ONLY_RHS, framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(heuristics.query_gemm_type(Query{ "g76", DataType::F32, 400, 199, 512, 4 }).second == GEMMType::RESHAPED, framework::LogLevel::ERRORS);

    ARM_COMPUTE_EXPECT((heuristics.query_gemm_config_reshaped_only_rhs(Query{ "g71", DataType::F16, 100, 1024, 20, 32 }).second == GEMMConfigReshapedOnlyRHS{ 4, 4, 4, 2, true, false, true }),
                       framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT((heuristics.query_gemm_config_reshaped_only_rhs(Query{ "g71", DataType::F16, 100, 1024, 20, 32 }).second == GEMMConfigReshapedOnlyRHS{ 4, 4, 4, 2, true, false, true }),
                       framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT((heuristics.query_gemm_config_reshaped_only_rhs(Query{ "g71", DataType::F16, 128, 101, 20, 1 }).second == GEMMConfigReshapedOnlyRHS{ 2, 2, 4, 2, true, true, true }),
                       framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT((heuristics.query_gemm_config_reshaped_only_rhs(Query{ "g71", DataType::F16, 400, 100, 512, 1 }).second == GEMMConfigReshapedOnlyRHS{ 5, 4, 4, 5, true, true, false }),
                       framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT((heuristics.query_gemm_config_reshaped_only_rhs(Query{ "g71", DataType::F16, 400, 100, 512, 1 }).second == GEMMConfigReshapedOnlyRHS{ 5, 4, 4, 5, true, true, false }),
                       framework::LogLevel::ERRORS);

    ARM_COMPUTE_EXPECT((heuristics.query_gemm_config_reshaped(Query{ "g76", DataType::F16, 100, 100, 20, 32 }).second == GEMMConfigReshaped{ 4, 2, 4, 2, 8, true, false, true, false }),
                       framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT((heuristics.query_gemm_config_reshaped(Query{ "g76", DataType::F16, 128, 512, 1024, 1 }).second == GEMMConfigReshaped{ 4, 2, 4, 2, 8, true, false, true, false }),
                       framework::LogLevel::ERRORS);
}

TEST_CASE(InvalidDotmlgoSyntaxShouldReturnInvalidStatus, framework::DatasetMode::ALL)
{
    std::string       mlgo_str = R"_(
        <header>
        gemm-version, [1,2,1]
        ip-type,pu
        </header>
        <heuristics-table>
        0, g76 , 8, f32, best-performance, static, gemm-config-reshaped, [m,n,k,n]

        </heurist
        <heuristic, 0>
        l ,0,gemm-config-reshaped,[4,2,4,2,8,1,0,1,0]
        </heuristic>
    )_";
    std::stringstream ss(mlgo_str);
    MLGOHeuristics    heuristics;
    ARM_COMPUTE_EXPECT(!heuristics.reload_from_stream(ss), framework::LogLevel::ERRORS);
}

TEST_SUITE(InvalidDotmlgoSemanticsShouldReturnInvalidStatus)
// If the semantics errors are local to some trees instead of the entire heuristics, an alternative is to simply
// ignore/remove those invalid trees. However the reason why we choose to throw, thus invalidating the entire
// heuristics is that if there are some invalid trees, the quality of the dotmlgo is called into question even if
// the rest of the trees are semantically valid, and they could severely degrade the performance of GEMM. Therefore
// this "all or nothing" approach when it comes to dotmlgo correctness is safer and more defensive.

// Also note that the semantic error of the tree only refers to those that obstruct its evaluation and thus query,
// (e.g. invalid tree structure, unsupported features etc.) instead of those affecting the desired outcome
// (usually in terms of final GEMM performance, e.g. the effectiveness of the decision tree)

// In the future we might want to check the content of the exceptions as well. But right now it suffices to only
// know that it throws exactly when it needs to.
TEST_CASE(MismatchesBetweenHeuristicsTableEntriesAndHeuristicTrees, framework::DatasetMode::ALL)
{
    {
        // Mismatching number of entries 1
        std::string       mlgo_str = R"_(
            <header>
            gemm-version, [1,2,1]
            ip-type,gpu
            </header>
            <heuristics-table>

            0, g76 , 8, f32, best-performance, static, gemm-config-reshaped, [m,n,k,n]

            </heuristics-table>
        )_";
        std::stringstream ss(mlgo_str);
        MLGOHeuristics    heuristics;
        // NOTE: This case might throw an internal error as the tree inserted by the heuristics-table cannot not be checked
        ARM_COMPUTE_EXPECT(!heuristics.reload_from_stream(ss), framework::LogLevel::ERRORS);
    }

    {
        // Mismatching number of entries 2
        std::string       mlgo_str = R"_(
            <header>
            gemm-version, [1,2,1]
            ip-type,gpu
            </header>
            <heuristics-table>
            </heuristics-table>
            <heuristic, 1>
            l ,0,gemm-config-reshaped,[4,2,4,2,8,1,0,1,0]
            </heuristic>
        )_";
        std::stringstream ss(mlgo_str);
        MLGOHeuristics    heuristics;
        ARM_COMPUTE_EXPECT(!heuristics.reload_from_stream(ss), framework::LogLevel::ERRORS);
    }

    {
        // Mismatching info
        std::string       mlgo_str = R"_(
            <header>
            gemm-version, [1,2,1]
            ip-type,gpu
            </header>
            <heuristics-table>
            0, g76 , 8, f32, best-performance, static, gemm-type, [m,n,k,n]
            </heuristics-table>
            <heuristic, 0>
            l ,0,gemm-config-reshaped,[4,2,4,2,8,1,0,1,0]
            </heuristic>
        )_";
        std::stringstream ss(mlgo_str);
        MLGOHeuristics    heuristics;
        ARM_COMPUTE_EXPECT(!heuristics.reload_from_stream(ss), framework::LogLevel::ERRORS);
    }
}

TEST_CASE(RepeatedHeuristicsTableEntriesId, framework::DatasetMode::ALL)
{
    std::string       mlgo_str = R"_(
        <header>
        gemm-version, [1,2,1]
        ip-type,gpu
        </header>
        <heuristics-table>
        0, g76 , 8, f32, best-performance, static, gemm-config-reshaped, [m,n,k,n]
        0, g71 , 8, f32, best-performance, static, gemm-config-reshaped, [m,n,k,n]
        </heuristics-table>
        <heuristic, 0>
        l ,0,gemm-config-reshaped,[4,2,4,2,8,1,0,1,0]
        </heuristic>
        <heuristic, 1>
        l ,0,gemm-config-reshaped,[4,2,4,2,8,1,0,1,0]
        </heuristic>
    )_";
    std::stringstream ss(mlgo_str);
    MLGOHeuristics    heuristics;
    ARM_COMPUTE_EXPECT(!heuristics.reload_from_stream(ss), framework::LogLevel::ERRORS);
}

TEST_CASE(RepeatedHeuristicsTableEntriesIndex, framework::DatasetMode::ALL)
{
    std::string       mlgo_str = R"_(
        <header>
        gemm-version, [1,2,1]
        ip-type,gpu
        </header>
        <heuristics-table>
        0, g76 , 8, f32, best-performance, static, gemm-config-reshaped, [m,n,k,n]
        1, g76 , 8, f32, best-performance, static, gemm-config-reshaped, [m,n,k,n]
        </heuristics-table>
        <heuristic, 0>
        l ,0,gemm-config-reshaped,[4,2,4,2,8,1,0,1,0]
        </heuristic>
        <heuristic, 1>
        l ,0,gemm-config-reshaped,[4,2,4,2,8,1,0,1,0]
        </heuristic>
    )_";
    std::stringstream ss(mlgo_str);
    MLGOHeuristics    heuristics;
    ARM_COMPUTE_EXPECT(!heuristics.reload_from_stream(ss), framework::LogLevel::ERRORS);
}

TEST_CASE(RepeatedHeuristicTreesId, framework::DatasetMode::ALL)
{
    std::string       mlgo_str = R"_(
        <header>
        gemm-version, [1,2,1]
        ip-type,gpu
        </header>
        <heuristics-table>
        0, g76 , 8, f32, best-performance, static, gemm-config-reshaped, [m,n,k,n]
        1, g71 , 8, f32, best-performance, static, gemm-config-reshaped, [m,n,k,n]
        </heuristics-table>
        <heuristic, 0>
        l ,0,gemm-config-reshaped,[4,2,4,2,8,1,0,1,0]
        </heuristic>
        <heuristic, 0>
        l ,0,gemm-config-reshaped,[4,2,4,2,8,1,0,1,0]
        </heuristic>
    )_";
    std::stringstream ss(mlgo_str);
    MLGOHeuristics    heuristics;
    ARM_COMPUTE_EXPECT(!heuristics.reload_from_stream(ss), framework::LogLevel::ERRORS);
}
TEST_CASE(EmptyTree, framework::DatasetMode::ALL)
{
    std::string       mlgo_str = R"_(
        <header>
        gemm-version, [1,2,1]
        ip-type,gpu
        </header>
        <heuristics-table>
        0, g76 , 8, f32, best-performance, static, gemm-config-reshaped, [m,n,k,n]
        </heuristics-table>
        <heuristic, 0>
        </heuristic>
    )_";
    std::stringstream ss(mlgo_str);
    MLGOHeuristics    heuristics;
    ARM_COMPUTE_EXPECT(!heuristics.reload_from_stream(ss), framework::LogLevel::ERRORS);
}

TEST_CASE(InvalidTreeMissingRoot, framework::DatasetMode::ALL)
{
    std::string       mlgo_str = R"_(
        <header>
        gemm-version, [1,2,1]
        ip-type,gpu
        </header>
        <heuristics-table>
        0, g76 , 8, f32, best-performance, static, gemm-config-reshaped, [m,n,k,n]
        </heuristics-table>
        <heuristic, 0>
        b ,2, var, m, ==, num, 10., 3, 4
        l ,3,gemm-config-reshaped,[4,2,4,2,8,1,0,1,0]
        l ,4,gemm-config-reshaped,[4,2,4,2,8,1,0,1,0]
        </heuristic>
    )_";
    std::stringstream ss(mlgo_str);
    MLGOHeuristics    heuristics;
    ARM_COMPUTE_EXPECT(!heuristics.reload_from_stream(ss), framework::LogLevel::ERRORS);
}
TEST_CASE(InvalidTreeMissingNodes, framework::DatasetMode::ALL)
{
    std::string       mlgo_str = R"_(
        <header>
        gemm-version, [1,2,1]
        ip-type,gpu
        </header>
        <heuristics-table>
        0, g76 , 8, f32, best-performance, static, gemm-config-reshaped, [m,n,k,n]
        </heuristics-table>
        <heuristic, 0>
        b ,0, var, m, ==, num, 10., 1, 2
        l ,1,gemm-config-reshaped,[4,2,4,2,8,1,0,1,0]
        </heuristic>
    )_";
    std::stringstream ss(mlgo_str);
    MLGOHeuristics    heuristics;
    ARM_COMPUTE_EXPECT(!heuristics.reload_from_stream(ss), framework::LogLevel::ERRORS);
}
TEST_CASE(InvalidTreeRepeatedNodeIds, framework::DatasetMode::ALL)
{
    std::string       mlgo_str = R"_(
        <header>
        gemm-version, [1,2,1]
        ip-type,gpu
        </header>
        <heuristics-table>
        0, g76 , 8, f32, best-performance, static, gemm-config-reshaped, [m,n,k,n]
        </heuristics-table>
        <heuristic, 0>
        b ,0, var, m, ==, num, 10., 1, 2
        l ,1,gemm-config-reshaped,[4,2,4,2,8,1,0,1,0]
        l ,1,gemm-config-reshaped,[1,2,4,2,8,1,0,1,0]
        l ,2,gemm-config-reshaped,[2,2,4,2,8,1,0,1,0]
        </heuristic>
    )_";
    std::stringstream ss(mlgo_str);
    MLGOHeuristics    heuristics;
    ARM_COMPUTE_EXPECT(!heuristics.reload_from_stream(ss), framework::LogLevel::ERRORS);
}
TEST_CASE(InvalidTreeDisjointNodes, framework::DatasetMode::ALL)
{
    std::string       mlgo_str = R"_(
        <header>
        gemm-version, [1,2,1]
        ip-type,gpu
        </header>
        <heuristics-table>
        0, g76 , 8, f32, best-performance, static, gemm-config-reshaped, [m,n,k,n]
        </heuristics-table>
        <heuristic, 0>
        b ,0, var, m, ==, num, 10., 1, 2
        l ,1,gemm-config-reshaped,[4,2,4,2,8,1,0,1,0]
        l ,2,gemm-config-reshaped,[2,2,4,2,8,1,0,1,0]

        b ,4, var, n, ==, num, 10., 5, 6
        l ,5,gemm-config-reshaped,[4,2,4,2,8,1,0,1,0]
        l ,6,gemm-config-reshaped,[2,2,4,2,8,1,0,1,0]

        l ,7,gemm-config-reshaped,[2,2,4,2,8,1,0,1,0]
        </heuristic>
    )_";
    std::stringstream ss(mlgo_str);
    MLGOHeuristics    heuristics;
    ARM_COMPUTE_EXPECT(!heuristics.reload_from_stream(ss), framework::LogLevel::ERRORS);
}
TEST_CASE(InvalidTreeLoop, framework::DatasetMode::ALL)
{
    std::string       mlgo_str = R"_(
        <header>
        gemm-version, [1,2,1]
        ip-type,gpu
        </header>
        <heuristics-table>
        0, g76 , 8, f32, best-performance, static, gemm-config-reshaped, [m,n,k,n]
        </heuristics-table>
        <heuristic, 0>
        b ,0, var, m, ==, num, 10., 0, 1
        l ,1,gemm-config-reshaped,[4,2,4,2,8,1,0,1,0]
        </heuristic>
    )_";
    std::stringstream ss(mlgo_str);
    MLGOHeuristics    heuristics;
    ARM_COMPUTE_EXPECT(!heuristics.reload_from_stream(ss), framework::LogLevel::ERRORS);
}
TEST_CASE(InvalidTreeCycle, framework::DatasetMode::ALL)
{
    std::string       mlgo_str = R"_(
        <header>
        gemm-version, [1,2,1]
        ip-type,gpu
        </header>
        <heuristics-table>
        0, g76 , 8, f32, best-performance, static, gemm-config-reshaped, [m,n,k,n]
        </heuristics-table>
        <heuristic, 0>
        b ,0, var, m, ==, num, 10., 1, 5
        b ,1, var, n, ==, num, 10., 2, 3
        l ,2,gemm-config-reshaped,[4,2,4,2,8,1,0,1,0]
        b ,3, var, k, ==, num, 10., 0, 4
        l ,4,gemm-config-reshaped,[4,2,4,2,8,1,0,1,0]
        l ,5,gemm-config-reshaped,[4,2,4,2,8,1,0,1,0]
        </heuristic>
    )_";
    std::stringstream ss(mlgo_str);
    MLGOHeuristics    heuristics;
    ARM_COMPUTE_EXPECT(!heuristics.reload_from_stream(ss), framework::LogLevel::ERRORS);
}
TEST_CASE(InvalidTreeInvalidFeatures, framework::DatasetMode::ALL)
{
    std::string       mlgo_str = R"_(
        <header>
        gemm-version, [1,2,1]
        ip-type,gpu
        </header>
        <heuristics-table>
        0, g76 , 8, f32, best-performance, static, gemm-config-reshaped, [m,n,k,n]
        </heuristics-table>
        <heuristic, 0>
        b ,0, var, magic_feature, ==, num, 10., 1, 2
        l ,1,gemm-config-reshaped,[4,2,4,2,8,1,0,1,0]
        l ,2,gemm-config-reshaped,[4,2,4,2,8,1,0,1,0]
        </heuristic>
    )_";
    std::stringstream ss(mlgo_str);
    MLGOHeuristics    heuristics;
    ARM_COMPUTE_EXPECT(!heuristics.reload_from_stream(ss), framework::LogLevel::ERRORS);
}
TEST_SUITE_END() // InvalidDotmlgoSemanticsShouldReturnInvalidStatus

TEST_CASE(InvalidUsageOfHeuristicsShouldReturnInvalidStatus, framework::DatasetMode::ALL)
{
    std::string       mlgo_str = R"_(
        <header>
        gemm-version, [1,2,1]
        ip-type,gpu
        </header>
        <heuristics-table>
        0, g76 , 8, f32, best-performance, static, gemm-type, [m,n,k,n]
        </heuristics-table>
        <heuristic, 0>
        b , 0, var, m, ==, num, 10., 1, 2
        l , 1, gemm-type, reshaped
        b , 2, var, r_mn, >=, num, 2., 3, 6
        b , 3, var, n, >=, num, 200., 4, 5
        l , 4, gemm-type, reshaped-only-rhs
        l , 5, gemm-type, reshaped
        l , 6, gemm-type, reshaped-only-rhs
        </heuristic>
    )_";
    std::stringstream ss(mlgo_str);
    MLGOHeuristics    heuristics;
    ARM_COMPUTE_EXPECT(heuristics.reload_from_stream(ss), framework::LogLevel::ERRORS);

    // Querying unavailable heuristic type should return invalid Status
    ARM_COMPUTE_EXPECT(!heuristics.query_gemm_config_reshaped(Query{ "g76", DataType::F32, 1024, 1024, 100, 3 }).first, framework::LogLevel::ERRORS);
    // Querying unavailable ip target should return invalid Status
    ARM_COMPUTE_EXPECT(!heuristics.query_gemm_type(Query{ "g77", DataType::F32, 1024, 1024, 100, 3 }).first, framework::LogLevel::ERRORS);
    // Querying unavailable data type should return invalid Status
    ARM_COMPUTE_EXPECT(!heuristics.query_gemm_config_reshaped_only_rhs(Query{ "g76", DataType::QASYMM8, 1024, 1024, 100, 3 }).first, framework::LogLevel::ERRORS);
}
TEST_SUITE_END() // MLGOHeuristics
TEST_SUITE_END() // UNIT
TEST_SUITE_END() // CL
} // namespace validation
} // namespace test
} // namespace arm_compute
