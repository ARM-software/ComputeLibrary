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
#ifdef ENABLE_EXPERIMENTAL_DYNAMIC_FUSION
#include "arm_compute/core/experimental/DependencyGraph.h"

#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"

using namespace arm_compute::experimental::dynamic_fusion;

namespace arm_compute
{
namespace test
{
namespace validation
{
TEST_SUITE(CL)

TEST_SUITE(UNIT)
TEST_SUITE(DYNAMIC_FUSION)
TEST_SUITE(DependencyGraph)

TEST_CASE(Correct_Graph_Creation_Should_Pass, framework::DatasetMode::ALL)
{
    DependencyGraph graph{};
    const auto      t0 = graph.add_tensor();
    const auto      t1 = graph.add_tensor();
    const auto      t2 = graph.add_tensor();
    const auto      t3 = graph.add_tensor();
    const auto      t4 = graph.add_tensor();

    const auto o0 = graph.add_operator({ t0, t1 }, { t2 }).second;
    const auto o1 = graph.add_operator({ t3, t2 }, { t4 }).second;

    ARM_COMPUTE_EXPECT_EQUAL(graph.number_of_ops(), 2U, framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT_EQUAL(graph.number_of_tensors(), 5U, framework::LogLevel::ERRORS);

    const DependencyGraph ref_graph
    {
        {
            // src_tensors
            { o0, { t0, t1 } },
            { o1, { t3, t2 } },
        },
        {
            // dst_tensors
            { o0, { t2 } },
            { o1, { t4 } },
        },
        {
            // src_ops
            { t0, {} },
            { t1, {} },
            { t2, { o0 } },
            { t3, {} },
            { t4, { o1 } },
        },
        {
            // dst_ops
            { t0, { o0 } },
            { t1, { o0 } },
            { t2, { o1 } },
            { t3, { o1 } },
            { t4, {} },
        }

    };
    ARM_COMPUTE_EXPECT(graph == ref_graph, framework::LogLevel::ERRORS);
}

TEST_CASE(Correct_Merge_Points_Should_Enable_Graph_Expansion, framework::DatasetMode::ALL)
{
    // Merge points are a simple way to collapse "graph of graphs" into a single graph
    // Suppose we have a top-level graph g0
    DependencyGraph g0{};
    const auto      g0_t0 = g0.add_tensor();
    const auto      g0_t1 = g0.add_tensor();
    const auto      g0_t2 = g0.add_tensor();
    const auto      g0_t3 = g0.add_tensor();
    const auto      g0_t4 = g0.add_tensor();
    g0.add_operator({ g0_t0, g0_t1 }, { g0_t2 }); // g0_o0
    g0.add_operator({ g0_t3, g0_t2 }, { g0_t4 }); // g0_o1

    // Then g0 expands into g1, with additional nodes added in-between "merge point tensors"
    // Note that the expansion logic may be local to each operator node
    DependencyGraph g1{};
    // g0_o0 expands into g1_o0, g1_o1, g1_o2
    const auto g1_t0 = g1.add_tensor(g0_t0);
    const auto g1_t1 = g1.add_tensor(g0_t1);
    const auto g1_t2 = g1.add_tensor();
    const auto g1_t3 = g1.add_tensor();
    const auto g1_t4 = g1.add_tensor(g0_t2);
    const auto g1_o0 = g1.add_operator({ g1_t0 }, { g1_t2 }).second;
    const auto g1_o1 = g1.add_operator({ g1_t1 }, { g1_t3 }).second;
    const auto g1_o2 = g1.add_operator({ g1_t2, g1_t3 }, { g1_t4 }).second;

    // g0_o1 expands into g1_o3
    const auto g1_t5 = g1.add_tensor(g0_t3);
    const auto g1_t6 = g1.add_tensor(g0_t2);
    const auto g1_t7 = g1.add_tensor(g0_t4);
    ARM_COMPUTE_EXPECT_EQUAL(g1_t4, g1_t6, framework::LogLevel::ERRORS); // both associate with the same merge point g0_t2, thus they should point to the same tensor in g1
    const auto g1_o3 = g1.add_operator({ g1_t5, g1_t6 }, { g1_t7 }).second;

    const DependencyGraph ref_graph
    {
        {
            // src_tensors
            { g1_o0, { g1_t0 } },
            { g1_o1, { g1_t1 } },
            { g1_o2, { g1_t2, g1_t3 } },
            { g1_o3, { g1_t5, g1_t4 } },
        },
        {
            // dst_tensors
            { g1_o0, { g1_t2 } },
            { g1_o1, { g1_t3 } },
            { g1_o2, { g1_t4 } },
            { g1_o3, { g1_t7 } },
        },
        {
            // src_ops
            { g1_t0, {} },
            { g1_t1, {} },
            { g1_t2, { g1_o0 } },
            { g1_t3, { g1_o1 } },
            { g1_t4, { g1_o2 } },
            { g1_t5, {} },
            { g1_t7, { g1_o3 } },
        },
        {
            // dst_ops
            { g1_t0, { g1_o0 } },
            { g1_t1, { g1_o1 } },
            { g1_t2, { g1_o2 } },
            { g1_t3, { g1_o2 } },
            { g1_t4, { g1_o3 } },
            { g1_t5, { g1_o3 } },
            { g1_t7, {} },
        },
        {
            // merge points
            { g0_t0, g1_t0 },
            { g0_t1, g1_t1 },
            { g0_t2, g1_t4 },
            { g0_t3, g1_t5 },
            { g0_t4, g1_t7 },
        }
    };
    ARM_COMPUTE_EXPECT(g1 == ref_graph, framework::LogLevel::ERRORS);
}

TEST_CASE(Path_Existence_Check_0, framework::DatasetMode::ALL)
{
    DependencyGraph graph{};
    const auto      t0 = graph.add_tensor();
    const auto      t1 = graph.add_tensor();
    const auto      t2 = graph.add_tensor();
    const auto      t3 = graph.add_tensor();
    const auto      t4 = graph.add_tensor();
    const auto      t5 = graph.add_tensor();
    const auto      t6 = graph.add_tensor();
    const auto      t7 = graph.add_tensor();
    const auto      o0 = graph.add_operator({ t1 }, { t3, t4 }).second;
    const auto      o1 = graph.add_operator({ t3 }, { t5 }).second;
    const auto      o2 = graph.add_operator({ t5, t6 }, { t7 }).second;
    const auto      o3 = graph.add_operator({ t4 }, { t6 }).second;
    const auto      o4 = graph.add_operator({ t0, t5 }, { t2 }).second;

    ARM_COMPUTE_UNUSED(o1, o3);

    ARM_COMPUTE_EXPECT((graph.path_exists_from_tensor_to_op(t3, o2)), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT((graph.path_exists_from_tensor_to_op(t1, o4)), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(!(graph.path_exists_from_tensor_to_op(t2, o4)), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(!(graph.path_exists_from_tensor_to_op(t0, o2)), framework::LogLevel::ERRORS);

    ARM_COMPUTE_EXPECT((graph.path_exists_from_op_to_op(o0, o2)), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(!(graph.path_exists_from_op_to_op(o2, o0)), framework::LogLevel::ERRORS);

    ARM_COMPUTE_EXPECT(!(graph.path_exists_from_op_to_op(o2, o4)), framework::LogLevel::ERRORS);
}

TEST_CASE(Correct_Topological_Sort_Should_Pass, framework::DatasetMode::ALL)
{
    DependencyGraph graph{};
    const auto      t0 = graph.add_tensor();
    const auto      t1 = graph.add_tensor();
    const auto      t2 = graph.add_tensor();
    const auto      t3 = graph.add_tensor();
    const auto      t4 = graph.add_tensor();
    const auto      t5 = graph.add_tensor();
    const auto      t6 = graph.add_tensor();
    const auto      t7 = graph.add_tensor();
    const auto      o0 = graph.add_operator({ t1 }, { t3, t4 }).second;
    const auto      o1 = graph.add_operator({ t3 }, { t5 }).second;
    const auto      o2 = graph.add_operator({ t5, t6 }, { t7 }).second;
    const auto      o3 = graph.add_operator({ t4 }, { t6 }).second;
    const auto      o4 = graph.add_operator({ t0, t5 }, { t2 }).second;

    const auto res = graph.topological_sort();
    ARM_COMPUTE_EXPECT(bool(res.first), framework::LogLevel::ERRORS);
    std::vector<DependencyGraph::OpPack> ref_sorted_op_packs
    {
        { o0, { t1 }, { t3, t4 } },
        { o1, { t3 }, { t5 } },
        { o3, { t4 }, { t6 } },
        { o4, { t0, t5 }, { t2 } },
        { o2, { t5, t6 }, { t7 } },

    };
    ARM_COMPUTE_EXPECT((res.second == ref_sorted_op_packs), framework::LogLevel::ERRORS);
}

TEST_CASE(Cycles_Should_Fail, framework::DatasetMode::ALL)
{
    DependencyGraph graph{};
    const auto      t0 = graph.add_tensor();
    const auto      t1 = graph.add_tensor();
    const auto      t2 = graph.add_tensor();
    const auto      t3 = graph.add_tensor();

    graph.add_operator({ t0, t1 }, { t2 });
    graph.add_operator({ t2 }, { t1, t3 }); // Ideally error should occur here

    const auto res = graph.topological_sort();
    ARM_COMPUTE_EXPECT(!bool(res.first), framework::LogLevel::ERRORS);
}
TEST_CASE(Loops_Should_Fail, framework::DatasetMode::ALL)
{
    DependencyGraph graph{};
    const auto      t0 = graph.add_tensor();
    const auto      t1 = graph.add_tensor();
    const auto      t2 = graph.add_tensor();

    ARM_COMPUTE_EXPECT_THROW(graph.add_operator({ t0, t2 }, { t1, t2 }).first, framework::LogLevel::ERRORS);
    ARM_COMPUTE_UNUSED(t0, t1, t2);
}
TEST_SUITE_END() // DependencyGraph
TEST_SUITE_END() // DYNAMIC_FUSION
TEST_SUITE_END() // UNIT

TEST_SUITE_END() // CL
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ENABLE_EXPERIMENTAL_DYNAMIC_FUSION */