/*
 * Copyright (c) 2025 Arm Limited.
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
#if defined(ARM_COMPUTE_OPENMP_SCHEDULER)

#include "arm_compute/runtime/OMP/OMPScheduler.h"

#include "arm_compute/core/CPP/ICPPKernel.h"

#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"

#include <algorithm>
#include <array>
#include <omp.h>

using namespace arm_compute;
using namespace arm_compute::test;

namespace
{
template <int num_threads>
class TestKernel : public ICPPKernel
{
public:
    TestKernel()
    {
        Window window;
        window.set(0, Window::Dimension(0, num_threads));
        configure(window);
    }

    const char *name() const override
    {
        return "TestKernel";
    }

    void run(const Window &window, const ThreadInfo &info) override
    {
        ARM_COMPUTE_UNUSED(window);
#pragma omp critical
        {
            roll_call[info.thread_id] = true;
        }
    }

    bool success()
    {
        return std::all_of(roll_call.begin(), roll_call.end(), [](bool b) { return b; });
        ;
    }

private:
    std::array<bool, num_threads> roll_call{};
};
} // namespace

TEST_SUITE(UNIT)
TEST_SUITE(OMPScheduler)
TEST_CASE(NestedParallelRegions, framework::DatasetMode::ALL)
{
    // We set nested to false, because the user might disable it, and
    // the scheduler should not depend on it. This test is introduced
    // because OMPScheduler was depending on omp_get_thread_num() and
    // this call does not return the correct thread if there are nested
    // parallel regions.
    omp_set_max_active_levels(1);

    constexpr int num_parallel_regions = 2;
    constexpr int kernel_parallelism   = 2;

#pragma omp parallel num_threads(num_parallel_regions)
    {
        OMPScheduler                   scheduler;
        OMPScheduler::Hints            hints(0);
        TestKernel<kernel_parallelism> kernel;

        scheduler.set_num_threads(kernel_parallelism);
        scheduler.schedule(&kernel, hints);

        ARM_COMPUTE_EXPECT(kernel.success(), framework::LogLevel::ERRORS);
    }
}
TEST_SUITE_END() // OMPScheduler
TEST_SUITE_END() // UNIT

#endif // defined(ARM_COMPUTE_OPENMP_SCHEDULER)
