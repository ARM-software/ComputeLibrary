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
#ifndef ARM_COMPUTE_TEST_TESTRESULT
#define ARM_COMPUTE_TEST_TESTRESULT

#include "Profiler.h"

namespace arm_compute
{
namespace test
{
namespace framework
{
/** Class to store results of a test.
 *
 * Currently the execution status and profiling information are stored.
 */
struct TestResult
{
    /** Execution status of a test. */
    enum class Status
    {
        NOT_RUN,
        SUCCESS,
        EXPECTED_FAILURE,
        FAILED,
        CRASHED,
        DISABLED
    };

    /** Default constructor. */
    TestResult() = default;

    /** Initialise the result with a status.
     *
     * @param[in] status Execution status.
     */
    TestResult(Status status)
        : status{ status }
    {
    }

    /** Initialise the result with a status and profiling information.
     *
     * @param[in] status       Execution status.
     * @param[in] measurements Profiling information.
     */
    TestResult(Status status, const Profiler::MeasurementsMap &measurements)
        : status{ status }, measurements{ measurements }
    {
    }

    Status                    status{ Status::NOT_RUN }; //< Execution status
    Profiler::MeasurementsMap measurements{};            //< Profiling information
};
} // namespace framework
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_TESTRESULT */
