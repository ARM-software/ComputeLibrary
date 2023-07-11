/*
 * Copyright (c) 2023 Arm Limited.
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

#include "tests/CLKernelWriterCommentTest.h"
#include "tests/CLKernelWriterDeclareTileTest.h"
#include "tests/CLConstantTileTest.hpp"
#include "tests/CLTileTest.hpp"
#include "tests/TensorBitMaskTest.hpp"
#include "tests/UtilsTest.hpp"

#include <memory>
#include <vector>

using namespace ckw;

/** Main test program
 */
int32_t main()
{
    std::vector<ITest *> tests;

    // Add your test here
    const auto test0 = std::make_unique<UtilsTest>();
    const auto test1 = std::make_unique<TensorBitMaskTrueTest>();
    const auto test2 = std::make_unique<TensorBitMaskFalseTest>();
    tests.push_back(test0.get());
    tests.push_back(test1.get());
    tests.push_back(test2.get());

#ifdef COMPUTE_KERNEL_WRITER_OPENCL_ENABLED
    const auto test3  = std::make_unique<CLTileInternalVariableNamesTest>();
    const auto test4  = std::make_unique<CLTileInternalNumVariablesTest>();
    const auto test5  = std::make_unique<CLTileAccessScalarVariableTest>();
    const auto test6  = std::make_unique<CLTileAccessScalarVariableBroadcastXTest>();
    const auto test7  = std::make_unique<CLTileAccessScalarVariableBroadcastYTest>();
    const auto test8  = std::make_unique<CLTileAccessVectorVariablesTest>();
    const auto test9  = std::make_unique<CLTileAccessSubVectorVariablesTest>();
    const auto test10 = std::make_unique<CLConstantTileInternalValuesTest>();
    const auto test11 = std::make_unique<CLConstantTileAccessScalarVariableBroadcastXTest>();
    const auto test12 = std::make_unique<CLConstantTileAccessScalarVariableBroadcastYTest>();
    const auto test13 = std::make_unique<CLConstantTileAccessVectorVariablesTest>();
    const auto test14 = std::make_unique<CLConstantTileAccessSubVectorVariablesTest>();
#ifdef COMPUTE_KERNEL_WRITER_DEBUG_ENABLED
    const auto test15 = std::make_unique<CLKernelWriterCommentTest>();
#endif /* COMPUTE_KERNEL_WRITER_DEBUG_ENABLED */
    const auto test16 = std::make_unique<CLKernelWriterDeclareTileTest>();

    tests.push_back(test3.get());
    tests.push_back(test4.get());
    tests.push_back(test5.get());
    tests.push_back(test6.get());
    tests.push_back(test7.get());
    tests.push_back(test8.get());
    tests.push_back(test9.get());
    tests.push_back(test10.get());
    tests.push_back(test11.get());
    tests.push_back(test12.get());
    tests.push_back(test13.get());
    tests.push_back(test14.get());
#ifdef COMPUTE_KERNEL_WRITER_DEBUG_ENABLED
    tests.push_back(test15.get());
#endif /* COMPUTE_KERNEL_WRITER_DEBUG_ENABLED */
    tests.push_back(test16.get());
#endif /* COMPUTE_KERNEL_WRITER_OPENCL_ENABLED */

    bool all_test_passed = true;

    for(auto &x : tests)
    {
        std::cout << x->name() << std::endl;
        all_test_passed &= x->run();
    }

    if(all_test_passed == true)
    {
        std::cout << "All tests passed" << std::endl;
    }
    else
    {
        throw std::runtime_error("One or more tests failed");
    }

    return 0;
}
