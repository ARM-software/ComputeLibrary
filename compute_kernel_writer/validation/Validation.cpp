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

#include "validation/tests/CLConstantTileTest.hpp"
#include "validation/tests/CLKernelWriterAssignTest.h"
#include "validation/tests/CLKernelWriterBinaryOpTest.h"
#include "validation/tests/CLKernelWriterCastTest.h"
#include "validation/tests/CLKernelWriterCommentTest.h"
#include "validation/tests/CLKernelWriterDeclareConstantTileTest.h"
#include "validation/tests/CLKernelWriterDeclareTensorTest.h"
#include "validation/tests/CLKernelWriterDeclareTileTest.h"
#include "validation/tests/CLKernelWriterForTest.h"
#include "validation/tests/CLKernelWriterIfTest.h"
#include "validation/tests/CLKernelWriterOpLoadStoreTest.h"
#include "validation/tests/CLKernelWriterReturnTest.h"
#include "validation/tests/CLKernelWriterTernaryOpTest.h"
#include "validation/tests/CLKernelWriterUnaryExpressionTest.h"
#include "validation/tests/CLTensorArgumentTest.h"
#include "validation/tests/CLTileTest.hpp"
#include "validation/tests/TensorBitMaskTest.h"
#include "validation/tests/UtilsTest.h"

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
    const auto test17 = std::make_unique<CLTensorArgumentComponentNamesTest>();
    const auto test18 = std::make_unique<CLTensorArgumentStorageNamesTest>();
    const auto test19 = std::make_unique<CLTensorArgumentComponentValuesTest>();
    const auto test20 = std::make_unique<CLTensorArgumentComponentsUsedPassByValueFalseTest>();
    const auto test21 = std::make_unique<CLTensorArgumentComponentsUsedPassByValueTrueTest>();
    const auto test22 = std::make_unique<CLTensorArgumentStoragesUsedTest>();
    const auto test23 = std::make_unique<CLTensorArgumentComponentsUsedPassByValueTrueDynamicDimTrueTest>();
    const auto test24 = std::make_unique<CLKernelWriterDeclareTensorTest>();
    const auto test25 = std::make_unique<CLKernelWriterOpLoadStoreTest>();
    const auto test26 = std::make_unique<CLKernelWriterAssignTest>();
    const auto test27 = std::make_unique<CLKernelWriterCastTest>();
    const auto test28 = std::make_unique<CLKernelWriterUnaryExpressionTest>();
    const auto test29 = std::make_unique<CLKernelWriterBinaryOpTest>();
    const auto test30 = std::make_unique<CLKernelWriterTernaryOpTest>();
    const auto test31 = std::make_unique<CLKernelWriterDeclareConstantTileTest>();
    const auto test32 = std::make_unique<CLKernelWriterIfTest>();
    const auto test33 = std::make_unique<CLKernelWriterForTest>();
    const auto test34 = std::make_unique<CLKernelWriterReturnTest>();

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
    tests.push_back(test17.get());
    tests.push_back(test18.get());
    tests.push_back(test19.get());
    tests.push_back(test20.get());
    tests.push_back(test21.get());
    tests.push_back(test22.get());
    tests.push_back(test23.get());
    tests.push_back(test24.get());
    CKW_UNUSED(test25); // CLKernelWriterOpLoadStoreTest test needs further changes.
    tests.push_back(test26.get());
    tests.push_back(test27.get());
    tests.push_back(test28.get());
    tests.push_back(test29.get());
    tests.push_back(test30.get());
    tests.push_back(test31.get());
    tests.push_back(test32.get());
    tests.push_back(test33.get());
    tests.push_back(test34.get());
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
