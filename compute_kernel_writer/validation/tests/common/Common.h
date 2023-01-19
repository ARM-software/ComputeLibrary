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
#ifndef COMPUTE_KERNEL_WRITER_TEST_COMMON_COMMON_H
#define COMPUTE_KERNEL_WRITER_TEST_COMMON_COMMON_H

#include <cassert>
#include <iostream>
#include <string>

namespace ckw
{
#define VALIDATE_ON_MSG(exp, msg) assert(((void)msg, exp))

#define VALIDATE_TEST(exp, all_tests_passed, id_test) \
    do                                      \
    {                                       \
        if((exp) == true)   \
        { \
            all_tests_passed &= true; \
            const std::string msg = "TEST " + std::to_string((id_test)) + ": [PASSED]"; \
            std::cout << msg << std::endl; \
        } \
        else \
        { \
            all_tests_passed &= false; \
            const std::string msg = "TEST " + std::to_string((id_test)) + ": [FAILED]"; \
            std::cout << msg << std::endl; \
        } \
    } while(false)

class ITest
{
public:
    virtual ~ITest() = default;
    /** Method to run the test
     *
     * @return it returns true if all tests passed
     */
    virtual bool run() = 0;
    /** Name of the test
     *
     * @return it returns the name of the test
     */
    virtual std::string name() = 0;
};
} // namespace ckw

#endif /* COMPUTE_KERNEL_WRITER_TEST_COMMON_COMMON_H */
