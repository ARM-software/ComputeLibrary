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

#ifndef CKW_VALIDATION_TESTS_COMMON_KERNELWRITERINTERCEPTOR_H
#define CKW_VALIDATION_TESTS_COMMON_KERNELWRITERINTERCEPTOR_H

#include <string>
#include <utility>

namespace ckw
{

/** This class provides the ability to capture only the code changed after a point
 * and compares that part of code with the expected value.
 *
 * It is useful for testing purpose when a particular sequence of instructions is interested
 * while the rest of the initialization code is out of scope.
 */
template <typename T>
class KernelWriterInterceptor : public T
{
public:
    template <typename... TArgs>
    KernelWriterInterceptor(const TArgs &&...args)
        : T(std::forward<TArgs>(args)...)
    {
    }

    /** Mark this point in the source code as the start position to capture.
     * Only source code added after this function is considered when check_add_code is called.
     */
    void start_capture_code()
    {
        _start_code = this->body_source_code();
    }

    /** Compare the source code added after start_capture_code is called the the specified expected code. */
    bool check_added_code(const std::string &expected_added_code)
    {
        const auto &end_code = this->body_source_code();

        // Code can only grow over time.
        if(end_code.length() < _start_code.length())
        {
            return false;
        }

        // New code must be added to the source code without changing the already existed code.
        if(end_code.substr(0, _start_code.length()) != _start_code)
        {
            return false;
        }

        // The newly added code must match the expected value.
        if(end_code.substr(_start_code.length(), end_code.length() - _start_code.length()) != expected_added_code)
        {
            return false;
        }

        return true;
    }

private:
    std::string _start_code{};
};

} // namespace ckw

#endif // CKW_VALIDATION_TESTS_COMMON_KERNELWRITERINTERCEPTOR_H
