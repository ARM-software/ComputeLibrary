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
#ifndef CKW_TESTS_UTILSTEST_H
#define CKW_TESTS_UTILSTEST_H

#include "ckw/TensorInfo.h"
#include "ckw/types/TensorDataLayout.h"
#include "common/Common.h"
#include "src/TensorUtils.h"

#include <vector>

namespace ckw
{
class UtilsTest : public ITest
{
public:
    UtilsTest()
    {
        _layout.push_back(TensorDataLayout::Nhwc);
        _layout.push_back(TensorDataLayout::Nhwc);
        _layout.push_back(TensorDataLayout::Nhwc);
        _layout.push_back(TensorDataLayout::Nhwc);
        _layout.push_back(TensorDataLayout::Ndhwc);
        _layout.push_back(TensorDataLayout::Ndhwc);
        _layout.push_back(TensorDataLayout::Ndhwc);
        _layout.push_back(TensorDataLayout::Ndhwc);
        _layout.push_back(TensorDataLayout::Ndhwc);

        _component.push_back(TensorDataLayoutComponent::N);
        _component.push_back(TensorDataLayoutComponent::H);
        _component.push_back(TensorDataLayoutComponent::W);
        _component.push_back(TensorDataLayoutComponent::C);
        _component.push_back(TensorDataLayoutComponent::N);
        _component.push_back(TensorDataLayoutComponent::D);
        _component.push_back(TensorDataLayoutComponent::H);
        _component.push_back(TensorDataLayoutComponent::W);
        _component.push_back(TensorDataLayoutComponent::C);

        _expected.push_back(TensorComponentType::Dim3);
        _expected.push_back(TensorComponentType::Dim2);
        _expected.push_back(TensorComponentType::Dim1);
        _expected.push_back(TensorComponentType::Dim0);
        _expected.push_back(TensorComponentType::Dim4);
        _expected.push_back(TensorComponentType::Dim3);
        _expected.push_back(TensorComponentType::Dim2);
        _expected.push_back(TensorComponentType::Dim1);
        _expected.push_back(TensorComponentType::Dim0);
    }

    bool run() override
    {
        // The status of this variable can change in VALIDATE_TEST()
        bool all_tests_passed = true;

        VALIDATE_ON_MSG(_layout.size() == _component.size(), "The number of layouts and components does not match");
        VALIDATE_ON_MSG(_layout.size() == _expected.size(),
                        "The number of layouts and expected outputs does not match");
        const size_t num_tests = _layout.size();
        for(size_t i = 0; i < num_tests; ++i)
        {
            const TensorDataLayout          layout    = _layout[i];
            const TensorDataLayoutComponent component = _component[i];
            const TensorComponentType       expected  = _expected[i];
            const TensorComponentType       out       = get_tensor_dimension(layout, component);
            VALIDATE_TEST(out == expected, all_tests_passed, i);
        }
        return all_tests_passed;
    }

    std::string name() override
    {
        return "UtilsTest";
    }

private:
    std::vector<TensorDataLayout>          _layout{};
    std::vector<TensorDataLayoutComponent> _component{};
    std::vector<TensorComponentType>       _expected{};
};
} // namespace ckw

#endif // CKW_TESTS_UTILSTEST_H
