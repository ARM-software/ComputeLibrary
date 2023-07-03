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
#ifndef COMPUTE_KERNEL_WRITER_TESTS_TENSORBITMASK_HPP
#define COMPUTE_KERNEL_WRITER_TESTS_TENSORBITMASK_HPP

#include "ckw/TensorInfo.h"
#include "common/Common.h"

#include <vector>

namespace ckw
{
class TensorBitMaskTrueTest : public ITest
{
public:
    TensorBitMaskTrueTest()
    {
        _component.push_back(TensorComponent::Dim0);
        _component.push_back(TensorComponent::Dim1);
        _component.push_back(TensorComponent::Dim2);
        _component.push_back(TensorComponent::Dim3);
        _component.push_back(TensorComponent::Dim4);
        _component.push_back(TensorComponent::Stride0);
        _component.push_back(TensorComponent::Stride1);
        _component.push_back(TensorComponent::Stride2);
        _component.push_back(TensorComponent::Stride3);
        _component.push_back(TensorComponent::Stride4);
        _component.push_back(TensorComponent::Dim1xDim2);
        _component.push_back(TensorComponent::Dim1xDim2xDim3);
        _component.push_back(TensorComponent::Dim2xDim3);
        _component.push_back(TensorComponent::OffsetFirstElement);

        _bitmask.push_back(TensorComponentBitmask::Dimension);
        _bitmask.push_back(TensorComponentBitmask::Dimension);
        _bitmask.push_back(TensorComponentBitmask::Dimension);
        _bitmask.push_back(TensorComponentBitmask::Dimension);
        _bitmask.push_back(TensorComponentBitmask::Dimension);
        _bitmask.push_back(TensorComponentBitmask::Stride);
        _bitmask.push_back(TensorComponentBitmask::Stride);
        _bitmask.push_back(TensorComponentBitmask::Stride);
        _bitmask.push_back(TensorComponentBitmask::Stride);
        _bitmask.push_back(TensorComponentBitmask::Stride);
        _bitmask.push_back(TensorComponentBitmask::FoldedDimensions);
        _bitmask.push_back(TensorComponentBitmask::FoldedDimensions);
        _bitmask.push_back(TensorComponentBitmask::FoldedDimensions);
        _bitmask.push_back(TensorComponentBitmask::OffsetFirstElement);
    }

    bool run() override
    {
        // The status of this variable can change in VALIDATE_TEST()
        bool all_tests_passed = true;

        VALIDATE_ON_MSG(_component.size() == _bitmask.size(),
                        "The number of layouts and components does not match");
        const size_t num_tests = _component.size();
        for(size_t i = 0; i < num_tests; ++i)
        {
            const TensorComponent        component = _component[i];
            const TensorComponentBitmask bitmask   = _bitmask[i];
            const bool                   out       = static_cast<uint32_t>(component) & static_cast<uint32_t>(bitmask);
            VALIDATE_TEST(out == true, all_tests_passed, i);
        }
        return all_tests_passed;
    }

    std::string name() override
    {
        return "TensorBitMaskTrueTest";
    }

private:
    std::vector<TensorComponent>        _component{};
    std::vector<TensorComponentBitmask> _bitmask{};
};

class TensorBitMaskFalseTest : public ITest
{
public:
    TensorBitMaskFalseTest()
    {
        _component.push_back(TensorComponent::Dim0);
        _component.push_back(TensorComponent::Dim1);
        _component.push_back(TensorComponent::Dim2);
        _component.push_back(TensorComponent::Dim3);
        _component.push_back(TensorComponent::Dim4);
        _component.push_back(TensorComponent::Dim0);
        _component.push_back(TensorComponent::Dim1);
        _component.push_back(TensorComponent::Dim2);
        _component.push_back(TensorComponent::Dim3);
        _component.push_back(TensorComponent::Dim4);
        _component.push_back(TensorComponent::Dim0);
        _component.push_back(TensorComponent::Dim1);
        _component.push_back(TensorComponent::Dim2);
        _component.push_back(TensorComponent::Dim3);
        _component.push_back(TensorComponent::Dim4);
        _component.push_back(TensorComponent::Stride0);
        _component.push_back(TensorComponent::Stride1);
        _component.push_back(TensorComponent::Stride2);
        _component.push_back(TensorComponent::Stride3);
        _component.push_back(TensorComponent::Stride4);
        _component.push_back(TensorComponent::Stride0);
        _component.push_back(TensorComponent::Stride1);
        _component.push_back(TensorComponent::Stride2);
        _component.push_back(TensorComponent::Stride3);
        _component.push_back(TensorComponent::Stride4);
        _component.push_back(TensorComponent::Stride0);
        _component.push_back(TensorComponent::Stride1);
        _component.push_back(TensorComponent::Stride2);
        _component.push_back(TensorComponent::Stride3);
        _component.push_back(TensorComponent::Stride4);
        _component.push_back(TensorComponent::Dim1xDim2);
        _component.push_back(TensorComponent::Dim1xDim2xDim3);
        _component.push_back(TensorComponent::Dim2xDim3);
        _component.push_back(TensorComponent::Dim1xDim2);
        _component.push_back(TensorComponent::Dim1xDim2xDim3);
        _component.push_back(TensorComponent::Dim2xDim3);
        _component.push_back(TensorComponent::Dim1xDim2);
        _component.push_back(TensorComponent::Dim1xDim2xDim3);
        _component.push_back(TensorComponent::Dim2xDim3);
        _component.push_back(TensorComponent::OffsetFirstElement);
        _component.push_back(TensorComponent::OffsetFirstElement);
        _component.push_back(TensorComponent::OffsetFirstElement);

        _bitmask.push_back(TensorComponentBitmask::Stride);
        _bitmask.push_back(TensorComponentBitmask::Stride);
        _bitmask.push_back(TensorComponentBitmask::Stride);
        _bitmask.push_back(TensorComponentBitmask::Stride);
        _bitmask.push_back(TensorComponentBitmask::Stride);
        _bitmask.push_back(TensorComponentBitmask::FoldedDimensions);
        _bitmask.push_back(TensorComponentBitmask::FoldedDimensions);
        _bitmask.push_back(TensorComponentBitmask::FoldedDimensions);
        _bitmask.push_back(TensorComponentBitmask::FoldedDimensions);
        _bitmask.push_back(TensorComponentBitmask::FoldedDimensions);
        _bitmask.push_back(TensorComponentBitmask::OffsetFirstElement);
        _bitmask.push_back(TensorComponentBitmask::OffsetFirstElement);
        _bitmask.push_back(TensorComponentBitmask::OffsetFirstElement);
        _bitmask.push_back(TensorComponentBitmask::OffsetFirstElement);
        _bitmask.push_back(TensorComponentBitmask::OffsetFirstElement);
        _bitmask.push_back(TensorComponentBitmask::Dimension);
        _bitmask.push_back(TensorComponentBitmask::Dimension);
        _bitmask.push_back(TensorComponentBitmask::Dimension);
        _bitmask.push_back(TensorComponentBitmask::Dimension);
        _bitmask.push_back(TensorComponentBitmask::Dimension);
        _bitmask.push_back(TensorComponentBitmask::FoldedDimensions);
        _bitmask.push_back(TensorComponentBitmask::FoldedDimensions);
        _bitmask.push_back(TensorComponentBitmask::FoldedDimensions);
        _bitmask.push_back(TensorComponentBitmask::FoldedDimensions);
        _bitmask.push_back(TensorComponentBitmask::FoldedDimensions);
        _bitmask.push_back(TensorComponentBitmask::OffsetFirstElement);
        _bitmask.push_back(TensorComponentBitmask::OffsetFirstElement);
        _bitmask.push_back(TensorComponentBitmask::OffsetFirstElement);
        _bitmask.push_back(TensorComponentBitmask::OffsetFirstElement);
        _bitmask.push_back(TensorComponentBitmask::OffsetFirstElement);
        _bitmask.push_back(TensorComponentBitmask::Dimension);
        _bitmask.push_back(TensorComponentBitmask::Dimension);
        _bitmask.push_back(TensorComponentBitmask::Dimension);
        _bitmask.push_back(TensorComponentBitmask::Stride);
        _bitmask.push_back(TensorComponentBitmask::Stride);
        _bitmask.push_back(TensorComponentBitmask::Stride);
        _bitmask.push_back(TensorComponentBitmask::OffsetFirstElement);
        _bitmask.push_back(TensorComponentBitmask::OffsetFirstElement);
        _bitmask.push_back(TensorComponentBitmask::OffsetFirstElement);
        _bitmask.push_back(TensorComponentBitmask::Dimension);
        _bitmask.push_back(TensorComponentBitmask::Stride);
        _bitmask.push_back(TensorComponentBitmask::FoldedDimensions);
    }

    bool run() override
    {
        // The status of this variable can change in VALIDATE_TEST()
        bool all_tests_passed = true;

        VALIDATE_ON_MSG(_component.size() == _bitmask.size(),
                        "The number of layouts and components does not match");
        const size_t num_tests = _component.size();
        for(size_t i = 0; i < num_tests; ++i)
        {
            const TensorComponent        component = _component[i];
            const TensorComponentBitmask bitmask   = _bitmask[i];
            const bool                   out       = static_cast<uint32_t>(component) & static_cast<uint32_t>(bitmask);
            VALIDATE_TEST(out == false, all_tests_passed, i);
        }
        return all_tests_passed;
    }

    std::string name() override
    {
        return "TensorBitMaskFalseTest";
    }

private:
    std::vector<TensorComponent>        _component{};
    std::vector<TensorComponentBitmask> _bitmask{};
};
} // namespace ckw

#endif /* COMPUTE_KERNEL_WRITER_TESTS_TENSORBITMASK_HPP */
