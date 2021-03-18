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
#ifndef ARM_COMPUTE_TEST_UNIT_TENSORPACK_FIXTURE
#define ARM_COMPUTE_TEST_UNIT_TENSORPACK_FIXTURE

#include "arm_compute/Acl.hpp"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/framework/Macros.h"
#include "tests/validation/Validation.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
/** Test case for AclCreateTensorPack
 *
 * Validate that AclCreateTensorPack behaves as expected with invalid context
 *
 * Test Steps:
 *  - Call AclCreateTensorPack with an invalid context
 *  - Confirm that AclInvalidArgument is reported
 *  - Confirm that the tensor pack is still nullptr
 */
class CreateTensorPackWithInvalidContextFixture : public framework::Fixture
{
public:
    void setup()
    {
        AclTensorPack pack = nullptr;
        ARM_COMPUTE_ASSERT(AclCreateTensorPack(&pack, nullptr) == AclStatus::AclInvalidArgument);
        ARM_COMPUTE_ASSERT(pack == nullptr);
    };
};

/** Test case for AclDestroyTensorPack
 *
 * Validate that AclDestroyTensorPack behaves as expected when an invalid tensor pack is given
 *
 * Test Steps:
 *  - Call AclDestroyTensorPack with null tensor pack
 *  - Confirm that AclInvalidArgument is reported
 *  - Call AclDestroyTensorPack on empty array
 *  - Confirm that AclInvalidArgument is reported
 *  - Call AclDestroyTensorPack on an ACL object other than AclTensorPack
 *  - Confirm that AclInvalidArgument is reported
 *  - Confirm that tensor pack is still nullptr
 */
template <acl::Target Target>
class DestroyInvalidTensorPackFixture : public framework::Fixture
{
public:
    void setup()
    {
        acl::Context ctx(Target);

        std::array<char, 256> empty_array{};
        AclTensorPack pack = nullptr;

        ARM_COMPUTE_ASSERT(AclDestroyTensorPack(pack) == AclStatus::AclInvalidArgument);
        ARM_COMPUTE_ASSERT(AclDestroyTensorPack(reinterpret_cast<AclTensorPack>(ctx.get())) == AclStatus::AclInvalidArgument);
        ARM_COMPUTE_ASSERT(AclDestroyTensorPack(reinterpret_cast<AclTensorPack>(empty_array.data())) == AclStatus::AclInvalidArgument);
        ARM_COMPUTE_ASSERT(pack == nullptr);
    };
};

/** Test case for AclPackTensor
 *
 * Validate that AclPackTensor behaves as expected when an invalid is being passed for packing
 *
 * Test Steps:
 *  - Create a valid TensorPack
 *  - Try to pack an empty object
 *  - Confirm that AclInvalidArgument is reported
 *  - Try to pack another API object other than tensor
 *  - Confirm that AclInvalidArgument is reported
 */
template <acl::Target Target>
class AddInvalidObjectToTensorPackFixture : public framework::Fixture
{
public:
    void setup()
    {
        auto err = acl::StatusCode::Success;

        acl::Context ctx(Target, &err);
        ARM_COMPUTE_ASSERT(err == acl::StatusCode::Success);

        acl::TensorPack pack(ctx, &err);
        ARM_COMPUTE_ASSERT(err == acl::StatusCode::Success);

        auto status = AclPackTensor(pack.get(),
                                    reinterpret_cast<AclTensor>(ctx.get()),
                                    AclTensorSlot::AclSrc);
        ARM_COMPUTE_ASSERT(status == AclInvalidArgument);

        status = AclPackTensor(pack.get(), nullptr, AclTensorSlot::AclSrc);
        ARM_COMPUTE_ASSERT(status == AclInvalidArgument);
    };
};

/** Test case for AclPackTensor
 *
 * Validate that a tensor can be added successfully to the TensorPack
 *
 * Test Steps:
 *  - Create a valid tensor pack
 *  - Create a valid tensor
 *  - Add tensor to the tensor pack
 *  - Confirm that AclSuccess is returned
 */
template <acl::Target Target>
class SimpleTensorPackFixture : public framework::Fixture
{
public:
    void setup()
    {
        acl::Context    ctx(Target);
        acl::TensorPack pack(ctx);
        acl::Tensor     t(ctx, acl::TensorDescriptor({ 3, 3, 5, 7 }, acl::DataType::Float32));

        ARM_COMPUTE_ASSERT(pack.add(t, AclTensorSlot::AclSrc) == acl::StatusCode::Success);
    };
};

/** Test case for AclPackTensor
 *
 * Validate that multiple tensor can be added successfully to the TensorPack
 *
 * Test Steps:
 *  - Create a valid tensor pack
 *  - Create a list of valid tensors
 *  - Add tensors to the tensor pack
 *  - Confirm that AclSuccess is returned
 */
template <acl::Target Target>
class MultipleTensorsInPackFixture : public framework::Fixture
{
public:
    void setup()
    {
        acl::Context    ctx(Target);
        acl::TensorPack pack(ctx);

        const acl::TensorDescriptor desc(acl::TensorDescriptor({ 3, 3, 5, 7 }, acl::DataType::Float32));
        const size_t                num_tensors = 256;

        std::vector<acl::Tensor> tensors;
        for(unsigned int i = 0; i < num_tensors; ++i)
        {
            auto err = acl::StatusCode::Success;
            tensors.emplace_back(acl::Tensor(ctx, desc, &err));
            ARM_COMPUTE_ASSERT(err == acl::StatusCode::Success);
            ARM_COMPUTE_ASSERT(pack.add(tensors.back(), static_cast<int32_t>(AclTensorSlot::AclSrcVec) + i) == acl::StatusCode::Success);
        }
    };
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_UNIT_TENSORPACK_FIXTURE */
