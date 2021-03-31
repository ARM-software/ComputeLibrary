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
#ifndef ARM_COMPUTE_TEST_UNIT_TENSOR
#define ARM_COMPUTE_TEST_UNIT_TENSOR

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
/** Test case for AclCreateTensor
 *
 * Validate that AclCreateTensor behaves as expected with invalid context
 *
 * Test Steps:
 *  - Call AclCreateTensor with an invalid context
 *  - Confirm that AclInvalidArgument is reported
 *  - Confirm that the tensor is still nullptr
 */
class CreateTensorWithInvalidContextFixture : public framework::Fixture
{
public:
    void setup()
    {
        AclTensor tensor = nullptr;
        ARM_COMPUTE_ASSERT(AclCreateTensor(&tensor, nullptr, nullptr, false) == AclStatus::AclInvalidArgument);
        ARM_COMPUTE_ASSERT(tensor == nullptr);
    };
};

/** Test-case for AclCreateTensor
 *
 * Validate that AclCreateTensor behaves as expected on invalid descriptor
 *
 * Test Steps:
 *  - Call AclCreateTensor with valid context but invalid descriptor
 *  - Confirm that AclInvalidArgument is reported
 *  - Confirm that tensor is still nullptr
 */
template <acl::Target Target>
class CreateTensorWithInvalidDescriptorFixture : public framework::Fixture
{
public:
    void setup()
    {
        acl::Context ctx(Target);
        AclTensor    tensor = nullptr;
        ARM_COMPUTE_ASSERT(AclCreateTensor(&tensor, ctx.get(), nullptr, false) == AclStatus::AclInvalidArgument);
        ARM_COMPUTE_ASSERT(tensor == nullptr);

        // Check invalid data type
        AclTensorDescriptor invalid_desc;
        invalid_desc.ndims     = 4;
        invalid_desc.data_type = static_cast<AclDataType>(-1);
        ARM_COMPUTE_ASSERT(AclCreateTensor(&tensor, ctx.get(), &invalid_desc, false) == AclStatus::AclInvalidArgument);
        ARM_COMPUTE_ASSERT(tensor == nullptr);

        // Check invalid number of dimensions
        invalid_desc.data_type = AclDataType::AclFloat32;
        invalid_desc.ndims     = 15;
        ARM_COMPUTE_ASSERT(AclCreateTensor(&tensor, ctx.get(), &invalid_desc, false) == AclStatus::AclInvalidArgument);
        ARM_COMPUTE_ASSERT(tensor == nullptr);
    };
};

/** Test case for AclDestroyTensor
*
* Validate that AclDestroyTensor behaves as expected when an invalid tensor is given
*
* Test Steps:
*  - Call AclDestroyTensor with null tensor
*  - Confirm that AclInvalidArgument is reported
*  - Call AclDestroyTensor on empty array
*  - Confirm that AclInvalidArgument is reported
*  - Call AclDestroyTensor on an ACL object other than AclTensor
*  - Confirm that AclInvalidArgument is reported
*  - Confirm that tensor is still nullptr
*/
template <acl::Target Target>
class DestroyInvalidTensorFixture : public framework::Fixture
{
public:
    void setup()
    {
        acl::Context ctx(Target);

        std::array<char, 256> empty_array{};
        AclTensor tensor = nullptr;

        ARM_COMPUTE_ASSERT(AclDestroyTensor(tensor) == AclStatus::AclInvalidArgument);
        ARM_COMPUTE_ASSERT(AclDestroyTensor(reinterpret_cast<AclTensor>(ctx.get())) == AclStatus::AclInvalidArgument);
        ARM_COMPUTE_ASSERT(AclDestroyTensor(reinterpret_cast<AclTensor>(empty_array.data())) == AclStatus::AclInvalidArgument);
        ARM_COMPUTE_ASSERT(tensor == nullptr);
    };
};

/** Test case for AclCreateTensor
 *
 * Validate that a tensor can be created successfully
 *
 * Test Steps:
 *  - Create a valid context
 *  - Create a valid tensor
 *  - Confirm that AclSuccess is returned
 */
template <acl::Target Target>
class SimpleTensorFixture : public framework::Fixture
{
public:
    void setup()
    {
        acl::StatusCode err = acl::StatusCode::Success;
        acl::Context    ctx(Target, &err);

        ARM_COMPUTE_ASSERT(err == acl::StatusCode::Success);
        acl::Tensor tensor(ctx, acl::TensorDescriptor({ 2, 3 }, acl::DataType::Float32), &err);
        ARM_COMPUTE_ASSERT(err == acl::StatusCode::Success);
    };
};

/** Test case for AclTensor
 *
 * Validate that multiple tensors can be created successfully
 * Possibly stress the possibility of memory leaks
 *
 * Test Steps:
 *  - Create a valid context
 *  - Create a lot of tensors
 *  - Confirm that AclSuccess is returned
 */
template <acl::Target Target>
class TensorStressFixture : public framework::Fixture
{
public:
    void setup()
    {
        acl::StatusCode err = acl::StatusCode::Success;

        acl::Context ctx(Target, &err);
        ARM_COMPUTE_ASSERT(err == acl::StatusCode::Success);

        const unsigned int num_tensors = 1024;
        for(unsigned int i = 0; i < num_tensors; ++i)
        {
            acl::Tensor tensor(ctx, acl::TensorDescriptor({ 1024, 1024 }, acl::DataType::Float32), &err);
            ARM_COMPUTE_ASSERT(err == acl::StatusCode::Success);
        }
    };
};

/** Test case for AclMapTensor
 *
 * Validate that map on an invalid object fails
 *
 * Test Steps:
 *  - Create a valid context
 *  - Pass and invalid object for mapping
 *  - Confirm that AclInvalidArgument is returned
 */
template <acl::Target Target>
class MapInvalidTensorFixture : public framework::Fixture
{
public:
    void setup()
    {
        acl::StatusCode err = acl::StatusCode::Success;

        acl::Context ctx(Target, &err);
        ARM_COMPUTE_ASSERT(err == acl::StatusCode::Success);

        void *handle = nullptr;
        ARM_COMPUTE_ASSERT(AclMapTensor(reinterpret_cast<AclTensor>(ctx.get()), &handle) == AclStatus::AclInvalidArgument);
    };
};

/** Test case for AclMapTensor
 *
 * Validate that map of an unallocated pointer is nullptr
 *
 * Test Steps:
 *  - Create a valid context
 *  - Create a valid tensor without allocating
 *  - Map tensor
 *  - Check that mapping is nullptr
 */
template <acl::Target Target>
class MapNotAllocatedTensorFixture : public framework::Fixture
{
public:
    void setup()
    {
        acl::StatusCode err = acl::StatusCode::Success;

        acl::Context ctx(Target, &err);
        ARM_COMPUTE_ASSERT(err == acl::StatusCode::Success);

        acl::Tensor tensor(ctx, acl::TensorDescriptor({ 8, 8 }, acl::DataType::Float32), false /* allocate */, &err);
        ARM_COMPUTE_ASSERT(err == acl::StatusCode::Success);
        ARM_COMPUTE_ASSERT(tensor.map() == nullptr);
    };
};

/** Test case for AclMapTensor
 *
 * Validate that map of a valid tensor return a non-nullptr value
 *
 * Test Steps:
 *  - Create a valid context
 *  - Create a valid tensor while allocating
 *  - Map tensor
 *  - Check that mapping is not nullptr
 */
template <acl::Target Target>
class MapAllocatedTensorFixture : public framework::Fixture
{
public:
    void setup()
    {
        acl::StatusCode err = acl::StatusCode::Success;

        acl::Context ctx(Target, &err);
        ARM_COMPUTE_ASSERT(err == acl::StatusCode::Success);

        acl::Tensor tensor(ctx, acl::TensorDescriptor({ 8, 8 }, acl::DataType::Float32), &err);
        ARM_COMPUTE_ASSERT(err == acl::StatusCode::Success);

        void *handle = tensor.map();
        ARM_COMPUTE_ASSERT(handle != nullptr);
        ARM_COMPUTE_ASSERT(tensor.unmap(handle) == acl::StatusCode::Success);
    };
};

/** Test case for AclTensorImport
 *
 * Validate that an externally memory can be successfully imported
 *
 * Test Steps:
 *  - Create a valid context
 *  - Create a valid tensor without allocating
 *  - Allocate external memory
 *  - Import memory to the tensor
 *  - Check that imported pointer matches
 */
template <acl::Target Target>
class ImportMemoryFixture : public framework::Fixture
{
public:
    void setup()
    {
        acl::StatusCode err = acl::StatusCode::Success;

        acl::Context ctx(Target, &err);
        ARM_COMPUTE_ASSERT(err == acl::StatusCode::Success);

        const int32_t size = 8;
        acl::Tensor   tensor(ctx, acl::TensorDescriptor({ size }, acl::DataType::Float32), false /* allocate */, &err);
        ARM_COMPUTE_ASSERT(err == acl::StatusCode::Success);

        std::vector<float> data(size);
        err = tensor.import(data.data(), acl::ImportType::Host);

        void *handle = tensor.map();
        ARM_COMPUTE_ASSERT(handle == data.data());
        ARM_COMPUTE_ASSERT(tensor.unmap(handle) == acl::StatusCode::Success);
    }
};
/** Test case for get_size() interface of Tensor
 *
 *
 * Test Steps:
 *  - Create a valid context
 *  - Create a valid tensor
 *  - Compare the size value returned with the expected value
 */
template <acl::Target Target>
class TensorSizeFixture : public framework::Fixture
{
public:
    void setup()
    {
        acl::StatusCode err = acl::StatusCode::Success;
        acl::Context    ctx(Target, &err);

        ARM_COMPUTE_ASSERT(err == acl::StatusCode::Success);
        acl::Tensor tensor(ctx, acl::TensorDescriptor({ 2, 3 }, acl::DataType::Float32), &err);

        // size should be 6 elements (2x3) times 4 bytes (float32) = 24 bytes
        constexpr size_t expected_size = 24;
        ARM_COMPUTE_ASSERT(tensor.get_size() == expected_size);
    };
};
/** Test case for get_size() dealing with invalid arguments
 *
 * Test Steps:
 *  - Test nullptr tensor can return a correct error
 *  - Create a valid tensor
 *  - Test C interface with null size argument can return a correct error
 */
template <acl::Target Target>
class InvalidTensorSizeFixture : public framework::Fixture
{
public:
    void setup()
    {
        // Null tensor
        AclTensor null_tensor = nullptr;
        uint64_t  size{ 0 };
        ARM_COMPUTE_ASSERT(AclGetTensorSize(null_tensor, &size) == AclStatus::AclInvalidArgument);

        // Create valid tensor
        acl::StatusCode err = acl::StatusCode::Success;
        acl::Context    ctx(Target, &err);
        ARM_COMPUTE_ASSERT(err == acl::StatusCode::Success);
        acl::Tensor tensor(ctx, acl::TensorDescriptor({ 2, 3 }, acl::DataType::Float32), &err);

        // Null size argument
        ARM_COMPUTE_ASSERT(AclGetTensorSize(tensor.get(), nullptr) == AclStatus::AclInvalidArgument);
    };
};

template <acl::Target Target>
class DescriptorConversionFixture : public framework::Fixture
{
    bool compare_descriptor(const AclTensorDescriptor &desc_a, const AclTensorDescriptor &desc_b)
    {
        auto are_descriptors_same = true;

        are_descriptors_same &= desc_a.ndims == desc_b.ndims;
        are_descriptors_same &= desc_a.data_type == desc_b.data_type;
        are_descriptors_same &= desc_a.shape != nullptr && desc_b.shape != nullptr;

        for(int32_t d = 0; d < desc_a.ndims; ++d)
        {
            are_descriptors_same &= desc_a.shape[d] == desc_b.shape[d];
        }

        // other attributes should be added here

        return are_descriptors_same;
    }

public:
    void setup()
    {
        auto err{ acl::StatusCode::Success };
        auto ctx{ acl::Context(Target, &err) };
        ARM_COMPUTE_ASSERT(err == acl::StatusCode::Success);

        auto        desc{ acl::TensorDescriptor({ 2, 3 }, acl::DataType::Float32) };
        acl::Tensor tensor(ctx, desc, &err);

        auto desc_from_tensor = tensor.get_descriptor();

        ARM_COMPUTE_ASSERT(compare_descriptor(*desc.get(), *desc_from_tensor.get()));
        ARM_COMPUTE_ASSERT(desc == desc_from_tensor);

        // Test c interface with "prepopulated" descriptor
        // Note: When c interface used, there are possibility of memory leak
        // if members are not correctly deleted (e.g., shape).
        // Since that is considered user's responsibility, we don't test here.
        AclTensorDescriptor prepopulated_descriptor
        {
            3, nullptr, AclDataType::AclBFloat16, nullptr, 0
        };

        ARM_COMPUTE_ASSERT(AclGetTensorDescriptor(tensor.get(), &prepopulated_descriptor) == AclStatus::AclSuccess);
        ARM_COMPUTE_ASSERT(compare_descriptor(*desc.get(), prepopulated_descriptor));
        ARM_COMPUTE_ASSERT(desc == acl::TensorDescriptor(prepopulated_descriptor));
    };
};

template <acl::Target Target>
class InvalidDescriptorConversionFixture : public framework::Fixture
{
public:
    void setup()
    {
        // Null tensor
        AclTensor           null_tensor = nullptr;
        AclTensorDescriptor desc{};
        ARM_COMPUTE_ASSERT(AclGetTensorDescriptor(null_tensor, &desc) == AclStatus::AclInvalidArgument);

        // Create valid tensor
        acl::StatusCode err = acl::StatusCode::Success;
        acl::Context    ctx(Target, &err);
        ARM_COMPUTE_ASSERT(err == acl::StatusCode::Success);
        acl::Tensor tensor(ctx, acl::TensorDescriptor({ 2, 3 }, acl::DataType::Float32), &err);

        // Null size argument
        ARM_COMPUTE_ASSERT(AclGetTensorDescriptor(tensor.get(), nullptr) == AclStatus::AclInvalidArgument);
    };
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_UNIT_TENSOR */
