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
#include "arm_compute/core/TensorFormat.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/COOTensor.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/Helpers.h"

#include "tests/NEON/Accessor.h"
#include "tests/NEON/Helper.h"

#include <cmath>
#include <cstring>
#include <sstream>
#include <vector>

namespace arm_compute
{
namespace
{
bool are_values_equal(const uint8_t *a, const uint8_t *b, DataType dt, size_t element_size)
{
    if(dt == DataType::F32)
    {
        float va = *reinterpret_cast<const float *>(a);
        float vb = *reinterpret_cast<const float *>(b);
        if(std::fabs(va - vb) > 0e-5f)
        {
            return false;
        }
    } else
    {
        if(std::memcmp(a, b, element_size) != 0)
        {
            return false;
        }
    }

    return true;
}

bool tensors_are_equal(const test::Accessor &a, const test::Accessor &b)
{
    if(a.shape() != b.shape() || a.data_type() != b.data_type())
        return false;

    const size_t element_size = a.element_size();
    Window window;
    window.use_tensor_dimensions(a.shape());

    bool equal = true;

    execute_window_loop(window, [&](const Coordinates &id)
    {
        const uint8_t *a_value = static_cast<const uint8_t *>(a(id));
        const uint8_t *b_value = static_cast<const uint8_t *>(b(id));

        equal &= are_values_equal(a_value, b_value, a.data_type(), element_size);
    });

    return equal;
}

/** Fills every element of @p t with the raw byte @p value.
 *  Works for any element type; for multi-byte types every byte is set to @p value.
 */
void fill_tensor_uniform(Tensor &t, uint8_t value)
{
    std::memset(t.buffer() + t.info()->offset_first_element_in_bytes(),
                value,
                t.info()->tensor_shape().total_size() * t.info()->element_size());
}
} // namespace

namespace test
{
namespace validation
{
TEST_SUITE(UNIT)
TEST_SUITE(SparseTensor)

// clang-format off
/** Validates TensorInfo Autopadding */
DATA_TEST_CASE(ConvertCOOTensorToDense, framework::DatasetMode::ALL, combine(
                framework::dataset::make("TensorShape", {
                TensorShape(8U),
                TensorShape(3U, 3U),
                TensorShape(2U, 5U, 5U),
                TensorShape(4U, 2U, 2U, 9U)}),
                framework::dataset::make("TensorType", {
                DataType::U8,
                DataType::S8,
                DataType::U32,
                DataType::S32,
                DataType::F16,
                DataType::F32})
            ), shape, type)
{
    const auto t_info = TensorInfo(shape, 1, type, DataLayout::NCHW);
    auto t            = create_tensor<Tensor>(t_info);
    auto t_zero       = create_tensor<Tensor>(t_info);

    t.allocator()->allocate();
    library->fill_tensor_sparse_random(Accessor(t), 0.2);

    t_zero.allocator()->allocate();
    library->fill_static_values(Accessor(t_zero), std::vector<unsigned>(shape.total_size(), 0));

    for(size_t sparse_dim = 1; sparse_dim <= shape.num_dimensions(); sparse_dim++)
    {
        auto          st = t.to_coo_sparse(sparse_dim);
        bool   is_sparse = st->info()->is_sparse();
        bool      is_coo = st->info()->tensor_format() == TensorFormat::COO;
        size_t dense_dim = shape.num_dimensions() - sparse_dim;
        size_t is_hybrid = dense_dim > 0;
        auto          td = st->to_dense();

        ARM_COMPUTE_EXPECT(is_sparse, framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(is_coo, framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(st->sparse_dim() == sparse_dim, framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(st->dense_dim() == dense_dim, framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(st->is_hybrid() == is_hybrid, framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(tensors_are_equal(Accessor(t), Accessor(*td)), framework::LogLevel::ERRORS);

        auto st_zero = t_zero.to_coo_sparse(sparse_dim);
        auto td_zero = st_zero->to_dense();
        ARM_COMPUTE_EXPECT(tensors_are_equal(Accessor(t_zero), Accessor(*td_zero)), framework::LogLevel::ERRORS);
    }
}
// clang-format on
// *INDENT-ON*

// clang-format off
/** Validates TensorInfo Autopadding */
DATA_TEST_CASE(ConvertCSRTensorToDense, framework::DatasetMode::ALL, combine(
                framework::dataset::make("TensorShape", {
                TensorShape(8U),
                TensorShape(3U, 3U),
                TensorShape(2U, 5U, 5U),
                TensorShape(4U, 2U, 2U, 9U)}),
                framework::dataset::make("TensorType", {
                DataType::U8,
                DataType::S8,
                DataType::U32,
                DataType::S32,
                DataType::F16,
                DataType::F32})
            ), shape, type)
{
    // Currently, CSRTensor only supports 2D tensors
    if(shape.num_dimensions() < 2)
    {
        return;
    }
    const TensorShape tensor_shape(shape[0], shape[1]);

    const auto t_info = TensorInfo(tensor_shape, 1, type, DataLayout::NCHW);
    auto t            = create_tensor<Tensor>(t_info);
    auto t_zero       = create_tensor<Tensor>(t_info);

    t.allocator()->allocate();
    library->fill_tensor_sparse_random(Accessor(t), 0.2);

    t_zero.allocator()->allocate();
    library->fill_static_values(Accessor(t_zero), std::vector<unsigned>(tensor_shape.total_size(), 0));

    auto st           = t.to_csr_sparse();
    auto td           = st->to_dense();
    bool   is_sparse  = st->info()->is_sparse();
    bool      is_csr  = st->info()->tensor_format() == TensorFormat::CSR;
    size_t sparse_dim = tensor_shape.num_dimensions();
    size_t dense_dim  = tensor_shape.num_dimensions() - sparse_dim;
    size_t is_hybrid  = dense_dim > 0;

    ARM_COMPUTE_EXPECT(is_sparse, framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(is_csr, framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(st->sparse_dim() == sparse_dim, framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(st->dense_dim() == dense_dim, framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(st->is_hybrid() == is_hybrid, framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(tensors_are_equal(Accessor(t), Accessor(*td)), framework::LogLevel::ERRORS);
    
    auto st_zero = t_zero.to_csr_sparse();
    auto td_zero = st_zero->to_dense();
    ARM_COMPUTE_EXPECT(tensors_are_equal(Accessor(t_zero), Accessor(*td_zero)), framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

TEST_SUITE(COO)

TEST_CASE(NNZAllZero, framework::DatasetMode::ALL)
{
    const TensorShape shape(4U, 4U);
    auto t = create_tensor<Tensor>(TensorInfo(shape, 1, DataType::U8, DataLayout::NCHW));
    t.allocator()->allocate();
    fill_tensor_uniform(t, 0U);

    auto st = t.to_coo_sparse(shape.num_dimensions());
    ARM_COMPUTE_EXPECT(st->nnz() == 0U, framework::LogLevel::ERRORS);
}

TEST_CASE(NNZAllNonZero, framework::DatasetMode::ALL)
{
    const TensorShape shape(4U, 4U);
    auto t = create_tensor<Tensor>(TensorInfo(shape, 1, DataType::U8, DataLayout::NCHW));
    t.allocator()->allocate();
    fill_tensor_uniform(t, 1U);

    auto st = t.to_coo_sparse(shape.num_dimensions());
    ARM_COMPUTE_EXPECT(st->nnz() == shape.total_size(), framework::LogLevel::ERRORS);
}

/** get_coordinates(i) must return a coordinate at which get_value is non-null. */
TEST_CASE(GetCoordinatesConsistency, framework::DatasetMode::ALL)
{
    const TensorShape shape(4U, 4U);
    auto t = create_tensor<Tensor>(TensorInfo(shape, 1, DataType::U8, DataLayout::NCHW));
    t.allocator()->allocate();
    fill_tensor_uniform(t, 1U);  // all non-zero

    auto st = t.to_coo_sparse(shape.num_dimensions());
    for(size_t i = 0; i < st->nnz(); ++i)
    {
        const Coordinates c  = st->get_coordinates(i);
        const uint8_t    *vp = st->get_value(c);
        ARM_COMPUTE_EXPECT(vp != nullptr, framework::LogLevel::ERRORS);
    }
}

/** get_value at a known non-zero coordinate returns the correct value. */
TEST_CASE(GetValueKnownNonZero, framework::DatasetMode::ALL)
{
    const TensorShape shape(4U, 4U);
    auto t = create_tensor<Tensor>(TensorInfo(shape, 1, DataType::U8, DataLayout::NCHW));
    t.allocator()->allocate();
    fill_tensor_uniform(t, 7U);

    auto st = t.to_coo_sparse(shape.num_dimensions());
    // (0,0) is guaranteed non-zero; value must equal 7
    const uint8_t *v = st->get_value(Coordinates{ 0, 0 });
    ARM_COMPUTE_EXPECT(v != nullptr, framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(*v == 7U, framework::LogLevel::ERRORS);
}

/** get_value returns nullptr for every coordinate on an all-zero tensor. */
TEST_CASE(GetValueAllZero, framework::DatasetMode::ALL)
{
    const TensorShape shape(4U, 4U);
    auto t = create_tensor<Tensor>(TensorInfo(shape, 1, DataType::U8, DataLayout::NCHW));
    t.allocator()->allocate();
    fill_tensor_uniform(t, 0U);

    auto st = t.to_coo_sparse(shape.num_dimensions());
    ARM_COMPUTE_EXPECT(st->get_value(Coordinates{ 0, 0 }) == nullptr, framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(st->get_value(Coordinates{ 3, 3 }) == nullptr, framework::LogLevel::ERRORS);
}

/** density() + sparsity() must equal 1 and both must be in [0, 1]. */
TEST_CASE(DensitySparsityComplement, framework::DatasetMode::ALL)
{
    const TensorShape shape(4U, 4U);
    auto t = create_tensor<Tensor>(TensorInfo(shape, 1, DataType::U8, DataLayout::NCHW));
    t.allocator()->allocate();
    library->fill_tensor_sparse_random(Accessor(t), 0.5);

    auto st = t.to_coo_sparse(shape.num_dimensions());
    ARM_COMPUTE_EXPECT(std::fabs(st->density() + st->sparsity() - 1.0f) < 1e-5f, framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(st->density() >= 0.0f && st->density() <= 1.0f, framework::LogLevel::ERRORS);
}

/** All-zero tensor: density == 0, sparsity == 1. */
TEST_CASE(DensityBoundaryAllZero, framework::DatasetMode::ALL)
{
    const TensorShape shape(4U, 4U);
    auto t = create_tensor<Tensor>(TensorInfo(shape, 1, DataType::U8, DataLayout::NCHW));
    t.allocator()->allocate();
    fill_tensor_uniform(t, 0U);

    auto st = t.to_coo_sparse(shape.num_dimensions());
    ARM_COMPUTE_EXPECT(st->density()  == 0.0f, framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(st->sparsity() == 1.0f, framework::LogLevel::ERRORS);
}

/** All-nonzero tensor: density == 1, sparsity == 0. */
TEST_CASE(DensityBoundaryAllNonZero, framework::DatasetMode::ALL)
{
    const TensorShape shape(4U, 4U);
    auto t = create_tensor<Tensor>(TensorInfo(shape, 1, DataType::U8, DataLayout::NCHW));
    t.allocator()->allocate();
    fill_tensor_uniform(t, 1U);

    auto st = t.to_coo_sparse(shape.num_dimensions());
    ARM_COMPUTE_EXPECT(std::fabs(st->density()  - 1.0f) < 1e-5f, framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(st->sparsity() == 0.0f, framework::LogLevel::ERRORS);
}

#ifdef ARM_COMPUTE_ASSERTS_ENABLED
/** print() on empty COOTensor must produce the empty-marker string. */
TEST_CASE(PrintEmpty, framework::DatasetMode::ALL)
{
    const TensorShape shape(3U, 3U);
    auto t = create_tensor<Tensor>(TensorInfo(shape, 1, DataType::F32, DataLayout::NCHW));
    t.allocator()->allocate();
    fill_tensor_uniform(t, 0U);

    auto st = t.to_coo_sparse(shape.num_dimensions());
    std::ostringstream oss;
    st->print(oss);
    ARM_COMPUTE_EXPECT(oss.str().find("[]") != std::string::npos, framework::LogLevel::ERRORS);
}

/** print() on a non-empty COOTensor must not crash and produce non-empty output. */
TEST_CASE(PrintNonEmpty, framework::DatasetMode::ALL)
{
    const TensorShape shape(3U, 3U);
    auto t = create_tensor<Tensor>(TensorInfo(shape, 1, DataType::F32, DataLayout::NCHW));
    t.allocator()->allocate();
    fill_tensor_uniform(t, 1U);

    auto st = t.to_coo_sparse(shape.num_dimensions());
    std::ostringstream oss;
    st->print(oss);
    ARM_COMPUTE_EXPECT(!oss.str().empty(), framework::LogLevel::ERRORS);
}
#endif // ARM_COMPUTE_ASSERTS_ENABLED

TEST_SUITE_END() // COO

TEST_SUITE(CSR)

TEST_CASE(NNZAllZero, framework::DatasetMode::ALL)
{
    const TensorShape shape(4U, 4U);
    auto t = create_tensor<Tensor>(TensorInfo(shape, 1, DataType::U8, DataLayout::NCHW));
    t.allocator()->allocate();
    fill_tensor_uniform(t, 0U);

    auto st = t.to_csr_sparse();
    ARM_COMPUTE_EXPECT(st->nnz() == 0U, framework::LogLevel::ERRORS);
}

TEST_CASE(NNZAllNonZero, framework::DatasetMode::ALL)
{
    const TensorShape shape(4U, 4U);
    auto t = create_tensor<Tensor>(TensorInfo(shape, 1, DataType::U8, DataLayout::NCHW));
    t.allocator()->allocate();
    fill_tensor_uniform(t, 1U);

    auto st = t.to_csr_sparse();
    ARM_COMPUTE_EXPECT(st->nnz() == shape.total_size(), framework::LogLevel::ERRORS);
}

/** get_coordinates(i) must return a coordinate at which get_value is non-null. */
TEST_CASE(GetCoordinatesConsistency, framework::DatasetMode::ALL)
{
    const TensorShape shape(4U, 4U);
    auto t = create_tensor<Tensor>(TensorInfo(shape, 1, DataType::U8, DataLayout::NCHW));
    t.allocator()->allocate();
    fill_tensor_uniform(t, 1U);

    auto st = t.to_csr_sparse();
    for(size_t i = 0; i < st->nnz(); ++i)
    {
        const Coordinates c  = st->get_coordinates(i);
        const uint8_t    *vp = st->get_value(c);
        ARM_COMPUTE_EXPECT(vp != nullptr, framework::LogLevel::ERRORS);
    }
}

/** get_value at a known non-zero coordinate returns the correct value. */
TEST_CASE(GetValueKnownNonZero, framework::DatasetMode::ALL)
{
    const TensorShape shape(4U, 4U);
    auto t = create_tensor<Tensor>(TensorInfo(shape, 1, DataType::U8, DataLayout::NCHW));
    t.allocator()->allocate();
    fill_tensor_uniform(t, 5U);

    auto st = t.to_csr_sparse();
    const uint8_t *v = st->get_value(Coordinates{ 0, 0 });
    ARM_COMPUTE_EXPECT(v != nullptr, framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(*v == 5U, framework::LogLevel::ERRORS);
}

/** get_value returns nullptr for every coordinate on an all-zero tensor. */
TEST_CASE(GetValueAllZero, framework::DatasetMode::ALL)
{
    const TensorShape shape(4U, 4U);
    auto t = create_tensor<Tensor>(TensorInfo(shape, 1, DataType::U8, DataLayout::NCHW));
    t.allocator()->allocate();
    fill_tensor_uniform(t, 0U);

    auto st = t.to_csr_sparse();
    ARM_COMPUTE_EXPECT(st->get_value(Coordinates{ 0, 0 }) == nullptr, framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(st->get_value(Coordinates{ 3, 3 }) == nullptr, framework::LogLevel::ERRORS);
}

/** density() + sparsity() must equal 1 and both must be in [0, 1]. */
TEST_CASE(DensitySparsityComplement, framework::DatasetMode::ALL)
{
    const TensorShape shape(4U, 4U);
    auto t = create_tensor<Tensor>(TensorInfo(shape, 1, DataType::U8, DataLayout::NCHW));
    t.allocator()->allocate();
    library->fill_tensor_sparse_random(Accessor(t), 0.5);

    auto st = t.to_csr_sparse();
    ARM_COMPUTE_EXPECT(std::fabs(st->density() + st->sparsity() - 1.0f) < 1e-5f, framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(st->density() >= 0.0f && st->density() <= 1.0f, framework::LogLevel::ERRORS);
}

/** All-zero tensor: density == 0, sparsity == 1. */
TEST_CASE(DensityBoundaryAllZero, framework::DatasetMode::ALL)
{
    const TensorShape shape(4U, 4U);
    auto t = create_tensor<Tensor>(TensorInfo(shape, 1, DataType::U8, DataLayout::NCHW));
    t.allocator()->allocate();
    fill_tensor_uniform(t, 0U);

    auto st = t.to_csr_sparse();
    ARM_COMPUTE_EXPECT(st->density()  == 0.0f, framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(st->sparsity() == 1.0f, framework::LogLevel::ERRORS);
}

/** All-nonzero tensor: density == 1, sparsity == 0. */
TEST_CASE(DensityBoundaryAllNonZero, framework::DatasetMode::ALL)
{
    const TensorShape shape(4U, 4U);
    auto t = create_tensor<Tensor>(TensorInfo(shape, 1, DataType::U8, DataLayout::NCHW));
    t.allocator()->allocate();
    fill_tensor_uniform(t, 1U);

    auto st = t.to_csr_sparse();
    ARM_COMPUTE_EXPECT(std::fabs(st->density()  - 1.0f) < 1e-5f, framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(st->sparsity() == 0.0f, framework::LogLevel::ERRORS);
}

#ifdef ARM_COMPUTE_ASSERTS_ENABLED
/** print() must not crash and must produce non-empty output. */
TEST_CASE(Print, framework::DatasetMode::ALL)
{
    const TensorShape shape(4U, 4U);
    auto t = create_tensor<Tensor>(TensorInfo(shape, 1, DataType::F32, DataLayout::NCHW));
    t.allocator()->allocate();
    fill_tensor_uniform(t, 1U);

    auto st = t.to_csr_sparse();
    std::ostringstream oss;
    st->print(oss);
    ARM_COMPUTE_EXPECT(!oss.str().empty(), framework::LogLevel::ERRORS);
}
#endif // ARM_COMPUTE_ASSERTS_ENABLED

TEST_SUITE_END() // CSR

TEST_SUITE_END() // SparseTensor
TEST_SUITE_END() // UNIT

} // namespace validation
} // namespace test
} // namespace arm_compute
