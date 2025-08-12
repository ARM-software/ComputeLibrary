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

#include <vector>

namespace arm_compute
{
namespace
{
std::vector<DataLayout> layouts = {
    DataLayout::NCHW,
    DataLayout::NHWC
};
std::vector<DataType> types = {
    DataType::U8,
    DataType::S8,
    DataType::U32,
    DataType::S32,
    DataType::F16,
    DataType::F32
};
std::vector<TensorShape> shapes = {
    TensorShape(8U),
    TensorShape(8U),
    TensorShape(3U, 3U),
    TensorShape(3U, 3U),
    TensorShape(2U, 5U, 5U),
    TensorShape(2U, 5U, 5U),
    TensorShape(4U, 2U, 2U, 9U),
    TensorShape(4U, 2U, 2U, 9U)
};
// For any shape, we have check the zero tensor, to be sure that `to_sparse`
// correctly returns an empty vector for each dimension.
std::vector<std::vector<unsigned>> sparse_data = {
    { 0, 0, 0, 0, 0, 0, 0, 0 },
    { 4, 0, 0, 0, 9, 0, 0, 0 },
    {
        0, 0, 0,
        0, 0, 0,
        0, 0, 0
    },
    {
        2, 0, 3,
        0, 1, 0,
        0, 0, 0
    },
    {
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0
    },
    {
        0, 0, 0,   0, 0,
        0, 2, 0,   0, 0,
        0, 0, 0,   0, 0,
        0, 0, 0,  10, 0,
        0, 0, 0,   0, 0,
        0, 0, 0,   0, 0,
        0, 3, 0,   0, 0,
        0, 0, 0,   0, 0,
        0, 0, 0, 155, 0,
        0, 0, 0,   0, 0
    },
    {
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0
    },
    {
        0, 0,  0, 0, 0, 0, 0,  0, 0,
        0, 0, 10, 0, 0, 0, 0,  0, 0,
        0, 0,  0, 0, 0, 0, 0,  0, 0,
        0, 0,  0, 0, 0, 0, 0,  0, 0,
        0, 0,  0, 0, 0, 0, 2,  0, 0,
        3, 0,  0, 0, 0, 0, 0,  0, 0,
        0, 0,  0, 0, 0, 0, 0,  0, 0,
        0, 0,  0, 0, 0, 0, 0,  0, 0,
        0, 0,  0, 0, 0, 0, 0,  0, 0,
        0, 0,  0, 0, 5, 0, 0,  0, 0,
        0, 0,  0, 0, 0, 0, 0,  0, 1,
        0, 0,  0, 0, 0, 0, 0,  0, 0,
        0, 0,  0, 0, 0, 0, 0,  0, 0,
        0, 1,  0, 0, 0, 0, 0,  0, 0,
        0, 0,  0, 0, 0, 0, 0, 55, 0,
        0, 0,  0, 0, 0, 0, 0,  0, 0
    }
};
std::vector<std::vector<std::vector<Coordinates>>> coo_indices = {
    {
        { }
    },
    {
        {
            Coordinates({ 0 }),
            Coordinates({ 4 })
        }
    },
    {
        { },
        { }
    },
    {
        {
            Coordinates({ 0, 0 }),
            Coordinates({ 1, 0 })
        },
        {
            Coordinates({ 0, 0 }),
            Coordinates({ 0, 2 }),
            Coordinates({ 1, 1 })
        }
    },
    {
        { },
        { },
        { }
    },
    {
        {
            Coordinates({ 0, 0, 0 }),
            Coordinates({ 1, 0, 0 })
        },
        {
            Coordinates({ 0, 1, 0 }),
            Coordinates({ 0, 3, 0 }),
            Coordinates({ 1, 1, 0}),
            Coordinates({ 1, 3, 0})
        },
        {
            Coordinates({ 0, 1, 1 }),
            Coordinates({ 0, 3, 3 }),
            Coordinates({ 1, 1, 1 }),
            Coordinates({ 1, 3, 3 })
        }
    },
    {
        { },
        { },
        { },
        { }
    },
    {
        {
            Coordinates({ 0, 0, 0, 0 }),
            Coordinates({ 1, 0, 0, 0 }),
            Coordinates({ 2, 0, 0, 0 }),
            Coordinates({ 3, 0, 0, 0 })
        },
        {
            Coordinates({ 0, 0, 0, 0 }),
            Coordinates({ 1, 0, 0, 0 }),
            Coordinates({ 2, 0, 0, 0 }),
            Coordinates({ 2, 1, 0, 0 }),
            Coordinates({ 3, 0, 0, 0 }),
            Coordinates({ 3, 1, 0, 0 })
        },
        {
            Coordinates({ 0, 0, 1, 0 }),
            Coordinates({ 1, 0, 0, 0 }),
            Coordinates({ 1, 0, 1, 0 }),
            Coordinates({ 2, 0, 1, 0 }),
            Coordinates({ 2, 1, 0, 0 }),
            Coordinates({ 3, 0, 1, 0 }),
            Coordinates({ 3, 1, 0, 0 })
        },
        {
            Coordinates({ 0, 0, 1, 2 }),
            Coordinates({ 1, 0, 0, 6 }),
            Coordinates({ 1, 0, 1, 0 }),
            Coordinates({ 2, 0, 1, 4 }),
            Coordinates({ 2, 1, 0, 8 }),
            Coordinates({ 3, 0, 1, 1 }),
            Coordinates({ 3, 1, 0, 7 })
        }
    }
};
std::vector<std::vector<Coordinates>> csr_nnz_coordinates = {
    // The first two shapes has a dimension < 2, so they are not represented in CSR format
    {
    },
    {
    },
    {
    },
    {
        Coordinates({ 0, 0 }),
        Coordinates({ 0, 2 }),
        Coordinates({ 1, 1 })
    },
    {
    },
    {
        Coordinates({ 1, 1 })
    },
    {
    },
    {
    }
};

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

        equal = are_values_equal(a_value, b_value, a.data_type(), element_size);
    });

    return equal;
}
} // namespace

namespace test
{
namespace validation
{
TEST_SUITE(CPU)
TEST_SUITE(UNIT)
TEST_SUITE(SparseTensor)

TEST_CASE(ConvertDenseTensorToCOOTensor, framework::DatasetMode::ALL)
{
    for(size_t i = 0; i < shapes.size(); ++i)
    {
        for(auto type : types)
        {
            const auto src_info = TensorInfo(shapes[i], 1, type, layouts[0]);
            auto            src = create_tensor<Tensor>(src_info);
            bool   is_src_dense = src.info()->tensor_format() == TensorFormat::Dense;

            src.allocator()->allocate();
            library->fill_static_values(Accessor(src), sparse_data[i]);

            ARM_COMPUTE_EXPECT(is_src_dense, framework::LogLevel::ERRORS);

            for(size_t sparse_dim = 1; sparse_dim <= shapes[i].num_dimensions(); sparse_dim++)
            {
                auto st = src.to_coo_sparse(sparse_dim);

#ifdef ARM_COMPUTE_ASSERTS_ENABLED
                st->print(std::cout);
#endif // ARM_COMPUTE_ASSERTS_ENABLED

                bool   is_sparse = st->info()->is_sparse();
                bool      is_coo = st->info()->tensor_format() == TensorFormat::COO;
                size_t dense_dim = shapes[i].num_dimensions() - sparse_dim;
                size_t is_hybrid = dense_dim > 0;
                size_t       nnz = coo_indices[i][sparse_dim - 1].size();

                ARM_COMPUTE_EXPECT(is_sparse, framework::LogLevel::ERRORS);
                ARM_COMPUTE_EXPECT(is_coo, framework::LogLevel::ERRORS);

                ARM_COMPUTE_EXPECT(st->sparse_dim() == sparse_dim, framework::LogLevel::ERRORS);
                ARM_COMPUTE_EXPECT(st->dense_dim() == dense_dim, framework::LogLevel::ERRORS);
                ARM_COMPUTE_EXPECT(st->is_hybrid() == is_hybrid, framework::LogLevel::ERRORS);
                ARM_COMPUTE_EXPECT(st->nnz() == nnz, framework::LogLevel::ERRORS);

                for(size_t j = 0; j < nnz; ++j)
                {
                    const Coordinates coord = st->get_coordinates(j);
                    
                    for(size_t k = 0; k < coord.num_dimensions(); ++k)
                    {
                        ARM_COMPUTE_EXPECT(coord[k] == coo_indices[i][sparse_dim - 1][j][k], framework::LogLevel::ERRORS);
                    }

                    const uint8_t *value = st->get_value(coord);
                    ARM_COMPUTE_EXPECT(value != nullptr, framework::LogLevel::ERRORS);
                }
            }
        }
    }
}

TEST_CASE(ConvertCOOTensorToDense, framework::DatasetMode::ALL)
{
    for(size_t i = 0; i < shapes.size(); ++i)
    {
        for(auto type : types)
        {
            const auto t_info = TensorInfo(shapes[i], 1, type, layouts[0]);
            auto t            = create_tensor<Tensor>(t_info);

            t.allocator()->allocate();
            library->fill_tensor_sparse_random(Accessor(t), 0.2);

            for(size_t sparse_dim = 1; sparse_dim <= shapes[i].num_dimensions(); sparse_dim++)
            {
                auto st = t.to_coo_sparse(sparse_dim);
                auto td = st->to_dense();
                ARM_COMPUTE_EXPECT(tensors_are_equal(Accessor(t), Accessor(*td)), framework::LogLevel::ERRORS);
            }
        }
    }
}

TEST_CASE(ConvertDenseTensorToCSRTensor, framework::DatasetMode::ALL)
{
    for(size_t i = 0; i < shapes.size(); ++i)
    {
        for(auto type : types)
        {
            // Currently, CSRTensor only supports 2D tensors
            if(shapes[i].num_dimensions() < 2)
            {
                continue;
            }
            const TensorShape shape(shapes[i][0], shapes[i][1]);
            std::vector<unsigned> data(sparse_data[i].begin(), sparse_data[i].begin() + shapes[i][0] * shapes[i][1]);

            const auto src_info = TensorInfo(shape, 1, type, layouts[0]);
            auto            src = create_tensor<Tensor>(src_info);
            bool   is_src_dense = src.info()->tensor_format() == TensorFormat::Dense;

            src.allocator()->allocate();
            library->fill_static_values(Accessor(src), data);

            ARM_COMPUTE_EXPECT(is_src_dense, framework::LogLevel::ERRORS);

            auto st = src.to_csr_sparse();

#ifdef ARM_COMPUTE_ASSERTS_ENABLED
            st->print(std::cout);
#endif // ARM_COMPUTE_ASSERTS_ENABLED

            bool is_sparse = st->info()->is_sparse();
            bool    is_csr = st->info()->tensor_format() == TensorFormat::CSR;
            size_t     nnz = csr_nnz_coordinates[i].empty() ? 0 : csr_nnz_coordinates[i].size();

            ARM_COMPUTE_EXPECT(is_sparse, framework::LogLevel::ERRORS);
            ARM_COMPUTE_EXPECT(is_csr, framework::LogLevel::ERRORS);
            ARM_COMPUTE_EXPECT(nnz == st->nnz(), framework::LogLevel::ERRORS);

            for(size_t j = 0; j < nnz; ++j)
            {
                const Coordinates coord = st->get_coordinates(j);
                for(size_t k = 0; k < coord.num_dimensions(); ++k)
                {
                    ARM_COMPUTE_EXPECT(csr_nnz_coordinates[i][j][k] == coord[k], framework::LogLevel::ERRORS);
                }

                const uint8_t *value = st->get_value(coord);
                ARM_COMPUTE_EXPECT(value != nullptr, framework::LogLevel::ERRORS);
            }
        }
    }
}

TEST_CASE(ConvertCSRTensorToDense, framework::DatasetMode::ALL)
{
    for(size_t i = 0; i < shapes.size(); ++i)
    {
        for(auto type : types)
        {
            // Currently, CSRTensor only supports 2D tensors
            if(shapes[i].num_dimensions() < 2)
            {
                continue;
            }
            const TensorShape shape(shapes[i][0], shapes[i][1]);

            const auto t_info = TensorInfo(shape, 1, type, layouts[0]);
            auto t            = create_tensor<Tensor>(t_info);

            t.allocator()->allocate();
            library->fill_tensor_sparse_random(Accessor(t), 0.2);

            auto st = t.to_csr_sparse();
            auto td = st->to_dense();

            ARM_COMPUTE_EXPECT(tensors_are_equal(Accessor(t), Accessor(*td)), framework::LogLevel::ERRORS);
        }
    }
}

TEST_SUITE_END() // SparseTensor
TEST_SUITE_END() // UNIT
TEST_SUITE_END() // CPU

} // namespace validation
} // namespace test
} // namespace arm_compute
