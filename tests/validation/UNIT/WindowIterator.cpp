/*
 * Copyright (c) 2017-2018 Arm Limited.
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
#include "arm_compute/core/WindowIterator.h"
#include "tests/Utils.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "utils/TypePrinter.h"

#include <stdexcept>

using namespace arm_compute;
using namespace arm_compute::test;
using namespace arm_compute::test::validation;

TEST_SUITE(UNIT)
TEST_SUITE(WindowIterator)

template <typename Dim, typename... Dims>
Window create_window(Dim &&dim0, Dims &&... dims)
{
    Window win;
    const std::array < Dim, 1 + sizeof...(Dims) > dimensions{ { dim0, std::forward<Dims>(dims)... } };
    for(size_t i = 0; i < dimensions.size(); i++)
    {
        win.set(i, dimensions[i]);
    }
    return win;
}

template <typename T>
std::vector<T> create_vector(std::initializer_list<T> list_objs)
{
    std::vector<T> vec_objs;
    for(auto it : list_objs)
    {
        vec_objs.push_back(it);
    }
    return vec_objs;
}

DATA_TEST_CASE(WholeWindow, framework::DatasetMode::ALL, zip(framework::dataset::make("Window", { create_window(Window::Dimension(0, 1)),
                                                                                                  create_window(Window::Dimension(1, 5, 2), Window::Dimension(3, 5)),
                                                                                                  create_window(Window::Dimension(4, 16, 4), Window::Dimension(3, 13, 5), Window::Dimension(1, 3, 2))
                                                                                                }),
                                                             framework::dataset::make("Expected", { create_vector({ Coordinates(0, 0) }),
                                                                                                    create_vector({ Coordinates(1, 3), Coordinates(3, 3), Coordinates(1, 4), Coordinates(3, 4) }),
                                                                                                    create_vector({ Coordinates(4, 3, 1), Coordinates(8, 3, 1), Coordinates(12, 3, 1), Coordinates(4, 8, 1), Coordinates(8, 8, 1), Coordinates(12, 8, 1) })
                                                                                                  })),
               window, expected)
{
    unsigned int i            = 0;
    int          row_size     = 0;
    TensorShape  window_shape = window.shape();
    Coordinates  start_offset = index2coords(window_shape, 0);
    Coordinates  end_offset   = index2coords(window_shape, window.num_iterations_total() - 1);
    auto window_iterator      = create_window_iterator(window, start_offset, end_offset, [&](const Coordinates & id)
    {
        ARM_COMPUTE_EXPECT_EQUAL(row_size, (window[0].end() - window[0].start()), framework::LogLevel::ERRORS);
        ARM_COMPUTE_ASSERT(i < expected.size());
        Coordinates expected_coords = expected[i++];
        //Set number of dimensions to the maximum (To match the number of dimensions used by the id passed to the lambda function)
        expected_coords.set_num_dimensions(Coordinates::num_max_dimensions);
        ARM_COMPUTE_EXPECT_EQUAL(id, expected_coords, framework::LogLevel::ERRORS);
    });
    window_iterator.iterate_3D([&](int start, int end)
    {
        ARM_COMPUTE_EXPECT_EQUAL(window[0].start(), start, framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT_EQUAL(window[0].end(), end, framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(end > start, framework::LogLevel::ERRORS);
        row_size = end - start;
    });
    ARM_COMPUTE_EXPECT_EQUAL(i, expected.size(), framework::LogLevel::ERRORS);
}

DATA_TEST_CASE(PartialWindow2D, framework::DatasetMode::ALL, zip(zip(zip(combine(framework::dataset::make("Window",
                                                                                                          create_window(Window::Dimension(4, 20, 4), Window::Dimension(3, 32, 5), Window::Dimension(1, 2, 1))),
                                                                                 framework::dataset::make("Start", { 0, 1, 3, 2, 4 })),
                                                                         framework::dataset::make("End", { 0, 2, 5, 8, 7 })),
                                                                     framework::dataset::make("RowSize",
{
    create_vector({ 4 }),
    create_vector({ 8, 8 }),
    create_vector({ 4, 8, 8 }),
    create_vector({ 8, 8, 16, 16, 16, 16, 4 }),
    create_vector({ 16, 16, 16, 16 }),
})),
framework::dataset::make("Expected", { create_vector({ Coordinates(4, 3, 1) }), create_vector({ Coordinates(8, 3, 1), Coordinates(12, 3, 1) }), create_vector({ Coordinates(16, 3, 1), Coordinates(4, 8, 1), Coordinates(8, 8, 1) }), create_vector({ Coordinates(12, 3, 1), Coordinates(16, 3, 1), Coordinates(4, 8, 1), Coordinates(8, 8, 1), Coordinates(12, 8, 1), Coordinates(16, 8, 1), Coordinates(4, 13, 1) }), create_vector({ Coordinates(4, 8, 1), Coordinates(8, 8, 1), Coordinates(12, 8, 1), Coordinates(16, 8, 1) }) })),
window, start, end, expected_row_size, expected)
{
    unsigned int i            = 0;
    int          row_size     = 0;
    TensorShape  window_shape = window.shape();
    Coordinates  start_offset = index2coords(window_shape, start);
    Coordinates  end_offset   = index2coords(window_shape, end);
    auto window_iterator      = create_window_iterator(window, start_offset, end_offset, [&](const Coordinates & id)
    {
        ARM_COMPUTE_ASSERT(i < expected.size());
        ARM_COMPUTE_EXPECT_EQUAL(expected_row_size[i], row_size, framework::LogLevel::ERRORS);
        Coordinates expected_coords = expected[i++];
        //Set number of dimensions to the maximum (To match the number of dimensions used by the id passed to the lambda function)
        expected_coords.set_num_dimensions(Coordinates::num_max_dimensions);
        ARM_COMPUTE_EXPECT_EQUAL(id, expected_coords, framework::LogLevel::ERRORS);
    });
    window_iterator.iterate_3D([&](int start, int end)
    {
        ARM_COMPUTE_EXPECT(start >= window[0].start(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(end <= window[0].end(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(end > start, framework::LogLevel::ERRORS);
        row_size = end - start;
    });
    ARM_COMPUTE_EXPECT_EQUAL(i, expected.size(), framework::LogLevel::ERRORS);
}

TEST_SUITE_END()
TEST_SUITE_END()
