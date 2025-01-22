//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/common/matrix_portion.hpp"

#include <algorithm>
#include <cstddef>

#include "kai/kai_common.h"
#include "test/common/rect.hpp"
#include "test/common/round.hpp"

namespace kai::test {

MatrixPortion::MatrixPortion(float start_row, float start_col, float height, float width) :
    _start_row(start_row), _start_col(start_col), _height(height), _width(width) {
}

float MatrixPortion::start_row() const {
    return _start_row;
}

float MatrixPortion::start_col() const {
    return _start_col;
}

float MatrixPortion::height() const {
    return _height;
}

float MatrixPortion::width() const {
    return _width;
}

Rect MatrixPortion::compute_portion(
    size_t full_height, size_t full_width, size_t scheduler_block_height, size_t scheduler_block_width) const {
    KAI_ASSUME(_start_row >= 0.0F && _start_row <= 1.0F);
    KAI_ASSUME(_start_col >= 0.0F && _start_col <= 1.0F);
    KAI_ASSUME(_height >= 0.0F && _height <= 1.0F);
    KAI_ASSUME(_width >= 0.0F && _width <= 1.0F);

    auto start_row = round_to_nearest_even_usize(_start_row * static_cast<float>(full_height));
    auto start_col = round_to_nearest_even_usize(_start_col * static_cast<float>(full_width));
    auto height = round_to_nearest_even_usize(_height * static_cast<float>(full_height));
    auto width = round_to_nearest_even_usize(_width * static_cast<float>(full_width));

    start_row = round_down_multiple(start_row, scheduler_block_height);
    start_col = round_down_multiple(start_col, scheduler_block_width);

    start_row = std::min(start_row, round_down_multiple(full_height, scheduler_block_height));
    start_col = std::min(start_col, round_down_multiple(full_width, scheduler_block_width));

    height = round_up_multiple(height, scheduler_block_height);
    width = round_up_multiple(width, scheduler_block_width);

    height = std::min(height, full_height - start_row);
    width = std::min(width, full_width - start_col);

    return {start_row, start_col, height, width};
}

}  // namespace kai::test
