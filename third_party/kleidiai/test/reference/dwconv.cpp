//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/reference/dwconv.hpp"

#include <ostream>

namespace kai::test {

std::ostream& operator<<(std::ostream& os, const Padding2D& pad) {
    os << " [ " << pad.left << " , " << pad.right << " ," << pad.top << " , " << pad.bottom << " ] ";
    return os;
}

void PrintTo(const Padding2D& pad, std::ostream* os) {
    *os << "PAD_" << pad.left << "_" << pad.right << "_" << pad.bottom << "_" << pad.top;
};

template <typename T>
Buffer depthwise_reference(
    const size_t batches, const size_t in_height, const size_t in_width, const size_t channels,
    const size_t filter_height, const size_t filter_width, const void* feature_map, const void* weights,
    const void* bias, const Padding2D& pad) {
    // Calculate output dims according to padding and input params.
    const size_t out_height = (in_height + pad.top + pad.bottom + 1 - filter_height);
    const size_t out_width = in_width + pad.left + pad.right + 1 - filter_width;
    const size_t out_size = out_height * out_width * batches * channels;

    // NOTE: We accumulate in datatype provided - this may need to change in the future.
    std::vector<T> acc(out_size, 0.0f);
    Buffer dst(out_size * size_in_bits<T> / 8);

    for (size_t b = 0; b < batches; ++b) {
        for (size_t out_h = 0; out_h < out_height; ++out_h) {
            for (size_t out_w = 0; out_w < out_width; ++out_w) {
                const size_t out_base = ((b * out_height + out_h) * out_width + out_w) * channels;

                // Apply filter to feature map.
                for (size_t ic = 0; ic < channels; ++ic) {
                    float sum = 0.0f;

                    for (size_t kernel_h = 0; kernel_h < filter_height; ++kernel_h) {
                        // Determine if input height bounds. If not, then this is padding.
                        const int in_y = static_cast<int>(out_h + kernel_h) - static_cast<int>(pad.top);
                        if (in_y < 0 || in_height <= static_cast<size_t>(in_y)) continue;

                        for (size_t kernel_w = 0; kernel_w < filter_width; ++kernel_w) {
                            // Determine if in input width bounds, if not this is padding.
                            const int in_x = static_cast<int>(out_w + kernel_w) - static_cast<int>(pad.left);
                            if (in_x < 0 || in_width <= static_cast<size_t>(in_x)) continue;

                            auto in_idx = ((b * in_height + in_y) * in_width + in_x) * channels + ic;
                            auto weights_idx = ((kernel_h * filter_width) + kernel_w) * channels + ic;

                            auto wei_value = read_array<T>(weights, weights_idx);
                            auto in_value = read_array<T>(feature_map, in_idx);

                            // Perform actual accumulation and store in output vector
                            sum += in_value * wei_value;
                        }
                    }

                    auto out_idx = out_base + ic;
                    sum = sum + (T)read_array<T>(bias, ic);
                    write_array<T>(dst.data(), out_idx, sum);
                }
            }
        }
    }
    return dst;
}

// Explicit template
template Buffer depthwise_reference<float>(
    const size_t batches, const size_t in_height, const size_t in_width, const size_t channels,
    const size_t filter_height, const size_t filter_width, const void* feature_map, const void* weights,
    const void* bias, const Padding2D& pad);

}  // namespace kai::test
