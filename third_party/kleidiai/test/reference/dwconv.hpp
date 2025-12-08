//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <cstddef>

#include "test/common/buffer.hpp"
#include "test/common/data_type.hpp"
#include "test/common/memory.hpp"

namespace kai::test {

struct Padding2D {
    size_t left;
    size_t right;
    size_t top;
    size_t bottom;

    struct Hash {
        size_t operator()(const Padding2D pad) const {
            return                                       //
                (std::hash<size_t>{}(pad.left) << 0) ^   //
                (std::hash<size_t>{}(pad.right) << 1) ^  //
                (std::hash<size_t>{}(pad.top) << 2) ^    //
                (std::hash<size_t>{}(pad.bottom) << 3);
        }
    };

private:
    friend bool operator==(const Padding2D& lhs, const Padding2D& rhs) {
        return  //
            lhs.left == rhs.left && lhs.right == rhs.right && lhs.top == rhs.top && lhs.bottom == rhs.bottom;
    }

    friend std::ostream& operator<<(std::ostream& os, const Padding2D& pad);
};

void PrintTo(const Padding2D& pad, std::ostream* os);

/// Depthwise Convolution function
///
/// @tparam T Data type.
///
/// @param[in] batches   Batch dimension of feature map.
/// @param[in] in_height height of feature map.
/// @param[in] in_width  width of feature map.
/// @param[in] channels  Number of channels in feature map.
/// @param[in] filter_height Height dimension in filter.
/// @param[in] filter_width  Width of convolution filter.
/// @param[in] feature_map Ptr to start of feature map.
/// @param[in] weights Ptr to start of weights buffer/tensor.
/// @param[in] bias Ptr to start of bias buffer.
/// @param[in] pad  Padding object.
///
/// @return The result data buffer.
template <typename T>
Buffer depthwise_reference(
    const size_t batches, const size_t in_height, const size_t in_width, const size_t channels,
    const size_t filter_height, const size_t filter_width, const void* feature_map, const void* weights,
    const void* bias, const Padding2D& pad);

}  // namespace kai::test
