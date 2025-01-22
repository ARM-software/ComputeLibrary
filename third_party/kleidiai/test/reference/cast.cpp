//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include <cstddef>
#include <cstdint>
#include <vector>

#include "kai/kai_common.h"
#include "test/common/bfloat16.hpp"
#include "test/common/data_type.hpp"
#include "test/common/float16.hpp"
#include "test/common/memory.hpp"
#include "test/common/round.hpp"

namespace kai::test {

template <typename DstType, typename SrcType>
std::vector<uint8_t> cast(const void* src, size_t length) {
    std::vector<uint8_t> dst(round_up_division(length * size_in_bits<DstType>, 8));

    for (size_t i = 0; i < length; ++i) {
        write_array(dst.data(), i, static_cast<DstType>(read_array<SrcType>(src, i)));
    }

    return dst;
}

template std::vector<uint8_t> cast<Float16, float>(const void* src, size_t length);
template std::vector<uint8_t> cast<BFloat16, float>(const void* src, size_t length);

std::vector<uint8_t> cast(const void* src, kai::test::DataType src_dt, DataType dst_dt, size_t height, size_t width) {
    const auto length = height * width;

    if (src_dt == DataType::BF16 && dst_dt == DataType::FP32) {
        return cast<float, BFloat16>(src, length);
    } else if (src_dt == DataType::FP16 && dst_dt == DataType::FP32) {
        return cast<float, Float16>(src, length);
    }

    KAI_ERROR("Unsupported cast data type!");
}

std::vector<uint8_t> cast_qsu4_qsi4(const void* src, size_t length) {
    std::vector<uint8_t> dst(round_up_division(length, 2));

    for (size_t i = 0; i < length; ++i) {
        write_array(dst.data(), i, static_cast<UInt4>(static_cast<int32_t>(read_array<Int4>(src, i)) + 8));
    }

    return dst;
}

}  // namespace kai::test
