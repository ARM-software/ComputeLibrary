//
// SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/reference/cast.hpp"

#include <cstddef>
#include <cstdint>

#include "kai/kai_common.h"
#include "test/common/bfloat16.hpp"
#include "test/common/buffer.hpp"
#include "test/common/data_type.hpp"
#include "test/common/float16.hpp"
#include "test/common/memory.hpp"
#include "test/common/round.hpp"

namespace kai::test {

template <typename DstType, typename SrcType>
Buffer cast(const void* src, size_t length) {
    Buffer dst(round_up_division(length * size_in_bits<DstType>, 8));

    for (size_t i = 0; i < length; ++i) {
        write_array(dst.data(), i, static_cast<DstType>(read_array<SrcType>(src, i)));
    }

    return dst;
}

template <>
Buffer cast<BFloat16<false>, Float16>(const void* src, size_t length) {
    Buffer dst(round_up_division(length * size_in_bits<BFloat16<>>, 8));

    for (size_t i = 0; i < length; ++i) {
        float interim = static_cast<float>(read_array<Float16>(src, i));
        write_array(dst.data(), i, BFloat16<false>(interim));
    }

    return dst;
}

template <>
Buffer cast<BFloat16<true>, Float16>(const void* src, size_t length) {
    Buffer dst(round_up_division(length * size_in_bits<BFloat16<>>, 8));

    for (size_t i = 0; i < length; ++i) {
        float interim = static_cast<float>(read_array<Float16>(src, i));
        write_array(dst.data(), i, BFloat16<true>(interim));
    }

    return dst;
}

template Buffer cast<Float16, float>(const void* src, size_t length);
template Buffer cast<BFloat16<false>, float>(const void* src, size_t length);
template Buffer cast<BFloat16<true>, float>(const void* src, size_t length);
template Buffer cast<float, Float16>(const void* src, size_t length);
template Buffer cast<float, BFloat16<false>>(const void* src, size_t length);
template Buffer cast<float, BFloat16<true>>(const void* src, size_t length);

Buffer cast(const void* src, kai::test::DataType src_dt, DataType dst_dt, size_t height, size_t width) {
    const auto length = height * width;

    if (src_dt == DataType::BF16 && dst_dt == DataType::FP32) {
        return cast<float, BFloat16<>>(src, length);
    } else if (src_dt == DataType::FP16 && dst_dt == DataType::BF16) {
        return cast<BFloat16<>, Float16>(src, length);
    } else if (src_dt == DataType::FP32 && dst_dt == DataType::BF16) {
        return cast<BFloat16<>, float>(src, length);
    }

    KAI_ERROR("Unsupported cast data type!");
}

Buffer cast_qsu4_qsi4(const void* src, size_t length) {
    Buffer dst(round_up_division(length, 2));

    for (size_t i = 0; i < length; ++i) {
        write_array(dst.data(), i, static_cast<UInt4>(static_cast<int32_t>(read_array<Int4>(src, i)) + 8));
    }

    return dst;
}

}  // namespace kai::test
