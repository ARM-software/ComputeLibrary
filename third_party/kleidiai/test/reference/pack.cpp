//
// SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/reference/pack.hpp"

#include <arm_neon.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "kai/kai_common.h"
#include "test/common/bfloat16.hpp"
#include "test/common/buffer.hpp"
#include "test/common/data_format.hpp"
#include "test/common/data_type.hpp"
#include "test/common/float16.hpp"
#include "test/common/memory.hpp"
#include "test/common/round.hpp"

namespace kai::test {

namespace {

BFloat16<> convert(const uint8_t* src_ptr_elm, DataType src_dtype, DataType dst_dtype) {
    KAI_ASSUME_ALWAYS((src_dtype == DataType::FP32 || src_dtype == DataType::FP16) && dst_dtype == DataType::BF16);

    switch (src_dtype) {
        case DataType::FP32:
            return BFloat16<>(*reinterpret_cast<const float*>(src_ptr_elm));
        case DataType::FP16:
            return BFloat16<>(static_cast<float>(*reinterpret_cast<const Float16*>(src_ptr_elm)));
        default:
            KAI_ERROR("Unsupported Data Type");
    }
}

Buffer pack_block(
    const void* src, DataType src_dtype, DataType dst_dtype, size_t src_esize, size_t dst_esize, size_t full_height,
    size_t full_width, size_t block_height, size_t block_width, size_t subblock_height, size_t subblock_width) {
    const auto dst_bytes =
        round_up_multiple(full_height, block_height) * round_up_multiple(full_width, block_width) * dst_esize;

    Buffer dst(dst_bytes, 0);

    const auto* src_ptr = reinterpret_cast<const uint8_t*>(src);
    auto* dst_ptr = dst.data();

    for (size_t y_block = 0; y_block < full_height; y_block += block_height) {
        for (size_t x_block = 0; x_block < full_width; x_block += block_width) {
            for (size_t y_subblock = 0; y_subblock < block_height; y_subblock += subblock_height) {
                for (size_t x_subblock = 0; x_subblock < block_width; x_subblock += subblock_width) {
                    for (size_t y_element = 0; y_element < subblock_height; ++y_element) {
                        if (src_dtype == dst_dtype) {
                            const size_t esize = dst_esize;

                            if (y_block + y_subblock + y_element < full_height) {
                                const size_t y_offset = (y_block + y_subblock + y_element) * full_width;
                                const size_t x_offset = x_block + x_subblock;
                                const size_t offset = y_offset + x_offset;
                                const auto len = std::min(subblock_width, full_width - x_offset);

                                memcpy(dst_ptr, src_ptr + offset * esize, len * esize);
                            }

                            dst_ptr += subblock_width * esize;
                        } else if (dst_esize == 2 /* 16 bits */) {
                            for (size_t x_element = 0; x_element < subblock_width; ++x_element) {
                                if (y_block + y_subblock + y_element < full_height) {
                                    if (x_block + x_subblock + x_element < full_width) {
                                        const uint8_t* src_ptr_elm = src_ptr +
                                            ((y_block + y_subblock + y_element) * full_width + x_block + x_subblock +
                                             x_element) *
                                                src_esize;

                                        const BFloat16 src_value = convert(src_ptr_elm, src_dtype, dst_dtype);
                                        memcpy(dst_ptr, &src_value, dst_esize);
                                    }
                                }

                                dst_ptr += dst_esize;
                            }
                        }
                    }
                }
            }
        }
    }

    KAI_ASSERT_ALWAYS(reinterpret_cast<uintptr_t>(dst_ptr) - reinterpret_cast<uintptr_t>(dst.data()) == dst_bytes);

    return dst;
}

/// Packs the matrix from raw to per-row bias format.
Buffer pack_bias_per_row(
    DataType src_dtype, DataType bias_dtype, DataType dst_dtype, size_t src_esize, size_t bias_esize, size_t dst_esize,
    const void* src, const void* bias, size_t height, size_t width, size_t block_height, size_t block_width,
    size_t subblock_height, size_t subblock_width) {
    KAI_ASSUME_ALWAYS(src_dtype == bias_dtype);

    const auto num_groups = (height + block_height - 1) / block_height;
    const auto group_num_blocks = (width + block_width - 1) / block_width;
    const auto group_bias_bytes = block_height * bias_esize;
    const auto block_data_bytes = block_height * block_width * dst_esize;
    const auto group_bytes = group_bias_bytes + group_num_blocks * block_data_bytes;
    const auto dst_bytes = num_groups * group_bytes;

    Buffer dst(dst_bytes, 0);

    const auto* src_ptr = reinterpret_cast<const uint8_t*>(src);
    const auto* bias_ptr = reinterpret_cast<const uint8_t*>(bias);
    auto* dst_ptr = dst.data();

    for (size_t y_block = 0; y_block < height; y_block += block_height) {
        // Packs the bias.
        const auto bias_len = std::min(block_height, height - y_block);
        memcpy(dst_ptr, bias_ptr, bias_len * bias_esize);
        bias_ptr += block_height * bias_esize;
        dst_ptr += block_height * bias_esize;

        for (size_t x_block = 0; x_block < width; x_block += block_width) {
            for (size_t y_subblock = 0; y_subblock < block_height; y_subblock += subblock_height) {
                for (size_t x_subblock = 0; x_subblock < block_width; x_subblock += subblock_width) {
                    for (size_t y_element = 0; y_element < subblock_height; ++y_element) {
                        if (src_dtype == dst_dtype) {
                            const size_t esize = dst_esize;
                            if (y_block + y_subblock + y_element < height) {
                                const auto len = std::min(subblock_width, width - x_block - x_subblock);

                                memcpy(
                                    dst_ptr,
                                    src_ptr +
                                        ((y_block + y_subblock + y_element) * width + x_block + x_subblock) * esize,
                                    len * esize);
                            }

                            dst_ptr += subblock_width * esize;
                        } else if (dst_esize == 2 /* 16 bits */) {
                            for (size_t x_element = 0; x_element < subblock_width; ++x_element) {
                                if (y_block + y_subblock + y_element < height) {
                                    if (x_block + x_subblock + x_element < width) {
                                        const uint8_t* src_ptr_elm = src_ptr +
                                            ((y_block + y_subblock + y_element) * width + x_block + x_subblock +
                                             x_element) *
                                                src_esize;

                                        const BFloat16 dst_value = convert(src_ptr_elm, src_dtype, dst_dtype);
                                        memcpy(dst_ptr, &dst_value, dst_esize);
                                    }
                                }

                                dst_ptr += dst_esize;
                            }
                        }
                    }
                }
            }
        }
    }

    KAI_ASSERT_ALWAYS(reinterpret_cast<uintptr_t>(dst_ptr) - reinterpret_cast<uintptr_t>(dst.data()) == dst_bytes);

    return dst;
}

}  // namespace

Buffer pack(
    const DataFormat& dst_format, const void* src, [[maybe_unused]] const void* scales, const void* bias,
    const DataFormat& src_format, size_t height, size_t width) {
    const auto dst_dt = dst_format.data_type();
    const auto dst_qf = dst_format.pack_format();
    const auto src_dt = src_format.data_type();
    const auto src_qf = src_format.pack_format();

    const auto block_height = dst_format.actual_block_height(height);
    const auto block_width = dst_format.actual_block_width(width);
    const auto subblock_height = dst_format.actual_subblock_height(height);
    const auto subblock_width = dst_format.actual_subblock_width(width);

    if (src_qf == DataFormat::PackFormat::NONE && dst_qf == DataFormat::PackFormat::BIAS_PER_ROW) {
        KAI_ASSUME_ALWAYS(
            (src_dt == dst_dt) || (src_dt == DataType::FP32 && dst_dt == DataType::BF16) ||
            (src_dt == DataType::FP16 && dst_dt == DataType::BF16));

        const auto src_esize = data_type_size_in_bits(src_dt);
        const auto dst_esize = data_type_size_in_bits(dst_dt);
        const auto bias_esize = data_type_size_in_bits(dst_format.zero_point_data_type());
        const auto bias_dt = dst_format.zero_point_data_type();

        KAI_ASSUME_ALWAYS(dst_esize % 8 == 0 && bias_esize % 8 == 0 && src_esize % 8 == 0);

        return pack_bias_per_row(
            src_dt, bias_dt, dst_dt, src_esize / 8, bias_esize / 8, dst_esize / 8, src, bias, height, width,
            block_height, block_width, subblock_height, subblock_width);
    }

    if (src_qf == DataFormat::PackFormat::NONE && dst_qf == DataFormat::PackFormat::NONE) {
        KAI_ASSUME_ALWAYS(
            (src_dt == dst_dt) || (src_dt == DataType::FP32 && dst_dt == DataType::BF16) ||
            (src_dt == DataType::FP16 && dst_dt == DataType::BF16));

        const auto dst_esize = data_type_size_in_bits(dst_dt);
        const auto src_esize = data_type_size_in_bits(src_dt);

        KAI_ASSUME_ALWAYS(src_esize % 8 == 0 && dst_esize % 8 == 0);

        return pack_block(
            src, src_dt, dst_dt, src_esize / 8, dst_esize / 8, height, width, block_height, block_width,
            subblock_height, subblock_width);
    }

    KAI_ERROR("Unsupported operation!");
}

template <typename Data, typename Scale>
Buffer pack_data_scales(const void* data, const void* scales, size_t height, size_t width, size_t quant_width) {
    KAI_ASSUME_ALWAYS_IF(size_in_bits<Data> < 8, quant_width % (8 / size_in_bits<Data>) == 0);
    KAI_ASSUME_ALWAYS_IF(size_in_bits<Data> < 8, width % (8 / size_in_bits<Data>) == 0);

    const auto num_quant_packets_x = round_up_multiple(width, quant_width) / quant_width;

    const auto data_bytes = height * width * size_in_bits<Data> / 8;
    const auto scales_bytes = height * num_quant_packets_x * sizeof(Scale);

    Buffer dst(data_bytes + scales_bytes);

    const auto* scales_ptr = reinterpret_cast<const Scale*>(scales);
    auto* dst_ptr = dst.data();

    for (size_t y = 0; y < height; ++y) {
        for (size_t x_quant = 0; x_quant < width; x_quant += quant_width) {
            write_array(dst_ptr, 0, *scales_ptr);
            dst_ptr += sizeof(Scale);
            ++scales_ptr;

            const auto len = std::min(x_quant + quant_width, width) - x_quant;

            for (size_t x_element = 0; x_element < len; ++x_element) {
                const auto x = x_quant + x_element;
                write_array(dst_ptr, x_element, read_array<Data>(data, y * width + x));
            }

            dst_ptr += len * size_in_bits<Data> / 8;
        }
    }

    KAI_ASSERT_ALWAYS(dst_ptr == dst.data() + dst.size());

    return dst;
}

template <typename ZeroPoint, typename Data, typename Scale>
Buffer pack_zero_points_data_scales_per_block(
    const void* zero_points, const void* data, const void* scales, size_t num_blocks, size_t block_num_zero_points,
    size_t block_num_data, size_t block_num_scales) {
    // Only data is allowed to be sub-byte.
    KAI_ASSUME_ALWAYS(size_in_bits<ZeroPoint> % 8 == 0);
    KAI_ASSUME_ALWAYS(size_in_bits<Scale> % 8 == 0);

    // Checks for memory alignment.
    KAI_ASSUME_ALWAYS(size_in_bits<ZeroPoint> % size_in_bits<Data> == 0);
    KAI_ASSUME_ALWAYS(
        (block_num_zero_points * size_in_bits<ZeroPoint> + block_num_data * size_in_bits<Data>) % size_in_bits<Scale> ==
        0);
    KAI_ASSUME_ALWAYS(
        (block_num_data * size_in_bits<Data> + block_num_scales * size_in_bits<Scale>) % size_in_bits<ZeroPoint> == 0);

    Buffer dst(round_up_division(
        num_blocks *
            (block_num_zero_points * size_in_bits<ZeroPoint> + block_num_data * size_in_bits<Data> +
             block_num_scales * size_in_bits<Scale>),
        8));
    auto* dst_ptr = dst.data();

    for (size_t block_no = 0; block_no < num_blocks; ++block_no) {
        for (size_t i = 0; i < block_num_zero_points; ++i) {
            write_array<ZeroPoint>(
                dst_ptr, i, read_array<ZeroPoint>(zero_points, block_no * block_num_zero_points + i));
        }
        dst_ptr += block_num_zero_points * sizeof(ZeroPoint);

        for (size_t i = 0; i < block_num_data; ++i) {
            write_array<Data>(dst_ptr, i, read_array<Data>(data, block_no * block_num_data + i));
        }
        dst_ptr += round_up_division(block_num_data * size_in_bits<Data>, 8);

        for (size_t i = 0; i < block_num_scales; ++i) {
            write_array<Scale>(dst_ptr, i, read_array<Scale>(scales, block_no * block_num_scales + i));
        }
        dst_ptr += block_num_scales * sizeof(Scale);
    }

    KAI_ASSERT_ALWAYS(dst_ptr == dst.data() + dst.size());

    return dst;
}

template Buffer pack_zero_points_data_scales_per_block<int32_t, int8_t, float>(
    const void* zero_points, const void* data, const void* scales, size_t num_blocks, size_t block_num_zero_points,
    size_t block_num_data, size_t block_num_scales);

template <typename Data, typename Scale>
Buffer pack_data_scales_interleave_block(
    const void* data, const void* scales, size_t height, size_t width, size_t quant_width) {
    KAI_ASSUME_ALWAYS_IF(size_in_bits<Data> < 8, quant_width % (8 / size_in_bits<Data>) == 0);
    KAI_ASSUME_ALWAYS_IF(size_in_bits<Data> < 8, width % (8 / size_in_bits<Data>) == 0);
    KAI_ASSUME_ALWAYS(width % quant_width == 0);
    KAI_ASSUME_ALWAYS(quant_width % 2 == 0);

    const auto num_quant_packets_x = round_up_multiple(width, quant_width) / quant_width;

    const auto data_bytes = height * width * size_in_bits<Data> / 8;
    const auto scales_bytes = scales != nullptr ? height * num_quant_packets_x * sizeof(Scale) : 0;

    Buffer dst(data_bytes + scales_bytes);

    const auto* scales_ptr = reinterpret_cast<const Scale*>(scales);
    auto* dst_ptr = dst.data();

    for (size_t y = 0; y < height; ++y) {
        for (size_t x_quant = 0; x_quant < width; x_quant += quant_width) {
            if (scales_ptr != nullptr) {
                write_array(dst_ptr, 0, *scales_ptr);
                dst_ptr += sizeof(Scale);
                ++scales_ptr;
            }

            for (size_t x_element = 0; x_element < quant_width; ++x_element) {
                const auto x = x_quant + x_element / 2 + (x_element % 2 != 0 ? quant_width / 2 : 0);
                write_array(dst_ptr, x_element, read_array<Data>(data, y * width + x));
            }

            dst_ptr += quant_width * size_in_bits<Data> / 8;
        }
    }

    KAI_ASSERT_ALWAYS(dst_ptr == dst.data() + dst.size());

    return dst;
}

template Buffer pack_data_scales_interleave_block<UInt4, Float16>(
    const void* data, const void* scales, size_t height, size_t width, size_t quant_width);
template Buffer pack_data_scales_interleave_block<UInt4, std::nullptr_t>(
    const void* data, const void* scales, size_t height, size_t width, size_t quant_width);

template <typename Data, typename ZeroPoint, typename Scale, typename Bias>
Buffer pack_block_data_zero_points_scale_bias(
    const void* data, const void* zero_points, const void* scales, const void* biases, size_t height, size_t width,
    size_t quant_height, size_t quant_width, size_t block_height, size_t block_width, size_t interleave_x_blocks) {
    if (quant_width == width) {
        quant_width = round_up_multiple(quant_width, block_width);
    }

    KAI_ASSERT_ALWAYS(quant_height == block_height);
    KAI_ASSERT_ALWAYS(quant_width % block_width == 0);

    if (interleave_x_blocks == 0) {
        interleave_x_blocks = quant_width / block_width;
    }

    const auto has_zero_points = zero_points != nullptr;
    const auto has_biases = biases != nullptr;

    const auto num_quant_packets_y = round_up_division(height, quant_height);
    const auto num_quant_packets_x = round_up_division(width, quant_width);

    const auto quant_packet_data_bytes = quant_height * quant_width * size_in_bits<Data> / 8;
    const auto quant_packet_zero_points_bytes = has_zero_points ? quant_height * sizeof(ZeroPoint) : 0;
    const auto quant_packet_scales_bytes = quant_height * sizeof(Scale);
    const auto quant_packet_bytes =
        quant_packet_zero_points_bytes + quant_packet_data_bytes + quant_packet_scales_bytes;

    const auto num_quant_packets_per_row = round_up_division(width, quant_width);
    const auto biases_bytes = has_biases ? height * sizeof(Bias) : 0;

    const auto dst_bytes = num_quant_packets_y * num_quant_packets_x * quant_packet_bytes + biases_bytes;
    Buffer dst(dst_bytes);

    const auto* zero_points_ptr = reinterpret_cast<const ZeroPoint*>(zero_points);
    const auto* scales_ptr = reinterpret_cast<const Scale*>(scales);
    const auto* biases_ptr = reinterpret_cast<const Bias*>(biases);
    auto* dst_ptr = dst.data();

    for (size_t y_quant = 0; y_quant < height; y_quant += quant_height) {
        for (size_t x_quant = 0; x_quant < width; x_quant += quant_width) {
            size_t dst_index = 0;

            // Packs the data.
            for (size_t y_pack = 0; y_pack < quant_height; y_pack += block_height) {
                for (size_t x_pack = 0; x_pack < block_width * interleave_x_blocks; x_pack += block_width) {
                    for (size_t y_element = 0; y_element < block_height; ++y_element) {
                        for (size_t x_element = 0; x_element < block_width; ++x_element) {
                            for (size_t x_interleave = 0; x_interleave < quant_width;
                                 x_interleave += block_width * interleave_x_blocks) {
                                const auto y = y_quant + y_pack + y_element;
                                const auto x = x_quant + x_pack + x_element + x_interleave;

                                if (y < height && x < width) {
                                    write_array(dst_ptr, dst_index, read_array<Data>(data, y * width + x));
                                }

                                ++dst_index;
                            }
                        }
                    }
                }
            }

            dst_ptr += dst_index * size_in_bits<Data> / 8;

            // Packs the zero points.
            if (has_zero_points) {
                for (size_t y_element = 0; y_element < quant_height; ++y_element) {
                    const auto y = y_quant + y_element;
                    const auto x = x_quant / quant_width;
                    memcpy(dst_ptr, &zero_points_ptr[y * num_quant_packets_per_row + x], sizeof(ZeroPoint));
                    dst_ptr += sizeof(ZeroPoint);
                }
            }

            // Packs the scales.
            for (size_t y_element = 0; y_element < quant_height; ++y_element) {
                const auto y = y_quant + y_element;
                const auto x = x_quant / quant_width;
                memcpy(dst_ptr, &scales_ptr[y * num_quant_packets_per_row + x], sizeof(Scale));
                dst_ptr += sizeof(Scale);
            }
        }

        // Packs the biases.
        if (has_biases) {
            for (size_t y_element = 0; y_element < quant_height; ++y_element) {
                const auto y = y_quant + y_element;
                memcpy(dst_ptr, &biases_ptr[y], sizeof(Bias));
                dst_ptr += sizeof(Bias);
            }
        }
    }

    KAI_ASSERT_ALWAYS(dst_ptr == dst.data() + dst.size());

    return dst;
}

}  // namespace kai::test
