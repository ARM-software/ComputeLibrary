//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "kai_rhs_dwconv_pack_x32p1vlx1b_x32_x32_sme.h"

#include <stdint.h>
#include <string.h>

#include "kai/kai_common.h"

size_t kai_rhs_get_dst_size_dwconv_pack_x32p1vlx1b_x32_x32_sme(
    size_t filter_height, size_t filter_width, size_t num_channels) {
    const size_t depth_elements = kai_roundup(num_channels, kai_get_sme_vector_length_u32());
    return depth_elements * (filter_height * filter_width + 1) * sizeof(float);
}

void kai_run_rhs_dwconv_pack_x32p1vlx1b_x32_x32_sme(
    size_t filter_height, size_t filter_width, size_t height, size_t width, size_t num_channels, const void* rhs,
    const void* bias, void* rhs_packed) {
    KAI_ASSUME(rhs != NULL);
    KAI_ASSUME(rhs_packed != NULL);
    KAI_ASSUME(bias != NULL);
    KAI_UNUSED(height);
    KAI_UNUSED(width);

    // Cast the pointers to byte sizes
    const uint8_t* src = (const uint8_t*)(rhs);
    const uint8_t* bias_ptr = (const uint8_t*)(bias);
    uint8_t* dst = (uint8_t*)(rhs_packed);

    const size_t vl = kai_get_sme_vector_length_u32();
    const size_t element_size = sizeof(float);

    for (size_t n = 0; n < num_channels; n += vl) {
        const size_t count = (vl < (num_channels - n)) ? vl : (num_channels - n);
        memcpy(dst, bias_ptr, count * element_size);
        dst += (vl * element_size);
        bias_ptr += (count * element_size);

        for (size_t idx = 0; idx < filter_height * filter_width; idx++) {
            const uint8_t* src_ptr = src + ((idx * num_channels + n) * element_size);
            memcpy(dst, src_ptr, count * element_size);
            dst += (vl * element_size);  // move ptr.
        }
    }
}
