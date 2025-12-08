//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/common/assembly.h"

    KAI_ASM_HEADER

    KAI_ASM_EXPORT(kai_test_round_to_nearest_even_i32_f32)
    KAI_ASM_EXPORT(kai_test_round_to_nearest_even_i64_f32)

KAI_ASM_FUNCTION(kai_test_round_to_nearest_even_i32_f32)
    fcvtns w0, s0
    ret

KAI_ASM_FUNCTION(kai_test_round_to_nearest_even_i64_f32)
    fcvtns x0, s0
    ret

    KAI_ASM_FOOTER
