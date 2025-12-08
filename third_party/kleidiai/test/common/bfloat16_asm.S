//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/common/assembly.h"

    KAI_ASM_HEADER

    KAI_ASM_EXPORT(kai_test_float_to_bfloat16_bfcvt)

KAI_ASM_FUNCTION(kai_test_float_to_bfloat16_bfcvt)
        KAI_ASM_INST(0x1e634000)  // bfcvt h0, s0
        fmov w0, h0
        ret

    KAI_ASM_FOOTER
