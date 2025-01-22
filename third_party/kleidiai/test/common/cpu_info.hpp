//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

namespace kai::test {

/// Returns a value indicating whether the current CPU supports FEAT_AdvSIMD.
bool cpu_has_advsimd();

/// Returns a value indicating whether the current CPU supports FEAT_DotProd.
bool cpu_has_dotprod();

/// Returns a value indicating whether the current CPU supports FEAT_I8MM.
bool cpu_has_i8mm();

/// Returns a value indicating whether the current CPU supports FEAT_FP16.
bool cpu_has_fp16();

/// Returns a value indicating whether the current CPU supports FEAT_BF16.
bool cpu_has_bf16();

/// Returns a value indicating whether the current CPU supports FEAT_SVE.
bool cpu_has_sve();

/// Returns a value indicating whether the current CPU supports FEAT_SVE2.
bool cpu_has_sve2();

/// Returns a value indicating whether the current CPU supports FEAT_SME.
bool cpu_has_sme();

/// Returns a value indicating whether the current CPU supports FEAT_SME2.
bool cpu_has_sme2();

}  // namespace kai::test
