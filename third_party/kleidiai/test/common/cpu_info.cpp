//
// SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/common/cpu_info.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <tuple>

#include "kai/kai_common.h"

#if defined(__aarch64__) && defined(__linux__)
#include <sys/auxv.h>
#endif  // defined(__aarch64__) && defined(__linux__)

#if defined(__aarch64__) && defined(__APPLE__)
#include <sys/sysctl.h>
#include <sys/types.h>

#include <string_view>
#endif  // defined(__aarch64__) && defined(__APPLE__)

#if (defined(__aarch64__) && defined(_WIN64)) || defined(_M_ARM64)
#include <Windows.h>
#include <processthreadsapi.h>
#include <sysinfoapi.h>
#include <winnt.h>
#endif  // (defined(__aarch64__) && defined(_WIN64)) || defined(_M_ARM64)

namespace kai::test {

namespace {

enum CpuFeatures {
    ADVSIMD = 0,  //
    DOTPROD,      //
    I8MM,         //
    FP16,         //
    BF16,         //
    SVE,          //
    SVE2,         //
    SME,          //
    SME2,         //
    LAST_ELEMENT  // This should be last element, please add new CPU capabilities before it
};

#if defined(__aarch64__) && defined(__linux__)
/// Define CPU capabilities not available in toolchain definitions
#ifndef HWCAP_ASIMD
constexpr uint64_t HWCAP_ASIMD = 1UL << 1;
#endif
#ifndef HWCAP_FPHP
constexpr uint64_t HWCAP_FPHP = 1UL << 9;
#endif
#ifndef HWCAP_ASIMDHP
constexpr uint64_t HWCAP_ASIMDHP = 1UL << 10;
#endif
#ifndef HWCAP_ASIMDDP
constexpr uint64_t HWCAP_ASIMDDP = 1UL << 20;
#endif
#ifndef HWCAP_SVE
constexpr uint64_t HWCAP_SVE = 1UL << 22;
#endif
#ifndef HWCAP2_SVE2
constexpr uint64_t HWCAP2_SVE2 = 1UL << 1;
#endif
#ifndef HWCAP2_I8MM
constexpr uint64_t HWCAP2_I8MM = 1UL << 13;
#endif
#ifndef HWCAP2_BF16
constexpr uint64_t HWCAP2_BF16 = 1UL << 14;
#endif
#ifndef HWCAP2_SME
constexpr uint64_t HWCAP2_SME = 1UL << 23;
#endif
#ifndef HWCAP2_SME2
constexpr uint64_t HWCAP2_SME2 = 1UL << 37;
#endif

const std::array<std::tuple<CpuFeatures, uint64_t, uint64_t>, CpuFeatures::LAST_ELEMENT> cpu_caps{{
    {CpuFeatures::ADVSIMD, AT_HWCAP, HWCAP_ASIMD},              //
    {CpuFeatures::DOTPROD, AT_HWCAP, HWCAP_ASIMDDP},            //
    {CpuFeatures::I8MM, AT_HWCAP2, HWCAP2_I8MM},                //
    {CpuFeatures::FP16, AT_HWCAP, HWCAP_FPHP | HWCAP_ASIMDHP},  //
    {CpuFeatures::BF16, AT_HWCAP2, HWCAP2_BF16},                //
    {CpuFeatures::SVE, AT_HWCAP, HWCAP_SVE},                    //
    {CpuFeatures::SVE2, AT_HWCAP2, HWCAP2_SVE2},                //
    {CpuFeatures::SME, AT_HWCAP2, HWCAP2_SME},                  //
    {CpuFeatures::SME2, AT_HWCAP2, HWCAP2_SME2},                //
}};

bool get_cap_support(CpuFeatures feature) {
    KAI_ASSERT_ALWAYS(feature < cpu_caps.size());

    auto [cpu_feature, cap_id, cap_bits] = cpu_caps[static_cast<int>(feature)];
    // Make sure CPU feature is correctly initialized
    KAI_ASSERT_ALWAYS(feature == cpu_feature);

    const uint64_t hwcaps = getauxval(cap_id);

    return (hwcaps & cap_bits) == cap_bits;
}
#elif defined(__aarch64__) && defined(__APPLE__)
const std::array<std::tuple<CpuFeatures, std::string_view>, CpuFeatures::LAST_ELEMENT> cpu_caps{{
    {CpuFeatures::ADVSIMD, "hw.optional.arm64"},  // Advanced SIMD is always present on arm64
    {CpuFeatures::DOTPROD, "hw.optional.arm.FEAT_DotProd"},
    {CpuFeatures::I8MM, "hw.optional.arm.FEAT_I8MM"},
    {CpuFeatures::FP16, "hw.optional.arm.FEAT_FP16"},
    {CpuFeatures::BF16, "hw.optional.arm.FEAT_BF16"},
    {CpuFeatures::SVE, ""},   // not supported
    {CpuFeatures::SVE2, ""},  // not supported
    {CpuFeatures::SME, "hw.optional.arm.FEAT_SME"},
    {CpuFeatures::SME2, "hw.optional.arm.FEAT_SME2"},
}};

bool get_cap_support(CpuFeatures feature) {
    KAI_ASSERT_ALWAYS(feature < CpuFeatures::LAST_ELEMENT);

    auto [cpu_feature, cap_name] = cpu_caps[static_cast<int>(feature)];
    KAI_ASSERT_ALWAYS(feature == cpu_feature);

    uint32_t value{};

    if (cap_name.length() > 0) {
        size_t size = sizeof(value);

        KAI_ASSERT_ALWAYS(sysctlbyname(cap_name.data(), nullptr, &size, nullptr, 0) == 0);
        KAI_ASSERT_ALWAYS(size == sizeof(value));

        [[maybe_unused]] int status = sysctlbyname(cap_name.data(), &value, &size, nullptr, 0);
        KAI_ASSERT_ALWAYS(status == 0);
    }

    return value == 1;
}
#elif (defined(__aarch64__) && defined(_WIN64)) || defined(_M_ARM64)
// Some system registers are provided in HARDWARE\DESCRIPTION\System\CentralProcessor\* registry.
//
// The registry name is encoded as
//   CP {op0 & 1, op1, CRn, CRm, op2}
//
// These can be used to detect architectural features that are unable to detect reliably
// using IsProcessorFeaturePresent. It must not be used to detect architectural features
// that require operating system support such as SVE and SME.
const char* ID_AA64PFR0_EL1 = "CP 4020";
const char* ID_AA64ISAR1_EL1 = "CP 4031";

const std::array<std::tuple<CpuFeatures, DWORD, const char*, uint64_t>, CpuFeatures::LAST_ELEMENT> cpu_caps{{
    {CpuFeatures::ADVSIMD, PF_ARM_NEON_INSTRUCTIONS_AVAILABLE, nullptr, 0},
    {CpuFeatures::DOTPROD, PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE, nullptr, 0},
    {CpuFeatures::I8MM, 0, ID_AA64ISAR1_EL1, 0x00f0000000000000ULL},
    {CpuFeatures::FP16, 0, ID_AA64PFR0_EL1, 0x00000000000f0000ULL},
    {CpuFeatures::BF16, 0, ID_AA64ISAR1_EL1, 0x0000f00000000000ULL},
    {CpuFeatures::SVE, 46, nullptr, 0},
    {CpuFeatures::SVE2, 47, nullptr, 0},
    {CpuFeatures::SME, 0, nullptr, 0},
    {CpuFeatures::SME2, 0, nullptr, 0},
}};

uint64_t read_sysreg(const char* name) {
    uint64_t value = 0;
    DWORD size = sizeof(value);

    const LSTATUS status = RegGetValueA(
        HKEY_LOCAL_MACHINE, "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0", name, RRF_RT_REG_QWORD, nullptr,
        &value, &size);

    KAI_ASSERT_ALWAYS(status == ERROR_SUCCESS);

    return value;
}

bool get_cap_support(CpuFeatures feature) {
    KAI_ASSERT_ALWAYS(feature < CpuFeatures::LAST_ELEMENT);
    auto [cpu_feature, cap_id, reg_name, reg_mask] = cpu_caps[static_cast<int>(feature)];

    if (cap_id != 0) {
        return IsProcessorFeaturePresent(cap_id);
    }

    if (reg_name != nullptr) {
        const uint64_t value = read_sysreg(reg_name);
        const bool is_aarch64 = IsProcessorFeaturePresent(PF_ARM_V8_INSTRUCTIONS_AVAILABLE);
        const bool has_feature = (value & reg_mask) != 0;

        return is_aarch64 && has_feature;
    }

    return false;
}
#elif defined(__aarch64__)
#error Please add a way how to check implemented CPU features
#else
bool get_cap_support(CpuFeatures feature) {
    KAI_UNUSED(feature);
    return false;
}
#endif

/// Information about the CPU that is executing the program.
struct CpuInfo {
    CpuInfo() :
        has_advsimd(get_cap_support(CpuFeatures::ADVSIMD)),
        has_dotprod(get_cap_support(CpuFeatures::DOTPROD)),
        has_i8mm(get_cap_support(CpuFeatures::I8MM)),
        has_fp16(get_cap_support(CpuFeatures::FP16)),
        has_bf16(get_cap_support(CpuFeatures::BF16)),
        has_sve(get_cap_support(CpuFeatures::SVE)),
        has_sve2(get_cap_support(CpuFeatures::SVE2)),
        has_sme(get_cap_support(CpuFeatures::SME)),
        has_sme2(get_cap_support(CpuFeatures::SME2)) {
    }

    /// Gets the singleton @ref CpuInfo object.
    static const CpuInfo& current() {
        static const CpuInfo cpu_info{};
        return cpu_info;
    }

    const bool has_advsimd{};  ///< AdvSIMD is supported.
    const bool has_dotprod{};  ///< DotProd is supported.
    const bool has_i8mm{};     ///< I8MM is supported.
    const bool has_fp16{};     ///< FP16 is supported.
    const bool has_bf16{};     ///< B16 is supported.
    const bool has_sve{};      ///< SVE is supported.
    const bool has_sve2{};     ///< SVE2 is supported.
    const bool has_sme{};      ///< SME is supported.
    const bool has_sme2{};     ///< SME2 is supported.
};

}  // namespace

/// Helper functions
bool cpu_has_advsimd() {
    return CpuInfo::current().has_advsimd;
}

bool cpu_has_dotprod() {
    return CpuInfo::current().has_dotprod;
}

bool cpu_has_dotprod_and_fp16() {
    return cpu_has_dotprod() && cpu_has_fp16();
}

bool cpu_has_i8mm() {
    return CpuInfo::current().has_i8mm;
}

bool cpu_has_i8mm_and_fp16() {
    return cpu_has_i8mm() && cpu_has_fp16();
}

bool cpu_has_fp16() {
    return CpuInfo::current().has_fp16;
}

bool cpu_has_bf16() {
    return CpuInfo::current().has_bf16;
}

bool cpu_has_sve() {
    return CpuInfo::current().has_sve;
}

bool cpu_has_sve_vl256() {
    if (CpuInfo::current().has_sve) {
        return (kai_get_sve_vector_length_u8() == 32);
    }
    return false;
}

bool cpu_has_sve2() {
    return CpuInfo::current().has_sve2;
}

bool cpu_has_sme() {
    return CpuInfo::current().has_sme;
}

bool cpu_has_sme2() {
    return CpuInfo::current().has_sme2;
}

bool cpu_has_dotprod_and_bf16() {
    return cpu_has_dotprod() && cpu_has_bf16();
}

bool cpu_has_i8mm_and_bf16() {
    return cpu_has_i8mm() && cpu_has_bf16();
}

}  // namespace kai::test
