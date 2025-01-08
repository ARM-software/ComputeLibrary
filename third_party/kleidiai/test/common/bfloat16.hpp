//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <iosfwd>
#include <type_traits>

#include "test/common/type_traits.hpp"

namespace kai::test {

/// Half-precision brain floating-point.
///
/// This class encapsulates `bfloat16_t` data type provided by `arm_bf16.h`.
class BFloat16 {
public:
    /// Constructor.
    BFloat16() = default;

    /// Destructor.
    ~BFloat16() = default;

    /// Copy constructor.
    BFloat16(const BFloat16&) = default;

    /// Copy assignment.
    BFloat16& operator=(const BFloat16&) = default;

    /// Move constructor.
    BFloat16(BFloat16&&) = default;

    /// Move assignment.
    BFloat16& operator=(BFloat16&&) = default;

    /// Creates a new object from the specified numeric value.
    BFloat16(float value) : _data(0) {
#ifdef __ARM_FEATURE_BF16
        __asm__ __volatile__("bfcvt %h[output], %s[input]" : [output] "=w"(_data) : [input] "w"(value));
#else
        const uint32_t* value_i32 = reinterpret_cast<const uint32_t*>(&value);
        _data = (*value_i32 >> 16);
#endif
    }

    /// Assigns to the specified numeric value which will be converted to `bfloat16_t`.
    template <typename T, std::enable_if_t<is_arithmetic<T>, bool> = true>
    BFloat16& operator=(T value) {
        const auto value_f32 = static_cast<float>(value);
#ifdef __ARM_FEATURE_BF16
        __asm__ __volatile__("bfcvt %h[output], %s[input]" : [output] "=w"(_data) : [input] "w"(value_f32));
#else
        const uint32_t* value_i32 = reinterpret_cast<const uint32_t*>(&value_f32);
        _data = (*value_i32 >> 16);
#endif
        return *this;
    }

    /// Converts to floating-point.
    operator float() const {
        union {
            float f32;
            uint32_t u32;
        } data;

        data.u32 = static_cast<uint32_t>(_data) << 16;

        return data.f32;
    }

    /// Equality operator.
    bool operator==(BFloat16 rhs) const {
        return _data == rhs._data;
    }

    /// Unequality operator.
    bool operator!=(BFloat16 rhs) const {
        return _data != rhs._data;
    }

    uint16_t data() const {
        return _data;
    }

    void set_data(uint16_t data) {
        _data = data;
    }

    /// Writes the value to the output stream.
    ///
    /// @param[in] os Output stream to be written to.
    /// @param[in] value Value to be written.
    ///
    /// @return The output stream.
    friend std::ostream& operator<<(std::ostream& os, BFloat16 value);

private:
    uint16_t _data;
};

}  // namespace kai::test
