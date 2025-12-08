//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <random>
#include <type_traits>

#include "kai/kai_common.h"
#include "test/reference/fill.hpp"

namespace kai::test {

/// Base data generator class.
template <typename T>
class DataGenerator {
public:
    virtual ~DataGenerator() = default;

    Buffer operator()(size_t rows, size_t cols) const {
        return generate(rows, cols);
    };

protected:
    virtual Buffer generate(size_t rows, size_t cols) const = 0;
};

/// Fills a matrix with sequentially increasing values in row-major order.
template <typename T>
class SequentialGenerator : public DataGenerator<T> {
public:
    explicit SequentialGenerator(float start = 0.0, float step = 1.0) : m_start(start), m_step(step) {
    }

protected:
    Buffer generate(size_t rows, size_t cols) const override {
        // Emit the initial value before advancing so sequences start exactly at m_start.
        return fill_matrix_raw<T>(
            rows, cols, [value = static_cast<T>(m_start), step = static_cast<T>(m_step)](size_t, size_t) mutable -> T {
                const T current = value;
                value += step;
                return current;
            });
    }

private:
    float m_start;
    float m_step;
};

/// Produces a matrix filled with a constant value (defaults to zero).
template <typename T>
class ConstantGenerator : public SequentialGenerator<T> {
public:
    explicit ConstantGenerator(float value = 0.0) : SequentialGenerator<T>(value, 0.0) {
    }
};

/// Generates uniformly distributed floating-point values on [low, high).
template <typename T>
class UniformRandomGenerator : public DataGenerator<T> {
public:
    static_assert(is_floating_point<T>, "UniformRandomGenerator requires floating-point type");

    explicit UniformRandomGenerator(float low = 0.0, float high = 1.0, const std::uint32_t seed = 0) :
        m_engine(seed), m_dist(low, high) {
        KAI_ASSERT_ALWAYS(low < high);
    }

protected:
    Buffer generate(size_t rows, size_t cols) const override {
        return fill_matrix_raw<T>(rows, cols, [&](size_t, size_t) mutable { return static_cast<T>(m_dist(m_engine)); });
    }

private:
    mutable std::mt19937 m_engine;
    mutable std::uniform_real_distribution<float> m_dist;
};

/// Generates normally distributed floating-point values.
template <typename T>
class NormalRandomGenerator : public DataGenerator<T> {
public:
    static_assert(is_floating_point<T>, "NormalRandomGenerator requires floating-point type");

    explicit NormalRandomGenerator(float mean = 0.0, float stddev = 1.0, const std::uint32_t seed = 0) :
        m_engine(seed), m_dist(mean, stddev) {
        KAI_ASSERT_ALWAYS(stddev > 0.0);
    }

protected:
    Buffer generate(size_t rows, size_t cols) const override {
        return fill_matrix_raw<T>(rows, cols, [&](size_t, size_t) mutable { return static_cast<T>(m_dist(m_engine)); });
    }

private:
    mutable std::mt19937 m_engine;
    mutable std::normal_distribution<float> m_dist;
};

}  // namespace kai::test
