//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

namespace kai::test {

class DataFormat;
class Rect;
class MismatchHandler;

/// Compares two matrices to check whether they are matched.
///
/// @param[in] imp_data Data buffer of the actual implementation matrix.
/// @param[in] ref_data Data buffer of the reference implementation matrix.
/// @param[in] format Data format.
/// @param[in] full_height Height of the full matrix.
/// @param[in] full_width Width of the full matrix.
/// @param[in] rect Rectangular region of the matrix that is populated with data.
/// @param[in] handler Mismatch handler.
///
/// @return `true` if the two matrices are considered matched.
bool compare(
    const void* imp_data, const void* ref_data, const DataFormat& format, size_t full_height, size_t full_width,
    const Rect& rect, MismatchHandler& handler);

/// Handles mismatches found by @ref validate function.
class MismatchHandler {
public:
    /// Constructor.
    MismatchHandler() = default;

    /// Destructor.
    virtual ~MismatchHandler() = default;

    /// Copy constructor.
    MismatchHandler(const MismatchHandler&) = default;

    /// Copy assignment.
    MismatchHandler& operator=(const MismatchHandler&) = default;

    /// Move constructor.
    MismatchHandler(MismatchHandler&&) noexcept = default;

    /// Move assignment.
    MismatchHandler& operator=(MismatchHandler&&) noexcept = default;

    /// Handles new mismatch result.
    ///
    /// This method must be called even when no error is detected.
    ///
    /// @param[in] absolute_error Absolute error.
    /// @param[in] relative_error Relative error.
    ///
    /// @return `true` if the mismatch is sufficiently large to be logged as real mismatch.
    virtual bool handle_data(float absolute_error, float relative_error) = 0;

    /// Marks the result as failed.
    ///
    /// It is zero tolerance if the data point is considered impossible to have mismatch
    /// regardless of implementation method.
    /// These normally include data point outside if the portion of interest (these must be 0)
    /// and data point belongs to quantization information.
    virtual void mark_as_failed() = 0;

    /// Returns a value indicating whether the two matrices are considered matched.
    ///
    /// @param[in] num_checks Total number of data points that have been checked.
    ///
    /// @return `true` if the two matrices are considered matched.
    [[nodiscard]] virtual bool success(size_t num_checks) const = 0;
};

/// This mismatch handler considers two values being mismatched when both the relative error
/// and the absolute error exceed their respective thresholds.
///
/// This mismatch handler considers two matrices being mismatched when the number of mismatches
/// exceed both the relative and absolute thresholds.
class DefaultMismatchHandler final : public MismatchHandler {
public:
    /// Creates a new mismatch handler.
    ///
    /// @param[in] abs_error_threshold Threshold for absolute error
    /// @param[in] rel_error_threshold Threshold for relative error.
    /// @param[in] abs_mismatched_threshold Threshold for the number of mismatched data points.
    /// @param[in] rel_mismatched_threshold Threshold for the ratio of mismatched data points.
    DefaultMismatchHandler(
        float abs_error_threshold, float rel_error_threshold, size_t abs_mismatched_threshold,
        float rel_mismatched_threshold);

    /// Destructur.
    ~DefaultMismatchHandler() = default;

    /// Copy constructor.
    DefaultMismatchHandler(const DefaultMismatchHandler& rhs);

    /// Copy assignment.
    DefaultMismatchHandler& operator=(const DefaultMismatchHandler& rhs);

    /// Move constructor.
    DefaultMismatchHandler(DefaultMismatchHandler&& rhs) noexcept = default;

    /// Move assignment.
    DefaultMismatchHandler& operator=(DefaultMismatchHandler&& rhs) noexcept = default;

    bool handle_data(float absolute_error, float relative_error) override;
    void mark_as_failed() override;
    [[nodiscard]] bool success(size_t num_checks) const override;

private:
    float _abs_error_threshold;
    float _rel_error_threshold;
    size_t _abs_mismatched_threshold;
    float _rel_mismatched_threshold;

    size_t _num_mismatches;
    bool _failed;
};

}  // namespace kai::test
