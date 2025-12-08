//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <string_view>
#include <vector>

#include "test/common/span.hpp"
#include "test/nextgen/harness/tensor.hpp"

namespace kai::test {

/// Wrapper to provide unified API for all micro-kernels.
class KernelWrapper {
public:
    KernelWrapper() = default;                                ///< Default constructor.
    virtual ~KernelWrapper() = default;                       ///< Destructor.
    KernelWrapper(const KernelWrapper&) = delete;             ///< No copy constructor.
    KernelWrapper& operator=(const KernelWrapper&) = delete;  ///< No copy assignment.
    KernelWrapper(KernelWrapper&&) = default;                 ///< Move constructor.
    KernelWrapper& operator=(KernelWrapper&&) = default;      ///< Move assignment.

    /// Gets the micro-kernel name.
    [[nodiscard]] virtual std::string_view name() const = 0;

    /// Gets the list of input tensors required to run the micro-kernel.
    ///
    /// @param[in] tensors The data pool.
    ///
    /// @return The list of tensor IDs.
    [[nodiscard]] virtual std::vector<size_t> run_inputs(Span<const Tensor> tensors) const = 0;

    /// Gets the list of input tensors required to run the reference implementation.
    ///
    /// @param[in] tensors The data pool.
    ///
    /// @return The list of tensor IDs.
    [[nodiscard]] virtual std::vector<size_t> ref_inputs(Span<const Tensor> tensors) const = 0;

    /// Gets the scheduling steps in each dimension.
    ///
    /// @param[in] shape The full problem shape.
    /// @param[in] tensors The data pool.
    ///
    /// @return The step in each dimension.
    [[nodiscard]] virtual std::vector<size_t> steps(Span<const size_t> shape, Span<const Tensor> tensors) const = 0;

    /// Populates the data pool with constant information.
    ///
    /// @param[in, out] tensors The data pool.
    virtual void populate_constant_info(Span<Tensor> tensors) const = 0;

    /// Runs the micro-kernel to process a tile of the problem shape.
    ///
    /// @param[in] full_shape The full problem shape.
    /// @param[in] tile_coords The starting coordinate of the tile to be processed by the kernel.
    /// @param[in] tile_shape The size of the tile to be processed by the kernel.
    /// @param[in, out] tensors The data pool.
    virtual void run(
        Span<const size_t> full_shape, Span<const size_t> tile_coords, Span<const size_t> tile_shape,
        Span<Tensor> tensors) const = 0;

    /// Computes the reference data.
    ///
    /// @param[in] shape The problem shape.
    /// @param[in, out] tensors The data pool.
    virtual void compute_reference(Span<const size_t> shape, Span<Tensor> tensors) const = 0;
};

}  // namespace kai::test
