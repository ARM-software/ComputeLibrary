/*
 * Copyright (c) 2023 Arm Limited.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef CKW_PROTOTYPE_INCLUDE_CKW_TENSOROPERAND_H
#define CKW_PROTOTYPE_INCLUDE_CKW_TENSOROPERAND_H

#include "ckw/OperandBase.h"
#include "ckw/TensorInfo.h"
#include "ckw/TensorTileSampler.h"
#include "ckw/TileOperand.h"
#include "ckw/types/DataType.h"

#include <memory>

namespace ckw
{

class TensorComponentOperand;

// =================================================================================================
// TensorOperand
// =================================================================================================

/** Tensor operand */
class TensorOperand : public OperandBase
{
public:
    /** Initialize a new instance of @ref TensorOperand class.
     *
     * @param[in] name         The name of the tensor.
     * @param[in] info         The tensor info.
     * @param[in] storage_type The tensor storage type.
     */
    TensorOperand(const ::std::string &name, const TensorInfo &info, TensorStorageType storage_type);

    /** No copy constructor. */
    TensorOperand(const TensorOperand &other) = delete;

    /** No copy assignment. */
    TensorOperand &operator=(const TensorOperand &other) = delete;

    /** (Internal use only) Create the implementation operand.
     *
     * @param[in] writer The implementation kernel writer.
     */
    virtual prototype::Operand create_impl_operand(prototype::IGpuKernelWriter *writer) const override;

    /** Get the tensor info. */
    const TensorInfo &info() const;

    /** Get the tensor info. */
    TensorInfo &info();

    /** Get the tensor storage type. */
    TensorStorageType storage_type() const;

    /** Get the data type. */
    virtual DataType data_type() const override;

    /** Get whether the tensor is compile-time constant. */
    virtual bool is_constant() const override;

    /** Get the default tile attached to the tensor. */
    const TileOperand &tile() const;

    /** Get the default tile attached to the tensor. */
    TileOperand &tile();

    /** Set the default tile attached to the tensor. */
    TensorOperand &tile(TileOperand &tile);

    /** Get the tensor sampler of the default tile. */
    const TensorTileSampler &tile_sampler() const;

    /** Get the tensor sampler of the default tile. */
    TensorTileSampler &tile_sampler();

    /** Set the tensor sampler of the default tile. */
    TensorOperand &tile_sampler(const TensorTileSampler &value);

    /** Get the operand that contains the stride in y dimension of the tensor. */
    TensorComponentOperand &stride1();

    /** Get the operand that contains the stride in z dimension of the tensor. */
    TensorComponentOperand &stride2();

    /** Get the operand that contains the stride in w dimension of the tensor. */
    TensorComponentOperand &stride3();

    /** Get the operand that contains the stride in w dimension of the tensor. */
    TensorComponentOperand &stride4();

    /** Get the operand that contains the size of dimension 0 of the tensor. */
    TensorComponentOperand &dim0();

    /** Get the operand that contains the size of dimension 1 of the tensor. */
    TensorComponentOperand &dim1();

    /** Get the operand that contains the size of dimension 2 of the tensor. */
    TensorComponentOperand &dim2();

    /** Get the operand that contains the size of dimension 3 of the tensor. */
    TensorComponentOperand &dim3();

    /** Get the operand that contains the size of dimension 4 of the tensor. */
    TensorComponentOperand &dim4();

    /** Get the operand that contains the size of dimensions 1 and 2 collapsed. */
    TensorComponentOperand &dim1_dim2();

    /** Get the operand that contains the size of dimensions 1, 2 and 3 collapsed. */
    TensorComponentOperand &dim1_dim2_dim3();

    /** Get the operand that contains the offset in bytes to the first element. */
    TensorComponentOperand &offset_first_element_in_bytes();

private:
    TensorInfo        _info;
    TensorStorageType _storage_type;

    TileOperand      *_tile{nullptr};
    TensorTileSampler _tile_sampler{};

    ::std::unique_ptr<TensorComponentOperand> _stride1{nullptr};
    ::std::unique_ptr<TensorComponentOperand> _stride2{nullptr};
    ::std::unique_ptr<TensorComponentOperand> _stride3{nullptr};
    ::std::unique_ptr<TensorComponentOperand> _stride4{nullptr};
    ::std::unique_ptr<TensorComponentOperand> _dim0{nullptr};
    ::std::unique_ptr<TensorComponentOperand> _dim1{nullptr};
    ::std::unique_ptr<TensorComponentOperand> _dim2{nullptr};
    ::std::unique_ptr<TensorComponentOperand> _dim3{nullptr};
    ::std::unique_ptr<TensorComponentOperand> _dim4{nullptr};
    ::std::unique_ptr<TensorComponentOperand> _dim1_dim2{nullptr};
    ::std::unique_ptr<TensorComponentOperand> _dim1_dim2_dim3{nullptr};
    ::std::unique_ptr<TensorComponentOperand> _offset_first_element_in_bytes{nullptr};
};

// =================================================================================================
// TensorComponentOperand
// =================================================================================================

/** Tile operand that contains tensor information. */
class TensorComponentOperand : public TileOperand
{
public:
    /** Initialize a new instance of @ref TensorComponentOperand class.
     *
     * @param[in] tensor    The tensor operand.
     * @param[in] component The tensor info component.
     */
    TensorComponentOperand(TensorOperand &tensor, TensorComponentType component);

    /** Get the tensor operand. */
    TensorOperand &tensor();

    /** Get the tensor operand. */
    const TensorOperand &tensor() const;

    /** Get the tensor component. */
    TensorComponentType component_type() const;

    /** (Internal use only) Create the implementation operand.
     *
     * @param[in] writer The implementation kernel writer.
     */
    virtual prototype::Operand create_impl_operand(prototype::IGpuKernelWriter *writer) const override;

private:
    TensorOperand      &_tensor;
    TensorComponentType _component;
};

} // namespace ckw

#endif // CKW_PROTOTYPE_INCLUDE_CKW_TENSOROPERAND_H
