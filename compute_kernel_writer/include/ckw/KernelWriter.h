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

#ifndef CKW_INCLUDE_CKW_KERNELWRITER_H
#define CKW_INCLUDE_CKW_KERNELWRITER_H

#include "ckw/Kernel.h"
#include "ckw/TensorInfo.h"
#include "ckw/TensorOperand.h"
#include "ckw/TileInfo.h"
#include "ckw/TileOperand.h"

#include <memory>

namespace ckw
{

namespace prototype
{
class GpuKernelWriterAttribute;
class IGpuKernelWriter;
} // namespace prototype

/** Kernel writer. */
class KernelWriter
{
public:
    // =============================================================================================
    // Constructors and destructor
    // =============================================================================================

    /** Initialize a new instance of kernel writer.
     *
     * @param[in] kernel The kernel to be written to.
     */
    explicit KernelWriter(Kernel &kernel);

    /** Destructor */
    ~KernelWriter();

    /** No copy constructor. */
    KernelWriter(const KernelWriter &) = delete;

    /** No copy assignment. */
    KernelWriter &operator=(const KernelWriter &) = delete;

    // =============================================================================================
    // Scope management
    // =============================================================================================

    /** Get the current ID space. */
    int32_t id_space() const;

    /** Set the current ID space. */
    KernelWriter &id_space(int32_t id_space);

    /** Switch to and return a new ID space. */
    int32_t next_id_space();

    // =============================================================================================
    // Tensor and tile declaration
    // =============================================================================================

    /** Define a tensor argument.
     *
     * @param[in] name The name of the tensor.
     * @param[in] info The tensor info.
     *
     * @return The @ref TensorOperand object.
     */
    TensorOperand &create_tensor_argument(const char *name, const TensorInfo &info);

    /** Define a compile-time constant scalar argument.
     *
     * @param[in] name  The name of the tile.
     * @param[in] value The value of the tile.
     *
     * @return The @ref TileOperand object.
     */
    TileOperand &create_tile_argument(const char *name, int32_t value);

    /** Declare a new tile.
     *
     * The name of the tile must be unique in the current ID space.
     *
     * @param[in] name The name of the tile.
     * @param[in] ...  The necessary arguments to create a new @ref TileOperand.
     *
     * @return The @ref TileOperand object.
     */
    template <typename... TArgs>
    TileOperand &declare_tile(const char *name, TArgs &&...args)
    {
        const auto var_name = generate_variable_name(name);
        auto       operand  = new TileOperand(var_name, ::std::forward<TArgs>(args)...);
        register_operand(operand, true);

        return *operand;
    }

    // =============================================================================================
    // Load and store
    // =============================================================================================

    /** Load the data from the tensor memory to the tile using the sampling information.
     *
     * @param[out] tile    The tile to be loaded.
     * @param[in]  tensor  The tensor to be read.
     * @param[in]  sampler The tensor sampling information.
     */
    void op_load(TileOperand &tile, TensorOperand &tensor, const TensorTileSampler &sampler);

    /** Store the tile to the tensor using the specified sampling information.
     *
     * @param[out] dst     The tensor that the tile is written to.
     * @param[in]  src     The tile to be stored.
     * @param[in]  sampler The tensor sampling information.
     */
    void op_store(TensorOperand &tensor, const TileOperand &tile, const TensorTileSampler &sampler);

    // =============================================================================================
    // Data processing
    // =============================================================================================

    /** Write assignment: `<dst> = <src>`.
     *
     * @param[in] dst The destination tile.
     * @param[in] src The source tile.
     */
    void op_assign(TileOperand &dst, const TileOperand &src);

    /** Write binary expression: `<dst> = <lhs> <op> <rhs>`.
     *
     * @param[in] dst The destination tile.
     * @param[in] lhs The LHS operand.
     * @param[in] rhs The RHS operand.
     * @param[in] op  The binary operator.
     */
    void op_binary_expression(TileOperand &dst, const TileOperand &lhs, const TileOperand &rhs, BinaryOp op);

    /** Write function applied to scalar value: `<dst> = <func>(<src>)`.
     *
     * @param[in] dst  The destination tile.
     * @param[in] src  The source tile.
     * @param[in] func The function to be applied to the source tile.
     */
    void op_scalar_function(TileOperand &dst, const TileOperand &src, ScalarUnaryFunction func);

    // =============================================================================================
    // Misc
    // =============================================================================================

    /** Set `dst` the global ID of dimension `dim`.
     *
     * @param[in] dst The tile to be written to.
     * @param[in] dim The global ID dimension.
     */
    void op_get_global_id(TileOperand &dst, int32_t dim);

    // =============================================================================================
    // Code generation
    // =============================================================================================

    /** Generate the source code of the kernel. */
    ::std::string generate_code();

private:
    /** Generate the full variable name based on the original name and the ID space.
     *
     * @param[in] name The name of the variable.
     *
     * @return The full variable name.
     */
    ::std::string generate_variable_name(const char *name) const;

    /** Register the operand to the kernel.
     *
     * The operand is uniquely owned by the kernel afterward.
     *
     * @param[in] operand   The operand to be registered.
     * @param[in] declaring Whether the tile declaration is generated.
     */
    void register_operand(OperandBase *operand, bool declaring);

private:
    Kernel                                                *_kernel;
    ::std::unique_ptr<prototype::GpuKernelWriterAttribute> _impl_attr;
    ::std::unique_ptr<prototype::IGpuKernelWriter>         _impl;

    int32_t _id_space{ 0 };
    int32_t _max_id_space{ 0 };
};

} // namespace ckw

#endif // CKW_INCLUDE_CKW_KERNELWRITER_H
