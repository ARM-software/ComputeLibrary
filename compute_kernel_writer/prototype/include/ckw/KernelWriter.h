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

#ifndef CKW_PROTOTYPE_INCLUDE_CKW_KERNELWRITER_H
#define CKW_PROTOTYPE_INCLUDE_CKW_KERNELWRITER_H

#include "ckw/Kernel.h"
#include "ckw/TensorInfo.h"
#include "ckw/TensorOperand.h"
#include "ckw/TileInfo.h"
#include "ckw/TileOperand.h"
#include "ckw/types/ConvertPolicy.h"
#include "ckw/types/Functions.h"
#include "ckw/types/Operators.h"

#include <memory>

namespace ckw
{

namespace prototype
{
struct GpuKernelWriterAttribute;

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

    /** Declare a tensor argument.
     *
     * @param[in] name         The name of the tensor.
     * @param[in] info         The tensor info.
     * @param[in] storage_type The tensor storage type.
     *
     * @return The @ref TensorOperand object.
     */
    TensorOperand &declare_tensor_argument(const std::string &name, const TensorInfo &info, TensorStorageType storage_type = TensorStorageType::BufferUint8Ptr);

    /** Declare a compile-time constant scalar argument.
     *
     * @param[in] name  The name of the tile.
     * @param[in] value The value of the tile.
     *
     * @return The @ref TileOperand object.
     */
    TileOperand &declare_tile_argument(const std::string &name, int32_t value);

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
    TileOperand &declare_tile(const std::string &name, TArgs &&...args)
    {
        const auto var_name = generate_variable_name(name);
        auto       operand  = std::make_unique<TileOperand>(var_name, ::std::forward<TArgs>(args)...);

        return declare_tile_operand(std::move(operand));
    }

    // =============================================================================================
    // Load and store
    // =============================================================================================

    /** Load the data from the tensor memory to the tile using the sampling information.
     *
     * @param[out] tile       The tile to be loaded.
     * @param[in]  tensor     The tensor to be read.
     * @param[in]  sampler    The tensor sampling information.
     * @param[in]  dilation_y Dilation in the Y dimension.
     */
    void op_load(TileOperand &tile, const TensorOperand &tensor, const TensorTileSampler &sampler, const TileOperand &dilation_y = TileOperand("dil_y", 1));

    /** Load the data from the tensor memory to the tile using the indirect buffer approach and respective of the sampling information.
     *
     * @param[out] tile    The tile to be loaded.
     * @param[in]  tensor  The tensor to be read.
     * @param[in]  sampler The tensor sampling information.
     */
    void op_load_indirect(TileOperand &tile, const TensorOperand &tensor, const TensorTileSampler &sampler);

    /** Construct an indirection buffer in @p tile containing the precalculated addresses of elements in the source tensor.
     *
     * @param[out] tile    The tile to be loaded.
     * @param[in]  tensor  The tensor the be read.
     * @param[in]  sampler The tensor sampling information.
     * @param[in]  x       The X coordinate.
     * @param[in]  y       The Y coordinate.
     * @param[in]  x_off   Offset in the X dimension.
     * @param[in]  y_off   Offset in the Y dimension.
     */
    void util_get_indirect_buffer(TileOperand             &tile,
                                  const TensorOperand     &tensor,
                                  const TensorTileSampler &sampler,
                                  const TileOperand       &x,
                                  const TileOperand       &y,
                                  const TileOperand       &x_off,
                                  const TileOperand       &y_off);

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

    /** Write assignment: `<dst> = <src>;`.
     *
     * @param[out] dst The destination tile.
     * @param[in]  src The source tile.
     */
    void op_assign(const TileOperand &dst, const TileOperand &src);

    /** Write the cast: `<dst> = convert_<dst.type><_sat>(<src>);`.
     *
     * @param[out] dst      The destination tile.
     * @param[in]  src      The source tile.
     * @param[in]  policy   The policy governing the behavior of the cast.
     */
    void op_cast_expression(const TileOperand &dst, const TileOperand &src, ConvertPolicy policy);

    /** Write the unary expression: `<dst> = <op> <src>`.
     *
     * @param[out]  dst The destination tile.
     * @param[in]   op  The unary operator.
     * @param[in]   src The source tile.
     */
    void op_unary_expression(const TileOperand &dst, UnaryOp op, const TileOperand &src);

    /** Write binary expression: `<dst> = <lhs> <op> <rhs>;`.
     *
     * @param[out] dst  The destination tile.
     * @param[in]  lhs  The LHS tile.
     * @param[in]  op   The binary operator.
     * @param[in]  rhs  The RHS tile.
     */
    void op_binary_expression(const TileOperand &dst, const TileOperand &lhs, BinaryOp op, const TileOperand &rhs);

    /** Write function applied to scalar value: `<dst> = <func>(<src>);`.
     *
     * @param[out] dst  The destination tile.
     * @param[in]  func The function to be applied to the source tile.
     * @param[in]  src  The source tile.
     */
    void op_unary_elementwise_function(const TileOperand &dst, UnaryFunction func, const TileOperand &src);

    /** Write function applied to scalar value: `<dst> = <func>(<first>, <second>);`.
     *
     * @param[out] dst      The destination tile.
     * @param[in]  func     The function to be applied to the source tiles.
     * @param[in]  first    The first argument tile.
     * @param[in]  second   The second argument tile.
     */
    void op_binary_elementwise_function(const TileOperand &dst, BinaryFunction func, const TileOperand &first, const TileOperand &second);

    /** Write function applied to scalar value: `<dst> = <func>(<first>, <second>, <third>);`.
     *
     * @param[out] dst      The destination tile.
     * @param[in]  func     The function to be applied to the source tiles.
     * @param[in]  first    The first argument tile.
     * @param[in]  second   The second argument tile.
     * @param[in]  third    The third argument tile.
     */
    void op_ternary_elementwise_function(const TileOperand &dst, TernaryFunction func, const TileOperand &first, const TileOperand &second, const TileOperand &third);

    /** Write if-statement: `if(<lhs> <op> <rhs>) { <body> }`.
     *
     * @param[in] lhs   The LHS tile of the condition.
     * @param[in] op    The relational binary operator.
     * @param[in] rhs   The RHS tile of the condition.
     * @param[in] body  The body of the if-statement.
     */
    void op_if(const TileOperand &lhs, BinaryOp op, const TileOperand &rhs, const std::function<void()> &body);

    /** Write else-if-statement: `else if(<lhs> <op> <rhs>) { <body> }`.
     *
     * @param[in] lhs   The LHS tile of the condition.
     * @param[in] op    The relational binary operator.
     * @param[in] rhs   The RHS tile of the condition.
     * @param[in] body  The body of the else-if-statement.
     */
    void op_else_if(const TileOperand &lhs, BinaryOp op, const TileOperand &rhs, const std::function<void()> &body);

    /** Write an else-statement: `else { <body> }`.
     *
     * @param[in] body The body of the else-statement.
     */
    void op_else(const std::function<void()> &body);

    /** Write for-loops: `for(; <var> <cond_op> <cond_value>; <var> <update_op> <update_value>) { body }`.
     *
     * @param[in]       var_name          The name of the variable used in condition.
     * @param[in]       cond_op           The relational binary operator used in condition.
     * @param[in]       cond_value_name   The value which the variable is compared against.
     * @param[in]       update_var_name   The name of the variable which is updated.
     * @param[in]       update_op         The assignment operator used for updating the update value.
     * @param[in, out]  update_value      The value which is updated at every iteration.
     * @param[in]       body              The body of the for-loop.
     */
    void op_for_loop(const TileOperand &var_name, BinaryOp cond_op, const TileOperand &cond_value_name, const TileOperand &update_var_name, AssignmentOp update_op, const TileOperand &update_value_name, const std::function<void()> &body);

    /** Write the return statement: `return;`
     */
    void op_return();

    // =============================================================================================
    // Misc
    // =============================================================================================

    /** Set `dst` the global ID of dimension `dim`.
     *
     * @param[out] dst The tile to be written to.
     * @param[in]  dim The global ID dimension.
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
    ::std::string generate_variable_name(const std::string &name) const;

    /** Declare the tile operand.
     *
     * @param[in] operand   The tile operand to be declared.
     */
    TileOperand &declare_tile_operand(std::unique_ptr<TileOperand> operand);

private:
    Kernel                                                *_kernel;
    ::std::unique_ptr<prototype::GpuKernelWriterAttribute> _impl_attr;
    ::std::unique_ptr<prototype::IGpuKernelWriter>         _impl;

    int32_t _id_space{ 0 };
    int32_t _max_id_space{ 0 };
};

} // namespace ckw

#endif // CKW_PROTOTYPE_INCLUDE_CKW_KERNELWRITER_H
