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

#ifndef CKW_SRC_CL_CLKERNELWRITER_H
#define CKW_SRC_CL_CLKERNELWRITER_H

#include "ckw/KernelWriter.h"

#include <memory>
#include <set>
#include <string>
#include <utility>

namespace ckw
{

// Forward Declarations
class CLTile;
class CLTensorArgument;
class ConstantData;
class TensorOperand;
class TensorSampler;
class TileOperand;

enum class DataType;
enum class MemoryOperation;

/** OpenCL kernel writer. */
class CLKernelWriter : public KernelWriter
{
public:
    // =============================================================================================
    // Construtors and destructor
    // =============================================================================================

    /** Initialize a new instance of @ref CLKernelWriter class. */
    CLKernelWriter();

    /** Destructor */
    ~CLKernelWriter();

    // =============================================================================================
    // Data processing
    // =============================================================================================

    void op_assign(const TileOperand &dst, const TileOperand &src) override;

    void op_cast(const TileOperand &dst, const TileOperand &src, ConvertPolicy policy) override;

    void op_unary(const TileOperand &dst, UnaryOp op, const TileOperand &src) override;

    void op_binary(const TileOperand &dst, BinaryOp op, const TileOperand &first, const TileOperand &second) override;

    void op_ternary(const TileOperand &dst, TernaryOp op, const TileOperand &first, const TileOperand &second, const TileOperand &third) override;

    // =============================================================================================
    // Flow control
    // =============================================================================================

    void op_if(const TileOperand &lhs, BinaryOp op, const TileOperand &rhs, const std::function<void()> &body) override;

    void op_else_if(const TileOperand &lhs, BinaryOp op, const TileOperand &rhs, const std::function<void()> &body) override;

    void op_else(const std::function<void()> &body) override;

    void op_for_loop(
        const TileOperand &var, BinaryOp cond_op, const TileOperand &cond_value,
        const TileOperand &update_var, AssignmentOp update_op, const TileOperand &update_value,
        const std::function<void()> &body) override;

    void op_return() override;

    // =============================================================================================
    // Misc
    // =============================================================================================

    void op_comment(const std::string &text) override;

    void op_write_raw_code(const std::string &raw_code) override;

    // =============================================================================================
    // Code generation
    // =============================================================================================

    std::unique_ptr<Kernel> emit_kernel(const std::string &name) override;

    // =============================================================================================
    // Tensor and tile declaration
    // =============================================================================================

    TensorOperand declare_tensor_argument(const std::string &name, const TensorInfo &info) override;

    /** Declare a tile given name and tile information
     *
     * Similar to @ref KernelWriter::declare_tile()
     */
    TileOperand declare_tile(const std::string &name, const TileInfo &tile_info) override;

    /** Declare a constant tile given a @ref:ConstantData object
     *
     * Similar to @ref KernelWriter::declare_constant_tile()
     */
    TileOperand declare_constant_tile(const ConstantData &data) override;

    // =============================================================================================
    // Memory Operations
    // =============================================================================================

    /** Load the data from the tensor memory to the tile using the sampling information.
     *
     * Similar to @ref KernelWriter::op_load()
     */
    void op_load(
        const TileOperand &tile_op, const TensorOperand &tensor_op, TensorSampler &sampler,
        const TileOperand &x, const TileOperand &y, const TileOperand &z, const TileOperand &batch) override;

    /** Load the data from the tensor memory to the tile in a dilated way using the sampling information.
     *
     * Similar to @ref KernelWriter::op_load_dilated()
     */
    void op_load_dilated(
        const TileOperand &tile_op, const TensorOperand &tensor_op, TensorSampler &sampler,
        const TileOperand &x, const TileOperand &y, const TileOperand &z, const TileOperand &batch,
        const TileOperand &dilation_x, const TileOperand &dilation_y) override;

    /** Store the data to the tensor memory from the tile using the sampling information.
     *
     * Similar to @ref KernelWriter::op_store()
     */
    void op_store(
        const TensorOperand &tensor_op, const TileOperand &tile_op, TensorSampler &sampler,
        const TileOperand &x, const TileOperand &y, const TileOperand &z, const TileOperand &batch) override;

    /** Store the data to the tensor memory from the tile in a dilated way using the sampling information.
     *
     * Similar to @ref KernelWriter::op_store_dilated()
     */
    void op_store_dilated(
        const TensorOperand &tensor_op, const TileOperand &tile_op, TensorSampler &sampler,
        const TileOperand &x, const TileOperand &y, const TileOperand &z, const TileOperand &batch,
        const TileOperand &dilation_x, const TileOperand &dilation_y) override;

protected:
    /** Return @ref CLTile object from the @ref TileOperand object.
     *
     * This function performs appropriate check before doing type casting.
     */
    const CLTile &to_cl_tile(const TileOperand &operand) const;

    /** Append the specified code to the kernel body source code. */
    template <typename T, typename... TArgs>
    void append_code(T &&code, TArgs &&...args)
    {
        append_code(std::forward<T>(code));
        append_code(std::forward<TArgs>(args)...);
    }

    /** Append the specified code to the kernel body source code. */
    template <typename T>
    void append_code(T &&code)
    {
        _body_source_code += std::forward<T>(code);
    }

    /** Get the current kernel body source code. */
    const std::string &body_source_code() const;

    // For helper functions
private:
    /** Helper function to consolidate all load/store logic in this class */
    void op_load_store(
        MemoryOperation op, const TileOperand &tile_op, const TensorOperand &tensor_op, TensorSampler &sampler,
        const TileOperand &x, const TileOperand &y, const TileOperand &z, const TileOperand &batch,
        const CLTile &dilation_x, const CLTile &dilation_y);

    /** This function is the generic function to write both `if` and `else if` blocks.
     *
     * It is used for both @ref CLKernelWriter::op_if and @ref CLKernelWriter::op_else_if.
     *
     * @param[in] is_else True if this is an `else if` block, otherwise this is an `if` block.
     * @param[in] lhs     The LHS tile of the condition.
     * @param[in] op      The relational binary operator.
     * @param[in] rhs     The RHS tile of the condition.
     * @param[in] body    The function that writes the body of the else-if block.
     */
    void op_if_generic(bool is_else, const TileOperand &lhs, BinaryOp op, const TileOperand &rhs, const std::function<void()> &body);

    // For attributes
private:
    /** This string contains the kernel body source code, not the full CL source code.
     * The full source code will only be generated when the user calls @ref KernelWriter::emit_kernel.
     *
     * In order to add code to this, use @ref CLKernelWriter::append_code.
     * Do not attempt to concatenate and alter this string directly.
     */
    std::string _body_source_code{};

    std::set<std::unique_ptr<CLTensorArgument>> _tensors{};
    std::set<std::unique_ptr<CLTile>>           _tiles{};
    std::set<std::unique_ptr<CLTile>>           _constant_tiles{};
};

} // namespace ckw

#endif // CKW_SRC_CL_CLKERNELWRITER_H
