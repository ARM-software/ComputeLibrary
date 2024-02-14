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
#include "ckw/TensorSampler.h"
#include "ckw/TileInfo.h"
#include "ckw/TileOperand.h"
#include "ckw/types/ConstantData.h"
#include "ckw/types/ConvertPolicy.h"
#include "ckw/types/DataType.h"
#include "ckw/types/Operators.h"
#include "ckw/types/TargetArchitecture.h"
#include "ckw/types/TargetLanguage.h"
#include "ckw/types/TensorComponentType.h"
#include "ckw/types/TensorDataLayout.h"
#include "ckw/types/TensorSamplerTypes.h"
#include "ckw/types/TensorStorageType.h"

#include <functional>
#include <memory>
#include <string>
#include <tuple>

namespace ckw
{

/** Forward Declarations */
class TileArea;

/** A kernel writer.
 *
 * This class is used to construct a new kernel by defining arguments, declaring variable and writing code.
 *
 * Use @ref KernelWriter::create_instance method to create the kernel writer for the specific target architecture and language.
 *
 * After having finished constructing the kernel, call @ref KernelWriter::emit_kernel to get the kernel object.
 */
class KernelWriter
{
public:
    // =============================================================================================
    // Construtors and destructor
    // =============================================================================================

    /** Initialize a new instance of @ref KernelWriter class for the specific architecture and language.
     *
     * Supported target architectures and languages:
     *
     * Architecture                  | Languages                    |
     * ------------------------------|------------------------------|
     * GpuArmMaliValhall             | OpenCL                       |
     *
     * @param[in] architecture The architecture on which the kernel is executed.
     * @param[in] language     The language to write the kernel.
     */
    static std::unique_ptr<KernelWriter> create_instance(TargetArchitecture architecture, TargetLanguage language);

    /** Destructor */
    virtual ~KernelWriter();

    // =============================================================================================
    // Data processing
    // =============================================================================================

    /** Write assignment statement: `<dst> = <src>;`.
     *
     * @param[in] dst The destination tile.
     * @param[in] src The source tile.
     */
    virtual void op_assign(const TileOperand &dst, const TileOperand &src) = 0;

    /** Write the cast statement: `<dst> = convert_<dst.type><policy>(<src>);`.
     *
     * @param[in] dst    The destination tile.
     * @param[in] src    The source tile.
     * @param[in] policy The policy governing the behavior of the cast.
     */
    virtual void op_cast(const TileOperand &dst, const TileOperand &src, ConvertPolicy policy) = 0;

    /** Write the unary expression statement: `<dst> = <op> <src>;`.
     *
     * @param[in] dst The destination tile.
     * @param[in] op  The unary operator.
     * @param[in] src The source tile.
     */
    virtual void op_unary(const TileOperand &dst, UnaryOp op, const TileOperand &src) = 0;

    /** Write the binary expression statement: `<dst> = <op>(<first>, <second>);`.
     *
     * @param[in] dst    The destination tile.
     * @param[in] op     The binary operator.
     * @param[in] first  The first source tile.
     * @param[in] second The second source tile.
     */
    virtual void
    op_binary(const TileOperand &dst, BinaryOp op, const TileOperand &first, const TileOperand &second) = 0;

    /** Write ternary expression statement: `<dst> = <op>(<first>, <second>, <third>);`.
     *
     * @param[in] dst    The destination tile.
     * @param[in] op     The ternary operator.
     * @param[in] first  The first source tile.
     * @param[in] second The second source tile.
     * @param[in] third  The third source tile.
     */
    virtual void op_ternary(const TileOperand &dst,
                            TernaryOp          op,
                            const TileOperand &first,
                            const TileOperand &second,
                            const TileOperand &third) = 0;

    // =============================================================================================
    // Flow control
    // =============================================================================================

    /** Write if block: `if(<lhs> <op> <rhs>) { <body> }`.
     *
     * @param[in] lhs  The LHS tile of the condition.
     * @param[in] op   The relational binary operator.
     * @param[in] rhs  The RHS tile of the condition.
     * @param[in] body The function that writes the body of the if block.
     */
    virtual void
    op_if(const TileOperand &lhs, BinaryOp op, const TileOperand &rhs, const std::function<void()> &body) = 0;

    /** Write else-if block: `else if(<lhs> <op> <rhs>) { <body> }`.
     *
     * @param[in] lhs  The LHS tile of the condition.
     * @param[in] op   The relational binary operator.
     * @param[in] rhs  The RHS tile of the condition.
     * @param[in] body The function that writes the body of the else-if block.
     */
    virtual void
    op_else_if(const TileOperand &lhs, BinaryOp op, const TileOperand &rhs, const std::function<void()> &body) = 0;

    /** Write an else block: `else { <body> }`.
     *
     * @param[in] body The function that writes the body of the else block.
     */
    virtual void op_else(const std::function<void()> &body) = 0;

    /** Write for-loop block: `for(; <var> <cond_op> <cond_value>; <update_var> <update_op> <update_value>) { body }`.
     *
     * @param[in] var          The scalar tile used in loop condition.
     * @param[in] cond_op      The relational binary operator used in loop condition.
     * @param[in] cond_value   The value which the variable is compared against.
     * @param[in] update_var   The scalar tile which is updated each iteration.
     * @param[in] update_op    The assignment operator used for updating the update value.
     * @param[in] update_value The value which is updated at every iteration.
     * @param[in] body         The function that writes the body of the for-loop block.
     */
    virtual void op_for_loop(const TileOperand           &var,
                             BinaryOp                     cond_op,
                             const TileOperand           &cond_value,
                             const TileOperand           &update_var,
                             AssignmentOp                 update_op,
                             const TileOperand           &update_value,
                             const std::function<void()> &body) = 0;

    /** Write the return statement. */
    virtual void op_return() = 0;

    // =============================================================================================
    // Misc
    // =============================================================================================

    /** Write the statement to get the global ID of the specified dimension.
     *
     * @param[in] dst The tile to write the global ID into.
     * @param[in] dim The dimension.
     */
    virtual void op_get_global_id(const TileOperand &dst, int32_t dim) = 0;

    /** Write the line comment in debug build.
     *
     * This function does not take effect on release build.
     *
     * The comment must only contain one line (i.e. no newline character is allowed).
     *
     * @param[in] text The comment to be written.
     */
    virtual void op_comment(const std::string &text) = 0;

    /** Write the statement to print out the value of all the specified tiles.
     *
     * The printing statement is constructed so that the prefix and each of the operand are printed in separate lines.
     * The format for each operand varies depending on whether it is a 2D tile, a vector or a scalar value.
     *
     * Example output of the printing statement when it is executed:
     *
     * prefix
     * scalar_name = scalar_value
     * vector_name = [vector_value_0, vector_value_1, vector_value_2]
     * tile_name = [[tile_value_00, tile_value_01], [tile_value_10, tile_value_11]]
     *
     * @param[in] prefix   The first string to be printed out before the list of operands.
     * @param[in] operands The list of tiles to be included in the printing statement.
     */
    virtual void op_print(const std::string &prefix, const std::vector<TileOperand> &operands) = 0;

    /** Write the given raw code to kernel source code
     *  It's used to address the cases where the user needs to
     *  explicitly add a code where it's not (yet) supported by
     *  the kernel writer utility calls.
     *
     * @param[in] raw_code raw code to write as string
    */
    virtual void op_write_raw_code(const std::string &raw_code) = 0;

    // =============================================================================================
    // Code generation
    // =============================================================================================

    /** Emit the kernel object.
     *
     * @param[in] name The name of the kernel object to be generated.
     */
    virtual std::unique_ptr<Kernel> emit_kernel(const std::string &name) = 0;

    // =============================================================================================
    // Tensor and tile declaration
    // =============================================================================================

    /** Declare a tensor argument.
     *
     * @param[in] name         The name of the tensor.
     * @param[in] info         The tensor info.
     *
     * @return The @ref TensorOperand object.
     */
    virtual TensorOperand declare_tensor_argument(const std::string &name, const TensorInfo &info) = 0;

    /** Declare a tile given its name and tile info
     *
     * @param[in] name Name of the tile
     * @param[in] tile_info Shape and data type of the tile
     *
     * @return The created tile operand
     */
    virtual TileOperand declare_tile(const std::string &name, const TileInfo &tile_info) = 0;

    /** Declare a constant tile given a @ref:ConstantData object
     *
     * @param[in] data a @ref ckw::ConstantData object that has the values and the
     *                 underlying data type of the constant tile
     *
     * @return The created constant tile operand
     */
    virtual TileOperand declare_constant_tile(const ConstantData &data) = 0;

    /** Load the data from the tensor memory to the tile using the sampling information.
     *
     * @param[in] tile_op   The tile to be loaded.
     * @param[in] tensor_op The tensor to be read.
     * @param[in] sampler   The tensor sampling information.
     * @param[in] x         x-coordinate
     * @param[in] y         y-coordinate
     * @param[in] z         z-coordinate
     * @param[in] batch     batch
     */
    virtual void op_load(const TileOperand   &tile_op,
                         const TensorOperand &tensor_op,
                         TensorSampler       &sampler,
                         const TileOperand   &x,
                         const TileOperand   &y,
                         const TileOperand   &z,
                         const TileOperand   &batch) = 0;

    /** Load the data from the tensor memory to the tile in a dilated way using the sampling information.
     *
     * Similar to @ref KernelWriter::op_load() and
     *
     * @param[in] dilation_x Dilation while reading in x-dimension
     * @param[in] dilation_y Dilation while reading in y-dimension
     */
    virtual void op_load_dilated(const TileOperand   &tile_op,
                                 const TensorOperand &tensor_op,
                                 TensorSampler       &sampler,
                                 const TileOperand   &x,
                                 const TileOperand   &y,
                                 const TileOperand   &z,
                                 const TileOperand   &batch,
                                 const TileOperand   &dilation_x,
                                 const TileOperand   &dilation_y) = 0;

    /** Store the data to the tensor memory from the tile using the sampling information.
     *
     * Similar to @ref KernelWriter::op_load()
     */
    virtual void op_store(const TensorOperand &tensor_op,
                          const TileOperand   &tile_op,
                          TensorSampler       &sampler,
                          const TileOperand   &x,
                          const TileOperand   &y,
                          const TileOperand   &z,
                          const TileOperand   &batch) = 0;

    /** Store the data to the tensor memory from the tile in a dilated way using the sampling information.
     *
     * Similar to @ref KernelWriter::op_load_dilated()
     */
    virtual void op_store_dilated(const TensorOperand &tensor_op,
                                  const TileOperand   &tile_op,
                                  TensorSampler       &sampler,
                                  const TileOperand   &x,
                                  const TileOperand   &y,
                                  const TileOperand   &z,
                                  const TileOperand   &batch,
                                  const TileOperand   &dilation_x,
                                  const TileOperand   &dilation_y) = 0;

    /** Load the data from the tensor memory to the tile using the indirect buffer approach and respecting the sampling information.
     *
     * @param[in] tile_op   The tile to be loaded.
     * @param[in] tensor_op The tensor to be read.
     * @param[in] sampler   The tensor sampling information.
     * @param[in] x         x-coordinate
     * @param[in] y         y-coordinate
     * @param[in] z         z-coordinate
     * @param[in] batch     batch
     */
    virtual void op_load_indirect(const TileOperand   &tile_op,
                                  const TensorOperand &tensor_op,
                                  TensorSampler       &sampler,
                                  const TileOperand   &x,
                                  const TileOperand   &y,
                                  const TileOperand   &z,
                                  const TileOperand   &batch_op) = 0;

    // =============================================================================================
    // ID space management
    // =============================================================================================

    /** Create the new unique ID space and return the value.
     *
     * This function changes the ID space to a new number which hasn't been used since the creation
     * of this kernel writer object.
     *
     * @return The new ID space value.
     */
    int32_t new_id_space();

    /** Get the current ID space. */
    int32_t id_space() const;

protected:
    /** Set the current ID space.
     *
     * @param[in] value The ID space to be used.
     */
    KernelWriter &id_space(int32_t value);

    /** Write the body code using the specified function.
     *
     * This function makes sure that a new ID space is created before and then is used solely
     * by the specified body writing function.
     * The ID space will not be reused after that.
     *
     * @param[in] body The function that writes the body code.
     */
    void write_body(const std::function<void()> &body);

protected:
    /** Generate full variable name by prefixing it with id space */
    std::string generate_full_name(const std::string &name) const;

    /** Create a new tile operand referring to the specified tile object. */
    static TileOperand create_tile_operand(ITile &tile);

    /** Get the reference to the tile object and the active area from the tile operand. */
    static std::tuple<ITile &, TileArea> get_tile(const TileOperand &operand);

    /** Create a new tensor operand from a tensor object. */
    static TensorOperand create_tensor_operand(ITensor &tensor);

    /** Get the reference to tensor object from the tensor operand. */
    static ITensor &get_tensor(const TensorOperand &operand);

    /** Get the values of a constant data object. */
    static const std::vector<std::vector<std::string>> &get_values(const ConstantData &data);

    /** Get the data type of a constant data object. */
    static DataType get_data_type(const ConstantData &data);

private:
    int32_t _id_space{0};
    int32_t _last_created_id_space{0};
};

} // namespace ckw

#endif // CKW_INCLUDE_CKW_KERNELWRITER_H
