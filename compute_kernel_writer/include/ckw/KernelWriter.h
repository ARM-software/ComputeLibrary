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

#include "ckw/TensorOperand.h"
#include "ckw/TileOperand.h"
#include "ckw/types/ConvertPolicy.h"
#include "ckw/types/Operators.h"

#include <functional>
#include <memory>
#include <string>

namespace ckw
{

class Kernel;

/** Forward Declerations */
class TensorInfo;
class TensorSampler;
class TileInfo;

enum class TargetArchitecture;
enum class TargetLanguage;

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
    virtual void op_binary(const TileOperand &dst, BinaryOp op, const TileOperand &first, const TileOperand &second) = 0;

    /** Write ternary expression statement: `<dst> = <op>(<first>, <second>, <third>);`.
     *
     * @param[in] dst    The destination tile.
     * @param[in] op     The ternary operator.
     * @param[in] first  The first source tile.
     * @param[in] second The second source tile.
     * @param[in] third  The third source tile.
     */
    virtual void op_ternary(const TileOperand &dst, TernaryOp op, const TileOperand &first, const TileOperand &second, const TileOperand &third) = 0;

    // =============================================================================================
    // Misc
    // =============================================================================================

    /** Write the line comment in debug build.
     *
     * This function does not take effect on release build.
     *
     * The comment must only contain one line (i.e. no newline character is allowed).
     *
     * @param[in] text The comment to be written.
     */
    virtual void op_comment(const std::string &text) = 0;

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
     * @returns The created tile operand
     */
    virtual TileOperand declare_tile(const std::string &name, const TileInfo &tile_info) = 0;

    /** Load the data from the tensor memory to the tile using the sampling information.
     *
     * @param[in] tile_op   The tile to be loaded.
     * @param[in] tensor_op The tensor to be read.
     * @param[in] sampler   The tensor sampling information.
     * @param[in] x         x-coordinate
     * @param[in] y         y-coordinate
     * @param[in] z         z-coordinate
     * @param[in] batch     batch offset
     */
    virtual void op_load(
        const TileOperand &tile_op, const TensorOperand &tensor_op, TensorSampler &sampler,
        const TileOperand &x, const TileOperand &y, const TileOperand &z, const TileOperand &batch) = 0;

    /** Load the data from the tensor memory to the tile in a dilated way using the sampling information.
     *
     * Similar to @ref KernelWriter::op_load() and
     *
     * @param[in] dilation_x Dilation while reading in x-dimension
     * @param[in] dilation_y Dilation while reading in y-dimension
     */
    virtual void op_load_dilated(
        const TileOperand &tile_op, const TensorOperand &tensor_op, TensorSampler &sampler,
        const TileOperand &x, const TileOperand &y, const TileOperand &z, const TileOperand &batch,
        const TileOperand &dilation_x, const TileOperand &dilation_y) = 0;

    /** Store the data to the tensor memory from the tile using the sampling information.
     *
     * Similar to @ref KernelWriter::op_load()
     */
    virtual void op_store(
        const TensorOperand &tensor_op, const TileOperand &tile_op, TensorSampler &sampler,
        const TileOperand &x, const TileOperand &y, const TileOperand &z, const TileOperand &batch) = 0;

    /** Store the data to the tensor memory from the tile in a dilated way using the sampling information.
     *
     * Similar to @ref KernelWriter::op_load_dilated()
     */
    virtual void op_store_dilated(
        const TensorOperand &tensor_op, const TileOperand &tile_op, TensorSampler &sampler,
        const TileOperand &x, const TileOperand &y, const TileOperand &z, const TileOperand &batch,
        const TileOperand &dilation_x, const TileOperand &dilation_y) = 0;

protected:
    int32_t id_space() const;

    /** Generate full variable name by prefixing it with id space */
    std::string generate_full_name(const std::string &name) const;

    /** Create a new tile operand referring to the specified tile object. */
    static TileOperand create_tile_operand(ITile &tile);

    /** Get the reference to tile object from the tile operand. */
    static ITile &get_tile(const TileOperand &operand);

    /** Create a new tensor operand from a tensor object. */
    static TensorOperand create_tensor_operand(ITensor &tensor);

    /** Get the reference to tensor object from the tensor operand. */
    static ITensor &get_tensor(const TensorOperand &operand);

private:
    int32_t _id_space{ 0 };
};

} // namespace ckw

#endif // CKW_INCLUDE_CKW_KERNELWRITER_H
