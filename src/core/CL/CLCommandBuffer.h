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

#ifndef ACL_SRC_CORE_CL_CLCOMMANDBUFFER_H
#define ACL_SRC_CORE_CL_CLCOMMANDBUFFER_H

#include "arm_compute/core/CL/OpenCL.h"

#include <cstdint>
#include <memory>
#include <type_traits>

namespace arm_compute
{

/** Command buffer contains a list of commands that is constructed once and later enqueued multiple times.
 *
 * To prepare a command buffer:
 *   - Construct a new command buffer targeting a command queue using @ref CLCommandBuffer::create.
 *   - Add kernel enqueue command to the buffer using @ref CLCommandBuffer::add_kernel.
 *     The kernel must be ready to be enqueued with all the arguments set.
 *   - Specify which kernel argument is mutable after the command buffer has been finalized.
 *   - When all the kernel enqueue commands have been added, call @ref CLCommandBuffer::finalize.
 *     After this point the command buffer is ready to be executed.
 *
 * To execute the command buffer:
 *   - Make any changes in the value which the mutable arguments are pointing to.
 *   - Call @ref CLCommandBuffer::update to apply the argument value changes.
 *   - Call @ref CLCommandBuffer::enqueue to enqueue the command buffer to execute.
 */
class CLCommandBuffer
{
public:
    /** Create a new command buffer targeting the specified command queue.
     *
     * @param[in] queue The command queue to execute the command buffer.
     *
     * @return A unique pointer to the newly created command buffer.
     */
    static std::unique_ptr<CLCommandBuffer> create(cl_command_queue queue);

    /** Constructor. */
    CLCommandBuffer();

    /** Destructor. */
    virtual ~CLCommandBuffer();

    /** Disallow copy constructor. */
    CLCommandBuffer(const CLCommandBuffer &) = delete;

    /** Disallow copy assignment. */
    CLCommandBuffer &operator=(const CLCommandBuffer &) = delete;

    /** Disallow move constructor. */
    CLCommandBuffer(CLCommandBuffer &&other) = delete;

    /** Disallow move assignment. */
    CLCommandBuffer &operator=(CLCommandBuffer &&other) = delete;

    /** Add a kernel enqueue command to the command queue.
     *
     * This function must be called before the command buffer has been finalized.
     *
     * @param[in] kernel The CL kernel.
     * @param[in] offset The global work offset.
     * @param[in] global The global work size.
     * @param[in] local  The local work size.
     */
    virtual void
    add_kernel(cl_kernel kernel, const cl::NDRange &offset, const cl::NDRange &global, const cl::NDRange &local) = 0;

    /** Add the mutable argument to the current kernel enqueue command.
     *
     * This function must be called after @ref CLCommandBuffer::add_kernel but before the command buffer
     * has been finalized.
     *
     * The pointer must be valid and it must point to the correct value at the time
     * @ref CLCommandBuffer::update is called so that the value of the argument
     * can be applied successfully to the kernel enqueue command.
     *
     * @param[in] arg_idx The index of the argument in the current kernel program.
     * @param[in] value   The pointer to the value of the argument.
     */
    template <typename T, typename = std::enable_if_t<std::is_arithmetic<T>::value || std::is_pointer<T>::value>>
    void add_mutable_argument(cl_uint arg_idx, const T *value)
    {
        add_mutable_argument_generic(arg_idx, value, sizeof(T));
    }

    /** Finalize the command buffer. */
    virtual void finalize() = 0;

    /** Update the command buffer with new kernel argument values.
     *
     * This function must be called after the command buffer has been finalized.
     *
     * All the value pointed by the mutable argument will be applied to the command buffer.
     */
    virtual void update() = 0;

    /** Enqueue the command buffer.
     *
     * This function must be called after the command buffer has been finalized.
     */
    virtual void enqueue() = 0;

    /** Check if the command buffer has been finalized.
     *
     * @return true if the command buffer has been finalized.
     */
    virtual bool is_finalized() const = 0;

protected:
    /** Add the mutable argument to the current kernel enqueue command.
     *
     * @see CLCommandBuffer::add_mutable_argument for more information.
     */
    virtual void add_mutable_argument_generic(cl_uint arg_idx, const void *value, size_t size) = 0;

    /** The state of the command buffer. */
    enum class State : int32_t
    {
        /** The command buffer has been created and is being specified. */
        Created,

        /** The command buffer has been finalized and is ready to be executed. */
        Finalized,
    };

    /** Get the state of the command buffer. */
    State state() const;

    /** Set the state of the command buffer. */
    CLCommandBuffer &state(State state);

private:
    State _state{State::Created};
};

} // namespace arm_compute

#endif // ACL_SRC_CORE_CL_CLCOMMANDBUFFER_H
