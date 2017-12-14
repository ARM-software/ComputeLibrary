/*
 * Copyright (c) 2017 ARM Limited.
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
#ifndef __ARM_COMPUTE_MEMORY_H__
#define __ARM_COMPUTE_MEMORY_H__

#include <cstddef>
#include <memory>

namespace arm_compute
{
/** CPU implementation of memory object */
class Memory
{
public:
    /** Default Constructor */
    Memory();
    /** Default Constructor
     *
     * @note Ownership of the memory is transferred to this object
     *
     * @param[in] memory Memory to be imported
     */
    Memory(std::shared_ptr<uint8_t> memory);
    /** Default Constructor
     *
     * @note Ownership of the memory is not transferred to this object.
     *       Thus management (allocate/free) should be done by the client.
     *
     * @param[in] memory Memory to be imported
     */
    Memory(uint8_t *memory);
    /** Allow instances of this class to be copied */
    Memory(const Memory &) = default;
    /** Allow instances of this class to be copy assigned */
    Memory &operator=(const Memory &) = default;
    /** Allow instances of this class to be moved */
    Memory(Memory &&) noexcept = default;
    /** Allow instances of this class to be move assigned */
    Memory &operator=(Memory &&) noexcept = default;

    /** Returns the pointer to the allocated data.
     *
     * @return Pointer to the allocated data
     */
    uint8_t *buffer();
    /** Returns the pointer to the allocated data.
     *
     * @return Pointer to the allocated data
     */
    uint8_t *buffer() const;
    /** Handle of internal memory
     *
     * @return Handle of memory
     */
    uint8_t **handle();

private:
    uint8_t                 *_memory;
    std::shared_ptr<uint8_t> _memory_owned;
};
}
#endif /* __ARM_COMPUTE_MEMORY_H__ */
