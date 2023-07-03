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

#ifndef COMPUTE_KERNEL_WRITER_SRC_HELPERS_H
#define COMPUTE_KERNEL_WRITER_SRC_HELPERS_H

#include <cstdint>
#include <string>

/** Generic helper functions */
namespace ckw
{
/** Helper function to convert a decimal number passed as int32_t variable to hexadecimal number as string
 *
 * @param[in] dec  Decimal number. It must be >= 0 and < 16
 *
 * @return the OpenCL datatype as a string
 */
std::string dec_to_hex_as_string(int32_t dec);

/** Helper function to clamp a value between min_val and max_val
 *
 * @param[in] val     Value to clamp
 * @param[in] min_val Lower value
 * @param[in] max_val Upper value
 *
 * @return the clamped value
 */
template <typename T>
T clamp(const T &val, const T &min_val, const T &max_val)
{
    return std::max(min_val, std::min(val, max_val));
}
} // namespace ckw
#endif /* COMPUTE_KERNEL_WRITER_SRC_HELPERS_H */
