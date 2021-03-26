/*
 * Copyright (c) 2020-2021 Arm Limited.
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

#ifndef ARM_COMPUTE_IDEVICE_H
#define ARM_COMPUTE_IDEVICE_H

#include <string>

namespace arm_compute
{
/** Device types */
enum class DeviceType
{
    NEON,
    CL,
};

/** Interface for device object */
class IDevice
{
public:
    /** Virtual Destructor */
    virtual ~IDevice() = default;

    /** Device type accessor */
    virtual DeviceType type() const = 0;

    /** Check if extensions on a device are supported
     *
     * @param[in] extension An extension to check if it's supported.
     *
     * @return True if the extension is supported else false
     */
    virtual bool supported(const std::string &extension) const = 0;
};
} // namespace arm_compute

#endif /* ARM_COMPUTE_IDEVICE_H */
