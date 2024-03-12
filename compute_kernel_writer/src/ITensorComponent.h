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

#ifndef CKW_SRC_ITENSORCOMPONENT_H
#define CKW_SRC_ITENSORCOMPONENT_H

#include "ckw/types/TensorComponentType.h"

#include "src/ITile.h"

namespace ckw
{

/** A tensor component provides access to tensor information such as shape, strides, etc. in the form of @ref ITile objects. */
class ITensorComponent
{
public:
    /** Destructor. */
    virtual ~ITensorComponent() = default;

    /** Get the tile variable for the component. */
    virtual ITile &tile() = 0;

    /** Get the const tile variable for the component. */
    virtual const ITile &tile() const = 0;

    /** Get the component type. */
    virtual TensorComponentType component_type() const = 0;
};

} // namespace ckw

#endif // CKW_SRC_ITENSORCOMPONENT_H
