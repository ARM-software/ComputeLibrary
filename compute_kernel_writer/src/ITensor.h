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

#ifndef CKW_SRC_ITENSOR_H
#define CKW_SRC_ITENSOR_H

#include "src/ITensorArgument.h"

namespace ckw
{

/** The generic class for all tensor objects in CKW.
 *
 * Tensors in CKW are always kernel arguments consisting of:
 *   - Essential information such as name, tensor info, etc.
 *   - Tensor storage access: allowing load/store operation to perform.
 *   - Tensor component access: allowing interaction with tensor information such as shape, strides, etc. in the form of tile objects.
 */
class ITensor : public ITensorArgument, public ITensorStorageAccess, public ITensorComponentAccess
{
};

} // namespace ckw

#endif // CKW_SRC_ITENSOR_H