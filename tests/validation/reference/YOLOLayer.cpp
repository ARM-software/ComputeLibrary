/*
 * Copyright (c) 2018-2020 Arm Limited.
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
#include "YOLOLayer.h"

#include "ActivationLayer.h"

#include "arm_compute/core/Types.h"
#include "tests/validation/Helpers.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace reference
{
template <typename T, typename std::enable_if<is_floating_point<T>::value, int>::type>
SimpleTensor<T> yolo_layer(const SimpleTensor<T> &src, const ActivationLayerInfo &info, int32_t num_classes)
{
    // Create reference
    SimpleTensor<T> dst{ src.shape(), src.data_type() };

    // Compute reference
    const T a(info.a());
    const T b(info.b());

    const uint32_t num_elements = src.num_elements();
#if defined(_OPENMP)
    #pragma omp parallel for
#endif /* _OPENMP */
    for(uint32_t i = 0; i < num_elements; ++i)
    {
        const size_t z = index2coord(dst.shape(), i).z() % (num_classes + 5);

        if(z != 2 && z != 3)
        {
            dst[i] = activate_float<T>(src[i], a, b, info.activation());
        }
        else
        {
            dst[i] = src[i];
        }
    }

    return dst;
}

template <>
SimpleTensor<uint8_t> yolo_layer<uint8_t>(const SimpleTensor<uint8_t> &src, const ActivationLayerInfo &info, int32_t num_classes)
{
    SimpleTensor<float>   src_tmp = convert_from_asymmetric(src);
    SimpleTensor<float>   dst_tmp = yolo_layer<float>(src_tmp, info, num_classes);
    SimpleTensor<uint8_t> dst     = convert_to_asymmetric<uint8_t>(dst_tmp, src.quantization_info());
    return dst;
}

template SimpleTensor<float> yolo_layer(const SimpleTensor<float> &src, const ActivationLayerInfo &info, int32_t num_classes);
template SimpleTensor<half> yolo_layer(const SimpleTensor<half> &src, const ActivationLayerInfo &info, int32_t num_classes);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
