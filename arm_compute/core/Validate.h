/*
 * Copyright (c) 2016, 2017 ARM Limited.
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
#ifndef __ARM_COMPUTE_VALIDATE_H__
#define __ARM_COMPUTE_VALIDATE_H__

#include "arm_compute/core/Error.h"
#include "arm_compute/core/HOGInfo.h"
#include "arm_compute/core/IKernel.h"
#include "arm_compute/core/IMultiHOG.h"
#include "arm_compute/core/IMultiImage.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/MultiImageInfo.h"
#include "arm_compute/core/Window.h"

#include <algorithm>

namespace arm_compute
{
/** Throw an error if the passed window is invalid.
 *
 * The subwindow is invalid if:
 * - It is not a valid window.
 * - Its dimensions don't match the full window's ones
 * - The step for each of its dimension is not identical to the corresponding one of the full window.
 *
 *  @param[in] function Function in which the error occurred.
 *  @param[in] file     Name of the file where the error occurred.
 *  @param[in] line     Line on which the error occurred.
 *  @param[in] full     Full size window
 *  @param[in] win      Window to validate.
 */
void error_on_mismatching_windows(const char *function, const char *file, const int line,
                                  const Window &full, const Window &win);
#define ARM_COMPUTE_ERROR_ON_MISMATCHING_WINDOWS(f, w) ::arm_compute::error_on_mismatching_windows(__func__, __FILE__, __LINE__, f, w)

/** Throw an error if the passed subwindow is invalid.
 *
 * The subwindow is invalid if:
 * - It is not a valid window.
 * - It is not fully contained inside the full window
 * - The step for each of its dimension is not identical to the corresponding one of the full window.
 *
 *  @param[in] function Function in which the error occurred.
 *  @param[in] file     Name of the file where the error occurred.
 *  @param[in] line     Line on which the error occurred.
 *  @param[in] full     Full size window
 *  @param[in] sub      Sub-window to validate.
 */
void error_on_invalid_subwindow(const char *function, const char *file, const int line,
                                const Window &full, const Window &sub);
#define ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(f, s) ::arm_compute::error_on_invalid_subwindow(__func__, __FILE__, __LINE__, f, s)

/** Throw an error if the passed coordinates have too many dimensions.
 *
 * The coordinates have too many dimensions if any of the dimensions greater or equal to max_dim is different from 0.
 *
 *  @param[in] function Function in which the error occurred.
 *  @param[in] file     Name of the file where the error occurred.
 *  @param[in] line     Line on which the error occurred.
 *  @param[in] pos      Coordinates to validate
 *  @param[in] max_dim  Maximum number of dimensions allowed.
 */
void error_on_coordinates_dimensions_gte(const char *function, const char *file, const int line,
                                         const Coordinates &pos, unsigned int max_dim);
#define ARM_COMPUTE_ERROR_ON_COORDINATES_DIMENSIONS_GTE(p, md) ::arm_compute::error_on_coordinates_dimensions_gte(__func__, __FILE__, __LINE__, p, md)

/** Throw an error if the passed window has too many dimensions.
 *
 * The window has too many dimensions if any of the dimension greater or equal to max_dim is different from 0.
 *
 *  @param[in] function Function in which the error occurred.
 *  @param[in] file     Name of the file where the error occurred.
 *  @param[in] line     Line on which the error occurred.
 *  @param[in] win      Window to validate
 *  @param[in] max_dim  Maximum number of dimensions allowed.
 */
void error_on_window_dimensions_gte(const char *function, const char *file, const int line,
                                    const Window &win, unsigned int max_dim);
#define ARM_COMPUTE_ERROR_ON_WINDOW_DIMENSIONS_GTE(w, md) ::arm_compute::error_on_window_dimensions_gte(__func__, __FILE__, __LINE__, w, md)

/* Check whether two tensors have different shapes.
 *
 * @param[in] tensor_1 First tensor to be compared
 * @param[in] tensor_2 Second tensor to be compared
 *
 * @return Return true if the two tensors have different shapes
 */
inline bool have_different_shapes(const ITensor *tensor_1, const ITensor *tensor_2)
{
    for(size_t i = 0; i < arm_compute::Coordinates::num_max_dimensions; ++i)
    {
        if(tensor_1->info()->dimension(i) != tensor_2->info()->dimension(i))
        {
            return true;
        }
    }

    return false;
}

/** Throw an error if the passed two tensors have different shapes
 *
 *  @param[in] function Function in which the error occurred.
 *  @param[in] file     Name of the file where the error occurred.
 *  @param[in] line     Line on which the error occurred.
 *  @param[in] tensor_1 The first tensor to be compared.
 *  @param[in] tensor_2 The second tensor to be compared.
 *  @param[in] tensors  (Optional) Further allowed tensors.
 */
template <typename... Ts>
void error_on_mismatching_shapes(const char *function, const char *file, const int line,
                                 const ITensor *tensor_1, const ITensor *tensor_2, Ts... tensors)
{
    ARM_COMPUTE_UNUSED(function);
    ARM_COMPUTE_UNUSED(file);
    ARM_COMPUTE_UNUSED(line);
    ARM_COMPUTE_UNUSED(tensor_1);
    ARM_COMPUTE_UNUSED(tensor_2);

    const std::array<const ITensor *, sizeof...(Ts)> tensors_array{ { std::forward<Ts>(tensors)... } };
    ARM_COMPUTE_UNUSED(tensors_array);

    ARM_COMPUTE_ERROR_ON_LOC_MSG(have_different_shapes(tensor_1, tensor_2) || std::any_of(tensors_array.begin(), tensors_array.end(), [&](const ITensor * tensor)
    {
        return have_different_shapes(tensor_1, tensor);
    }),
    function, file, line, "Tensors have different shapes");
}
#define ARM_COMPUTE_ERROR_ON_MISMATCHING_SHAPES(...) ::arm_compute::error_on_mismatching_shapes(__func__, __FILE__, __LINE__, __VA_ARGS__)

/** Throw an error if the passed two tensors have different data types
 *
 *  @param[in] function Function in which the error occurred.
 *  @param[in] file     Name of the file where the error occurred.
 *  @param[in] line     Line on which the error occurred.
 *  @param[in] tensor_1 The first tensor to be compared.
 *  @param[in] tensor_2 The second tensor to be compared.
 *  @param[in] tensors  (Optional) Further allowed tensors.
 */
template <typename... Ts>
void error_on_mismatching_data_types(const char *function, const char *file, const int line,
                                     const ITensor *tensor_1, const ITensor *tensor_2, Ts... tensors)
{
    ARM_COMPUTE_UNUSED(function);
    ARM_COMPUTE_UNUSED(file);
    ARM_COMPUTE_UNUSED(line);
    ARM_COMPUTE_UNUSED(tensor_1);
    ARM_COMPUTE_UNUSED(tensor_2);

    DataType &&first_data_type = tensor_1->info()->data_type();
    ARM_COMPUTE_UNUSED(first_data_type);

    const std::array<const ITensor *, sizeof...(Ts)> tensors_array{ { std::forward<Ts>(tensors)... } };
    ARM_COMPUTE_UNUSED(tensors_array);

    ARM_COMPUTE_ERROR_ON_LOC_MSG(tensor_2->info()->data_type() != first_data_type || std::any_of(tensors_array.begin(), tensors_array.end(), [&](const ITensor * tensor)
    {
        return tensor->info()->data_type() != first_data_type;
    }),
    function, file, line, "Tensors have different data types");
}

#define ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(...) ::arm_compute::error_on_mismatching_data_types(__func__, __FILE__, __LINE__, __VA_ARGS__)

/** Throw an error if the format of the passed tensor/multi-image does not match any of the formats provided.
 *
 *  @param[in] function Function in which the error occurred.
 *  @param[in] file     Name of the file where the error occurred.
 *  @param[in] line     Line on which the error occurred.
 *  @param[in] object   Tensor/multi-image to validate.
 *  @param[in] format   First format allowed.
 *  @param[in] formats  (Optional) Further allowed formats.
 */
template <typename T, typename F, typename... Fs>
void error_on_format_not_in(const char *function, const char *file, const int line,
                            const T *object, F &&format, Fs &&... formats)
{
    ARM_COMPUTE_ERROR_ON_LOC(object == nullptr, function, file, line);

    Format &&object_format = object->info()->format();
    ARM_COMPUTE_UNUSED(object_format);

    ARM_COMPUTE_ERROR_ON_LOC(object_format == Format::UNKNOWN, function, file, line);

    const std::array<F, sizeof...(Fs)> formats_array{ { std::forward<Fs>(formats)... } };
    ARM_COMPUTE_UNUSED(formats_array);

    ARM_COMPUTE_ERROR_ON_LOC_MSG(object_format != format && std::none_of(formats_array.begin(), formats_array.end(), [&](const F & f)
    {
        return f == object_format;
    }),
    function, file, line, "Format %s not supported by this kernel", string_from_format(object_format).c_str());
}
#define ARM_COMPUTE_ERROR_ON_FORMAT_NOT_IN(t, ...) ::arm_compute::error_on_format_not_in(__func__, __FILE__, __LINE__, t, __VA_ARGS__)

/** Throw an error if the data type of the passed tensor does not match any of the data types provided.
 *
 *  @param[in] function Function in which the error occurred.
 *  @param[in] file     Name of the file where the error occurred.
 *  @param[in] line     Line on which the error occurred.
 *  @param[in] tensor   Tensor to validate.
 *  @param[in] dt       First data type allowed.
 *  @param[in] dts      (Optional) Further allowed data types.
 */
template <typename T, typename... Ts>
void error_on_data_type_not_in(const char *function, const char *file, const int line,
                               const ITensor *tensor, T &&dt, Ts &&... dts)
{
    ARM_COMPUTE_ERROR_ON_LOC(tensor == nullptr, function, file, line);

    DataType &&tensor_dt = tensor->info()->data_type();
    ARM_COMPUTE_UNUSED(tensor_dt);

    ARM_COMPUTE_ERROR_ON_LOC(tensor_dt == DataType::UNKNOWN, function, file, line);

    const std::array<T, sizeof...(Ts)> dts_array{ { std::forward<Ts>(dts)... } };
    ARM_COMPUTE_UNUSED(dts_array);

    ARM_COMPUTE_ERROR_ON_LOC_MSG(tensor_dt != dt && std::none_of(dts_array.begin(), dts_array.end(), [&](const T & d)
    {
        return d == tensor_dt;
    }),
    function, file, line, "ITensor data type %s not supported by this kernel", string_from_data_type(tensor_dt).c_str());
}
#define ARM_COMPUTE_ERROR_ON_DATA_TYPE_NOT_IN(d, ...) ::arm_compute::error_on_data_type_not_in(__func__, __FILE__, __LINE__, d, __VA_ARGS__)

/** Throw an error if the data type or the number of channels of the passed tensor does not match any of the data types and number of channels provided.
 *
 *  @param[in] function     Function in which the error occurred.
 *  @param[in] file         Name of the file where the error occurred.
 *  @param[in] line         Line on which the error occurred.
 *  @param[in] tensor       Tensor to validate.
 *  @param[in] num_channels Number of channels to check
 *  @param[in] dt           First data type allowed.
 *  @param[in] dts          (Optional) Further allowed data types.
 */
template <typename T, typename... Ts>
void error_on_data_type_channel_not_in(const char *function, const char *file, const int line,
                                       const ITensor *tensor, size_t num_channels, T &&dt, Ts &&... dts)
{
    error_on_data_type_not_in(function, file, line, tensor, std::forward<T>(dt), std::forward<Ts>(dts)...);

    const size_t tensor_nc = tensor->info()->num_channels();
    ARM_COMPUTE_UNUSED(tensor_nc);

    ARM_COMPUTE_ERROR_ON_LOC_MSG(tensor_nc != num_channels, function, file, line, "Number of channels %d. Required number of channels %d", tensor_nc, num_channels);
}
#define ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(d, c, ...) ::arm_compute::error_on_data_type_channel_not_in(__func__, __FILE__, __LINE__, d, c, __VA_ARGS__)

/** Throw an error if the tensor is not 2D.
 *
 *  @param[in] function Function in which the error occurred.
 *  @param[in] file     Name of the file where the error occurred.
 *  @param[in] line     Line on which the error occurred.
 *  @param[in] tensor   Tensor to validate.
 */
void error_on_tensor_not_2d(const char *function, const char *file, const int line,
                            const ITensor *tensor);
#define ARM_COMPUTE_ERROR_ON_TENSOR_NOT_2D(t) ::arm_compute::error_on_tensor_not_2d(__func__, __FILE__, __LINE__, t)

/** Throw an error if the channel is not in channels.
 *
 *  @param[in] function Function in which the error occurred.
 *  @param[in] file     Name of the file where the error occurred.
 *  @param[in] line     Line on which the error occurred.
 *  @param[in] cn       Input channel
 *  @param[in] channel  First channel allowed.
 *  @param[in] channels (Optional) Further allowed channels.
 */
template <typename T, typename... Ts>
void error_on_channel_not_in(const char *function, const char *file, const int line,
                             T cn, T &&channel, Ts &&... channels)
{
    ARM_COMPUTE_ERROR_ON_LOC(cn == Channel::UNKNOWN, function, file, line);

    const std::array<T, sizeof...(Ts)> channels_array{ { std::forward<Ts>(channels)... } };
    ARM_COMPUTE_UNUSED(channels_array);
    ARM_COMPUTE_ERROR_ON_LOC(channel != cn && std::none_of(channels_array.begin(), channels_array.end(), [&](const T & f)
    {
        return f == cn;
    }),
    function, file, line);
}
#define ARM_COMPUTE_ERROR_ON_CHANNEL_NOT_IN(c, ...) ::arm_compute::error_on_channel_not_in(__func__, __FILE__, __LINE__, c, __VA_ARGS__)

/** Throw an error if the channel is not in format.
 *
 *  @param[in] function Function in which the error occurred.
 *  @param[in] file     Name of the file where the error occurred.
 *  @param[in] line     Line on which the error occurred.
 *  @param[in] fmt      Input channel
 *  @param[in] cn       First channel allowed.
 */
void error_on_channel_not_in_known_format(const char *function, const char *file, const int line,
                                          Format fmt, Channel cn);
#define ARM_COMPUTE_ERROR_ON_CHANNEL_NOT_IN_KNOWN_FORMAT(f, c) ::arm_compute::error_on_channel_not_in_known_format(__func__, __FILE__, __LINE__, f, c)

/** Throw an error if the @ref IMultiHOG container is invalid
 *
 * An @ref IMultiHOG container is invalid if:
 *
 * -# it is a nullptr
 * -# it doesn't contain models
 * -# it doesn't have the HOG data objects with the same phase_type, normalization_type and l2_hyst_threshold (if normalization_type == L2HYS_NORM)
 *
 *  @param[in] function  Function in which the error occurred.
 *  @param[in] file      Name of the file where the error occurred.
 *  @param[in] line      Line on which the error occurred.
 *  @param[in] multi_hog IMultiHOG container to validate
 */
void error_on_invalid_multi_hog(const char *function, const char *file, const int line,
                                const IMultiHOG *multi_hog);
#define ARM_COMPUTE_ERROR_ON_INVALID_MULTI_HOG(m) ::arm_compute::error_on_invalid_multi_hog(__func__, __FILE__, __LINE__, m)

/** Throw an error if the kernel is not configured.
 *
 *  @param[in] function Function in which the error occurred.
 *  @param[in] file     Name of the file where the error occurred.
 *  @param[in] line     Line on which the error occurred.
 *  @param[in] kernel   Kernel to validate.
 */
void error_on_unconfigured_kernel(const char *function, const char *file, const int line,
                                  const IKernel *kernel);
#define ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(k) ::arm_compute::error_on_unconfigured_kernel(__func__, __FILE__, __LINE__, k)
}
#endif /* __ARM_COMPUTE_VALIDATE_H__*/
