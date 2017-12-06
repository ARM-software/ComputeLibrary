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
#include "arm_compute/core/Validate.h"

arm_compute::Status arm_compute::error_on_mismatching_windows(const char *function, const char *file, const int line,
                                                              const arm_compute::Window &full, const arm_compute::Window &win)
{
    full.validate();
    win.validate();

    for(size_t i = 0; i < arm_compute::Coordinates::num_max_dimensions; ++i)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_LOC(full[i].start() != win[i].start(), function, file, line);
        ARM_COMPUTE_RETURN_ERROR_ON_LOC(full[i].end() != win[i].end(), function, file, line);
        ARM_COMPUTE_RETURN_ERROR_ON_LOC(full[i].step() != win[i].step(), function, file, line);
    }
    return arm_compute::Status{};
}

arm_compute::Status arm_compute::error_on_invalid_subwindow(const char *function, const char *file, const int line,
                                                            const arm_compute::Window &full, const arm_compute::Window &sub)
{
    full.validate();
    sub.validate();

    for(size_t i = 0; i < arm_compute::Coordinates::num_max_dimensions; ++i)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_LOC(full[i].start() > sub[i].start(), function, file, line);
        ARM_COMPUTE_RETURN_ERROR_ON_LOC(full[i].end() < sub[i].end(), function, file, line);
        ARM_COMPUTE_RETURN_ERROR_ON_LOC(full[i].step() != sub[i].step(), function, file, line);
        ARM_COMPUTE_RETURN_ERROR_ON_LOC((sub[i].start() - full[i].start()) % sub[i].step(), function, file, line);
    }
    return arm_compute::Status{};
}

arm_compute::Status arm_compute::error_on_window_not_collapsable_at_dimension(const char *function, const char *file, const int line,
                                                                              const arm_compute::Window &full, const arm_compute::Window &window, const int dim)
{
    full.validate();
    window.validate();

    ARM_COMPUTE_RETURN_ERROR_ON_LOC(window[dim].start() != 0, function, file, line);
    ARM_COMPUTE_RETURN_ERROR_ON_LOC(window[dim].start() != full[dim].start(), function, file, line);
    ARM_COMPUTE_RETURN_ERROR_ON_LOC(full[dim].end() != window[dim].end(), function, file, line);

    return arm_compute::Status{};
}

arm_compute::Status arm_compute::error_on_coordinates_dimensions_gte(const char *function, const char *file, const int line,
                                                                     const arm_compute::Coordinates &pos, unsigned int max_dim)
{
    for(unsigned int i = max_dim; i < arm_compute::Coordinates::num_max_dimensions; ++i)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_LOC(pos[i] != 0, function, file, line);
    }
    return arm_compute::Status{};
}

arm_compute::Status arm_compute::error_on_window_dimensions_gte(const char *function, const char *file, const int line,
                                                                const arm_compute::Window &win, unsigned int max_dim)
{
    for(unsigned int i = max_dim; i < arm_compute::Coordinates::num_max_dimensions; ++i)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_LOC_MSG(win[i].start() != 0 || win[i].end() != win[i].step(),
                                            function, file, line,
                                            "Maximum number of dimensions expected %u but dimension %u is not empty", max_dim, i);
    }
    return arm_compute::Status{};
}

arm_compute::Status arm_compute::error_on_tensor_not_2d(const char *function, const char *file, const int line,
                                                        const arm_compute::ITensor *tensor)
{
    ARM_COMPUTE_RETURN_ERROR_ON_LOC(tensor == nullptr, function, file, line);
    ARM_COMPUTE_RETURN_ERROR_ON_LOC(tensor->info() == nullptr, function, file, line);
    ARM_COMPUTE_RETURN_ERROR_ON_LOC_MSG(tensor->info()->num_dimensions() != 2,
                                        function, file, line,
                                        "Only 2D Tensors are supported by this kernel (%d passed)", tensor->info()->num_dimensions());
    return arm_compute::Status{};
}

arm_compute::Status arm_compute::error_on_channel_not_in_known_format(const char *function, const char *file, const int line,
                                                                      arm_compute::Format fmt, arm_compute::Channel cn)
{
    ARM_COMPUTE_RETURN_ERROR_ON_LOC(fmt == arm_compute::Format::UNKNOWN, function, file, line);
    ARM_COMPUTE_RETURN_ERROR_ON_LOC(cn == arm_compute::Channel::UNKNOWN, function, file, line);

    switch(fmt)
    {
        case arm_compute::Format::RGB888:
            arm_compute::error_on_channel_not_in(function, file, line, cn, arm_compute::Channel::R, arm_compute::Channel::G, arm_compute::Channel::B);
            break;
        case arm_compute::Format::RGBA8888:
            arm_compute::error_on_channel_not_in(function, file, line, cn, arm_compute::Channel::R, arm_compute::Channel::G, arm_compute::Channel::B, arm_compute::Channel::A);
            break;
        case arm_compute::Format::UV88:
            arm_compute::error_on_channel_not_in(function, file, line, cn, arm_compute::Channel::U, arm_compute::Channel::V);
            break;
        case arm_compute::Format::IYUV:
        case arm_compute::Format::UYVY422:
        case arm_compute::Format::YUYV422:
        case arm_compute::Format::NV12:
        case arm_compute::Format::NV21:
        case arm_compute::Format::YUV444:
            arm_compute::error_on_channel_not_in(function, file, line, cn, arm_compute::Channel::Y, arm_compute::Channel::U, arm_compute::Channel::V);
            break;
        default:
            ARM_COMPUTE_ERROR_LOC(function, file, line, "Not supported format.");
    }
    return arm_compute::Status{};
}

arm_compute::Status arm_compute::error_on_invalid_multi_hog(const char *function, const char *file, const int line,
                                                            const arm_compute::IMultiHOG *multi_hog)
{
    ARM_COMPUTE_RETURN_ERROR_ON_LOC(nullptr == multi_hog, function, file, line);
    ARM_COMPUTE_RETURN_ERROR_ON_LOC(0 == multi_hog->num_models(), function, file, line);

    for(size_t i = 1; i < multi_hog->num_models(); ++i)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_LOC_MSG(multi_hog->model(0)->info()->phase_type() != multi_hog->model(i)->info()->phase_type(),
                                            function, file, line,
                                            "All HOG parameters must have the same phase type");
        ARM_COMPUTE_RETURN_ERROR_ON_LOC_MSG(multi_hog->model(0)->info()->normalization_type() != multi_hog->model(i)->info()->normalization_type(),
                                            function, file, line,
                                            "All HOG parameters must have the same normalization type");
        ARM_COMPUTE_RETURN_ERROR_ON_LOC_MSG((multi_hog->model(0)->info()->l2_hyst_threshold() != multi_hog->model(i)->info()->l2_hyst_threshold())
                                            && (multi_hog->model(0)->info()->normalization_type() == arm_compute::HOGNormType::L2HYS_NORM),
                                            function, file, line,
                                            "All HOG parameters must have the same l2 hysteresis threshold if you use L2 hysteresis normalization type");
    }
    return arm_compute::Status{};
}

arm_compute::Status arm_compute::error_on_unconfigured_kernel(const char *function, const char *file, const int line,
                                                              const arm_compute::IKernel *kernel)
{
    ARM_COMPUTE_RETURN_ERROR_ON_LOC(kernel == nullptr, function, file, line);
    ARM_COMPUTE_RETURN_ERROR_ON_LOC_MSG((kernel->window().x().start() == kernel->window().x().end()) && (kernel->window().x().end() == 0) && (kernel->window().x().step() == 0),
                                        function, file, line,
                                        "This kernel hasn't been configured.");
    return arm_compute::Status{};
}

arm_compute::Status arm_compute::error_on_invalid_subtensor(const char *function, const char *file, const int line,
                                                            const TensorShape &parent_shape, const Coordinates &coords, const TensorShape &shape)
{
    // Subtensor should not index in x, y dimensions.
    ARM_COMPUTE_RETURN_ERROR_ON_LOC(((coords.x() != 0) && (coords.y() != 0)), function, file, line);
    // Subtensor shape should match parent tensor in x, y dimensions.
    ARM_COMPUTE_RETURN_ERROR_ON_LOC(((parent_shape.x() != shape.x()) && (parent_shape.y() != parent_shape.y())), function, file, line);

    // Check dimensions
    for(unsigned int i = 0; i < TensorShape::num_max_dimensions; ++i)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_LOC(((coords[i] >= static_cast<int>(parent_shape[i])) || (coords[i] + static_cast<int>(shape[i]) > static_cast<int>(parent_shape[i]))),
                                        function, file, line);
    }
    return arm_compute::Status{};
}

arm_compute::Status arm_compute::error_on_invalid_subtensor_valid_region(const char *function, const char *file, const int line,
                                                                         const ValidRegion &parent_valid_region, const ValidRegion &valid_region)
{
    // Check valid regions
    for(unsigned int d = 0; d < TensorShape::num_max_dimensions; ++d)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_LOC((parent_valid_region.anchor[d] > valid_region.anchor[d]), function, file, line);
        ARM_COMPUTE_RETURN_ERROR_ON_LOC((parent_valid_region.anchor[d] + static_cast<int>(parent_valid_region.shape[d])) < (valid_region.anchor[d] + static_cast<int>(valid_region.shape[d])),
                                        function, file, line);
    }

    return arm_compute::Status{};
}
