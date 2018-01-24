/*
 * Copyright (c) 2017-2018 ARM Limited.
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
#ifndef __ARM_COMPUTE_GRAPH_UTILS_H__
#define __ARM_COMPUTE_GRAPH_UTILS_H__

#include "arm_compute/core/PixelValue.h"
#include "arm_compute/graph/Graph.h"
#include "arm_compute/graph/ITensorAccessor.h"
#include "arm_compute/graph/Types.h"

#include <random>
#include <string>
#include <vector>

namespace arm_compute
{
namespace graph_utils
{
/** PPM writer class */
class PPMWriter : public graph::ITensorAccessor
{
public:
    /** Constructor
     *
     * @param[in] name    PPM file name
     * @param[in] maximum Maximum elements to access
     */
    PPMWriter(std::string name, unsigned int maximum = 1);
    /** Allows instances to move constructed */
    PPMWriter(PPMWriter &&) = default;

    // Inherited methods overriden:
    bool access_tensor(ITensor &tensor) override;

private:
    const std::string _name;
    unsigned int      _iterator;
    unsigned int      _maximum;
};

/** Dummy accessor class */
class DummyAccessor final : public graph::ITensorAccessor
{
public:
    /** Constructor
     *
     * @param[in] maximum Maximum elements to write
     */
    DummyAccessor(unsigned int maximum = 1);
    /** Allows instances to move constructed */
    DummyAccessor(DummyAccessor &&) = default;

    // Inherited methods overriden:
    bool access_tensor(ITensor &tensor) override;

private:
    unsigned int _iterator;
    unsigned int _maximum;
};

/** PPM accessor class */
class PPMAccessor final : public graph::ITensorAccessor
{
public:
    /** Constructor
     *
     * @param[in] ppm_path Path to PPM file
     * @param[in] bgr      (Optional) Fill the first plane with blue channel (default = false)
     * @param[in] mean_r   (Optional) Red mean value to be subtracted from red channel
     * @param[in] mean_g   (Optional) Green mean value to be subtracted from green channel
     * @param[in] mean_b   (Optional) Blue mean value to be subtracted from blue channel
     * @param[in] std_r    (Optional) Red standard deviation value to be divided from red channel
     * @param[in] std_g    (Optional) Green standard deviation value to be divided from green channel
     * @param[in] std_b    (Optional) Blue standard deviation value to be divided from blue channel
     */
    PPMAccessor(std::string ppm_path, bool bgr = true,
                float mean_r = 0.0f, float mean_g = 0.0f, float mean_b = 0.0f,
                float std_r = 1.f, float std_g = 1.f, float std_b = 1.f);
    /** Allow instances of this class to be move constructed */
    PPMAccessor(PPMAccessor &&) = default;

    // Inherited methods overriden:
    bool access_tensor(ITensor &tensor) override;

private:
    const std::string _ppm_path;
    const bool        _bgr;
    const float       _mean_r;
    const float       _mean_g;
    const float       _mean_b;
    const float       _std_r;
    const float       _std_g;
    const float       _std_b;
};

/** Result accessor class */
class TopNPredictionsAccessor final : public graph::ITensorAccessor
{
public:
    /** Constructor
     *
     * @param[in]  labels_path   Path to labels text file.
     * @param[in]  top_n         (Optional) Number of output classes to print
     * @param[out] output_stream (Optional) Output stream
     */
    TopNPredictionsAccessor(const std::string &labels_path, size_t top_n = 5, std::ostream &output_stream = std::cout);
    /** Allow instances of this class to be move constructed */
    TopNPredictionsAccessor(TopNPredictionsAccessor &&) = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    TopNPredictionsAccessor(const TopNPredictionsAccessor &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    TopNPredictionsAccessor &operator=(const TopNPredictionsAccessor &) = delete;

    // Inherited methods overriden:
    bool access_tensor(ITensor &tensor) override;

private:
    template <typename T>
    void access_predictions_tensor(ITensor &tensor);

    std::vector<std::string> _labels;
    std::ostream            &_output_stream;
    size_t                   _top_n;
};

/** Random accessor class */
class RandomAccessor final : public graph::ITensorAccessor
{
public:
    /** Constructor
     *
     * @param[in] lower Lower bound value.
     * @param[in] upper Upper bound value.
     * @param[in] seed  (Optional) Seed used to initialise the random number generator.
     */
    RandomAccessor(PixelValue lower, PixelValue upper, const std::random_device::result_type seed = 0);
    /** Allows instances to move constructed */
    RandomAccessor(RandomAccessor &&) = default;

    // Inherited methods overriden:
    bool access_tensor(ITensor &tensor) override;

private:
    template <typename T, typename D>
    void fill(ITensor &tensor, D &&distribution);
    PixelValue                      _lower;
    PixelValue                      _upper;
    std::random_device::result_type _seed;
};

/** Numpy Binary loader class*/
class NumPyBinLoader final : public graph::ITensorAccessor
{
public:
    /** Default Constructor
     *
     * @param filename Binary file name
     */
    NumPyBinLoader(std::string filename);
    /** Allows instances to move constructed */
    NumPyBinLoader(NumPyBinLoader &&) = default;

    // Inherited methods overriden:
    bool access_tensor(ITensor &tensor) override;

private:
    const std::string _filename;
};

/** Generates appropriate random accessor
 *
 * @param[in] lower Lower random values bound
 * @param[in] upper Upper random values bound
 * @param[in] seed  Random generator seed
 *
 * @return A ramdom accessor
 */
inline std::unique_ptr<graph::ITensorAccessor> get_random_accessor(PixelValue lower, PixelValue upper, const std::random_device::result_type seed = 0)
{
    return arm_compute::support::cpp14::make_unique<RandomAccessor>(lower, upper, seed);
}

/** Generates appropriate weights accessor according to the specified path
 *
 * @note If path is empty will generate a DummyAccessor else will generate a NumPyBinLoader
 *
 * @param[in] path      Path to the data files
 * @param[in] data_file Relative path to the data files from path
 *
 * @return An appropriate tensor accessor
 */
inline std::unique_ptr<graph::ITensorAccessor> get_weights_accessor(const std::string &path, const std::string &data_file)
{
    if(path.empty())
    {
        return arm_compute::support::cpp14::make_unique<DummyAccessor>();
    }
    else
    {
        return arm_compute::support::cpp14::make_unique<NumPyBinLoader>(path + data_file);
    }
}

/** Generates appropriate input accessor according to the specified ppm_path
 *
 * @note If ppm_path is empty will generate a DummyAccessor else will generate a PPMAccessor
 *
 * @param[in] ppm_path Path to PPM file
 * @param[in] mean_r   Red mean value to be subtracted from red channel
 * @param[in] mean_g   Green mean value to be subtracted from green channel
 * @param[in] mean_b   Blue mean value to be subtracted from blue channel
 * @param[in] std_r    (Optional) Red standard deviation value to be divided from red channel
 * @param[in] std_g    (Optional) Green standard deviation value to be divided from green channel
 * @param[in] std_b    (Optional) Blue standard deviation value to be divided from blue channel
 * @param[in] bgr      (Optional) Fill the first plane with blue channel (default = true)
 *
 * @return An appropriate tensor accessor
 */
inline std::unique_ptr<graph::ITensorAccessor> get_input_accessor(const std::string &ppm_path,
                                                                  float mean_r = 0.f, float mean_g = 0.f, float mean_b = 0.f,
                                                                  float std_r = 1.f, float std_g = 1.f, float std_b = 1.f,
                                                                  bool bgr = true)
{
    if(ppm_path.empty())
    {
        return arm_compute::support::cpp14::make_unique<DummyAccessor>();
    }
    else
    {
        return arm_compute::support::cpp14::make_unique<PPMAccessor>(ppm_path, bgr,
                                                                     mean_r, mean_g, mean_b,
                                                                     std_r, std_g, std_b);
    }
}

/** Utility function to return the TargetHint
 *
 * @param[in] target Integer value which expresses the selected target. Must be 0 for NEON or 1 for OpenCL
 *
 * @return the TargetHint
 */
inline graph::TargetHint set_target_hint(int target)
{
    ARM_COMPUTE_ERROR_ON_MSG(target > 1, "Invalid target. Target must be 0 (NEON) or 1 (OpenCL)");
    if(target == 1 && graph::Graph::opencl_is_available())
    {
        // If type of target is OpenCL, check if OpenCL is available and initialize the scheduler
        return graph::TargetHint::OPENCL;
    }
    else
    {
        return graph::TargetHint::NEON;
    }
}

/** Generates appropriate output accessor according to the specified labels_path
 *
 * @note If labels_path is empty will generate a DummyAccessor else will generate a TopNPredictionsAccessor
 *
 * @param[in]  labels_path   Path to labels text file
 * @param[in]  top_n         (Optional) Number of output classes to print
 * @param[out] output_stream (Optional) Output stream
 *
 * @return An appropriate tensor accessor
 */
inline std::unique_ptr<graph::ITensorAccessor> get_output_accessor(const std::string &labels_path, size_t top_n = 5, std::ostream &output_stream = std::cout)
{
    if(labels_path.empty())
    {
        return arm_compute::support::cpp14::make_unique<DummyAccessor>(0);
    }
    else
    {
        return arm_compute::support::cpp14::make_unique<TopNPredictionsAccessor>(labels_path, top_n, output_stream);
    }
}
} // namespace graph_utils
} // namespace arm_compute

#endif /* __ARM_COMPUTE_GRAPH_UTILS_H__ */
