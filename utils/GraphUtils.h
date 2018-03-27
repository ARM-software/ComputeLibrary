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

#include "arm_compute/core/CL/OpenCL.h"

#include "arm_compute/graph2/Types.h"

#include <array>
#include <random>
#include <string>
#include <vector>

namespace arm_compute
{
namespace graph_utils
{
/** Preprocessor interface **/
class IPreprocessor
{
public:
    /** Default destructor. */
    virtual ~IPreprocessor() = default;
    /** Preprocess the given tensor.
     *
     * @param[in] tensor Tensor to preprocess.
     */
    virtual void preprocess(ITensor &tensor) = 0;
};

/** Caffe preproccessor */
class CaffePreproccessor : public IPreprocessor
{
public:
    /** Default Constructor
     *
     * @param mean Mean array in RGB ordering
     * @param bgr  Boolean specifying if the preprocessing should assume BGR format
     */
    CaffePreproccessor(std::array<float, 3> mean = std::array<float, 3> { { 0, 0, 0 } }, bool bgr = true);
    void preprocess(ITensor &tensor) override;

private:
    std::array<float, 3> _mean;
    bool _bgr;
};

/** TF preproccessor */
class TFPreproccessor : public IPreprocessor
{
public:
    void preprocess(ITensor &tensor) override;
};

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
     * @param[in] ppm_path     Path to PPM file
     * @param[in] bgr          (Optional) Fill the first plane with blue channel (default = false)
     * @param[in] preprocessor (Optional) PPM pre-processing object
     */
    PPMAccessor(std::string ppm_path, bool bgr = true, std::unique_ptr<IPreprocessor> preprocessor = nullptr);
    /** Allow instances of this class to be move constructed */
    PPMAccessor(PPMAccessor &&) = default;

    // Inherited methods overriden:
    bool access_tensor(ITensor &tensor) override;

private:
    const std::string              _ppm_path;
    const bool                     _bgr;
    std::unique_ptr<IPreprocessor> _preprocessor;
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
 * @param[in] ppm_path     Path to PPM file
 * @param[in] preprocessor Preproccessor object
 * @param[in] bgr          (Optional) Fill the first plane with blue channel (default = true)
 *
 * @return An appropriate tensor accessor
 */
inline std::unique_ptr<graph::ITensorAccessor> get_input_accessor(const std::string             &ppm_path,
                                                                  std::unique_ptr<IPreprocessor> preprocessor = nullptr,
                                                                  bool                           bgr          = true)
{
    if(ppm_path.empty())
    {
        return arm_compute::support::cpp14::make_unique<DummyAccessor>();
    }
    else
    {
        return arm_compute::support::cpp14::make_unique<PPMAccessor>(ppm_path, bgr, std::move(preprocessor));
    }
}

/** Utility function to return the TargetHint
 *
 * @param[in] target Integer value which expresses the selected target. Must be 0 for NEON, 1 for OpenCL or 2 for OpenCL with Tuner
 *
 * @return the TargetHint
 */
inline graph::TargetHint set_target_hint(int target)
{
    ARM_COMPUTE_ERROR_ON_MSG(target > 2, "Invalid target. Target must be 0 (NEON), 1 (OpenCL) or 2 (OpenCL with Tuner)");
    if((target == 1 || target == 2) && graph::Graph::opencl_is_available())
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

/** Utility function to return the TargetHint
 *
 * @param[in] target Integer value which expresses the selected target. Must be 0 for NEON or 1 for OpenCL or 2 (OpenCL with Tuner)
 *
 * @return the TargetHint
 */
inline graph2::Target set_target_hint2(int target)
{
    ARM_COMPUTE_ERROR_ON_MSG(target > 3, "Invalid target. Target must be 0 (NEON), 1 (OpenCL), 2 (OpenCL + Tuner), 3 (GLES)");
    if((target == 1 || target == 2) && arm_compute::opencl_is_available())
    {
        // If type of target is OpenCL, check if OpenCL is available and initialize the scheduler
        return graph2::Target::CL;
    }
    else if(target == 3)
    {
        return graph2::Target::GC;
    }
    else
    {
        return graph2::Target::NEON;
    }
}
} // namespace graph_utils
} // namespace arm_compute

#endif /* __ARM_COMPUTE_GRAPH_UTILS_H__ */
