/*
 * Copyright (c) 2018-2021 Arm Limited.
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
#ifndef __UTILS_IMAGE_LOADER_H__
#define __UTILS_IMAGE_LOADER_H__

#include "arm_compute/core/Error.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"

#include "utils/Utils.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wswitch-default"
#pragma GCC diagnostic ignored "-Wstrict-overflow"
#include "stb/stb_image.h"
#pragma GCC diagnostic pop

#include <cstdlib>
#include <memory>
#include <string>

namespace arm_compute
{
namespace utils
{
/** Image feeder interface */
class IImageDataFeeder
{
public:
    /** Virtual base destructor */
    virtual ~IImageDataFeeder() = default;
    /** Gets a character from an image feed */
    virtual uint8_t get() = 0;
    /** Feed a whole row to a destination pointer
     *
     * @param[out] dst      Destination pointer
     * @param[in]  row_size Row size in terms of bytes
     */
    virtual void get_row(uint8_t *dst, size_t row_size) = 0;
};
/** File Image feeder concrete implementation */
class FileImageFeeder : public IImageDataFeeder
{
public:
    /** Default constructor
     *
     * @param[in] fs Image file stream
     */
    FileImageFeeder(std::ifstream &fs)
        : _fs(fs)
    {
    }
    // Inherited overridden methods
    uint8_t get() override
    {
        return _fs.get();
    }
    void get_row(uint8_t *dst, size_t row_size) override
    {
        ARM_COMPUTE_ERROR_ON(dst == nullptr);
        _fs.read(reinterpret_cast<std::fstream::char_type *>(dst), row_size);
    }

private:
    std::ifstream &_fs;
};
/** Memory Image feeder concrete implementation */
class MemoryImageFeeder : public IImageDataFeeder
{
public:
    /** Default constructor
     *
     * @param[in] data Pointer to data
     */
    MemoryImageFeeder(const uint8_t *data)
        : _data(data)
    {
    }
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    MemoryImageFeeder(const MemoryImageFeeder &) = delete;
    /** Default move constructor */
    MemoryImageFeeder(MemoryImageFeeder &&) = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    MemoryImageFeeder &operator=(const MemoryImageFeeder &) = delete;
    /** Default move assignment operator */
    MemoryImageFeeder &operator=(MemoryImageFeeder &&) = default;
    // Inherited overridden methods
    uint8_t get() override
    {
        return *_data++;
    }
    void get_row(uint8_t *dst, size_t row_size) override
    {
        ARM_COMPUTE_ERROR_ON(dst == nullptr);
        memcpy(dst, _data, row_size);
        _data += row_size;
    }

private:
    const uint8_t *_data;
};

/** Image loader interface */
class IImageLoader
{
public:
    /** Default Constructor */
    IImageLoader()
        : _feeder(nullptr), _width(0), _height(0)
    {
    }
    /** Virtual base destructor */
    virtual ~IImageLoader() = default;
    /** Return the width of the currently open image file. */
    unsigned int width() const
    {
        return _width;
    }
    /** Return the height of the currently open image file. */
    unsigned int height() const
    {
        return _height;
    }
    /** Return true if the image file is currently open */
    virtual bool is_open() = 0;
    /** Open an image file and reads its metadata (Width, height)
     *
     * @param[in] filename File to open
     */
    virtual void open(const std::string &filename) = 0;
    /** Closes an image file */
    virtual void close() = 0;
    /** Initialise an image's metadata with the dimensions of the image file currently open
     *
     * @param[out] image  Image to initialise
     * @param[in]  format Format to use for the image (Must be RGB888 or U8)
     */
    template <typename T>
    void init_image(T &image, Format format)
    {
        ARM_COMPUTE_ERROR_ON(!is_open());
        ARM_COMPUTE_ERROR_ON(format != Format::RGB888 && format != Format::U8);

        // Use the size of the input image
        TensorInfo image_info(_width, _height, format);
        image.allocator()->init(image_info);
    }
    /** Fill an image with the content of the currently open image file.
     *
     * @note If the image is a CLImage, the function maps and unmaps the image
     *
     * @param[in,out] image Image to fill (Must be allocated, and of matching dimensions with the opened image file).
     */
    template <typename T>
    void fill_image(T &image)
    {
        ARM_COMPUTE_ERROR_ON(!is_open());
        ARM_COMPUTE_ERROR_ON(image.info()->dimension(0) != _width || image.info()->dimension(1) != _height);
        ARM_COMPUTE_ERROR_ON_FORMAT_NOT_IN(&image, Format::U8, Format::RGB888);
        ARM_COMPUTE_ERROR_ON(_feeder.get() == nullptr);
        try
        {
            // Map buffer if creating a CLTensor
            map(image, true);

            // Validate feeding data
            validate_info(image.info());

            switch(image.info()->format())
            {
                case Format::U8:
                {
                    // We need to convert the data from RGB to grayscale:
                    // Iterate through every pixel of the image
                    Window window;
                    window.set(Window::DimX, Window::Dimension(0, _width, 1));
                    window.set(Window::DimY, Window::Dimension(0, _height, 1));

                    Iterator out(&image, window);

                    unsigned char red   = 0;
                    unsigned char green = 0;
                    unsigned char blue  = 0;

                    execute_window_loop(window, [&](const Coordinates &)
                    {
                        red   = _feeder->get();
                        green = _feeder->get();
                        blue  = _feeder->get();

                        *out.ptr() = 0.2126f * red + 0.7152f * green + 0.0722f * blue;
                    },
                    out);

                    break;
                }
                case Format::RGB888:
                {
                    // There is no format conversion needed: we can simply copy the content of the input file to the image one row at the time.
                    // Create a vertical window to iterate through the image's rows:
                    Window window;
                    window.set(Window::DimY, Window::Dimension(0, _height, 1));

                    Iterator out(&image, window);
                    size_t   row_size = _width * image.info()->element_size();

                    execute_window_loop(window, [&](const Coordinates &)
                    {
                        _feeder->get_row(out.ptr(), row_size);
                    },
                    out);

                    break;
                }
                default:
                    ARM_COMPUTE_ERROR("Unsupported format");
            }

            // Unmap buffer if creating a CLTensor
            unmap(image);
        }
        catch(const std::ifstream::failure &e)
        {
            ARM_COMPUTE_ERROR_VAR("Loading image file: %s", e.what());
        }
    }
    /** Fill a tensor with 3 planes (one for each channel) with the content of the currently open image file.
     *
     * @note If the image is a CLImage, the function maps and unmaps the image
     *
     * @param[in,out] tensor Tensor with 3 planes to fill (Must be allocated, and of matching dimensions with the opened image). Data types supported: U8/F16/F32
     * @param[in]     bgr    (Optional) Fill the first plane with blue channel (default = false)
     */
    template <typename T>
    void fill_planar_tensor(T &tensor, bool bgr = false)
    {
        ARM_COMPUTE_ERROR_ON(!is_open());
        ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(&tensor, 1, DataType::U8, DataType::QASYMM8, DataType::F32, DataType::F16);

        const DataLayout  data_layout  = tensor.info()->data_layout();
        const TensorShape tensor_shape = tensor.info()->tensor_shape();

        ARM_COMPUTE_UNUSED(tensor_shape);
        ARM_COMPUTE_ERROR_ON(tensor_shape[get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH)] != _width);
        ARM_COMPUTE_ERROR_ON(tensor_shape[get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT)] != _height);
        ARM_COMPUTE_ERROR_ON(tensor_shape[get_data_layout_dimension_index(data_layout, DataLayoutDimension::CHANNEL)] != 3);

        ARM_COMPUTE_ERROR_ON(_feeder.get() == nullptr);

        try
        {
            // Map buffer if creating a CLTensor
            map(tensor, true);

            // Validate feeding data
            validate_info(tensor.info());

            // Stride across channels
            size_t stride_z = 0;

            // Iterate through every pixel of the image
            Window window;
            if(data_layout == DataLayout::NCHW)
            {
                window.set(Window::DimX, Window::Dimension(0, _width, 1));
                window.set(Window::DimY, Window::Dimension(0, _height, 1));
                window.set(Window::DimZ, Window::Dimension(0, 1, 1));
                stride_z = tensor.info()->strides_in_bytes()[2];
            }
            else
            {
                window.set(Window::DimX, Window::Dimension(0, 1, 1));
                window.set(Window::DimY, Window::Dimension(0, _width, 1));
                window.set(Window::DimZ, Window::Dimension(0, _height, 1));
                stride_z = tensor.info()->strides_in_bytes()[0];
            }

            Iterator out(&tensor, window);

            unsigned char red   = 0;
            unsigned char green = 0;
            unsigned char blue  = 0;

            execute_window_loop(window, [&](const Coordinates &)
            {
                red   = _feeder->get();
                green = _feeder->get();
                blue  = _feeder->get();

                switch(tensor.info()->data_type())
                {
                    case DataType::U8:
                    case DataType::QASYMM8:
                    {
                        *(out.ptr() + 0 * stride_z) = bgr ? blue : red;
                        *(out.ptr() + 1 * stride_z) = green;
                        *(out.ptr() + 2 * stride_z) = bgr ? red : blue;
                        break;
                    }
                    case DataType::F32:
                    {
                        *reinterpret_cast<float *>(out.ptr() + 0 * stride_z) = static_cast<float>(bgr ? blue : red);
                        *reinterpret_cast<float *>(out.ptr() + 1 * stride_z) = static_cast<float>(green);
                        *reinterpret_cast<float *>(out.ptr() + 2 * stride_z) = static_cast<float>(bgr ? red : blue);
                        break;
                    }
                    case DataType::F16:
                    {
                        *reinterpret_cast<half *>(out.ptr() + 0 * stride_z) = static_cast<half>(bgr ? blue : red);
                        *reinterpret_cast<half *>(out.ptr() + 1 * stride_z) = static_cast<half>(green);
                        *reinterpret_cast<half *>(out.ptr() + 2 * stride_z) = static_cast<half>(bgr ? red : blue);
                        break;
                    }
                    default:
                    {
                        ARM_COMPUTE_ERROR("Unsupported data type");
                    }
                }
            },
            out);

            // Unmap buffer if creating a CLTensor
            unmap(tensor);
        }
        catch(const std::ifstream::failure &e)
        {
            ARM_COMPUTE_ERROR_VAR("Loading image file: %s", e.what());
        }
    }

protected:
    /** Validate metadata */
    virtual void validate_info(const ITensorInfo *tensor_info)
    {
        ARM_COMPUTE_UNUSED(tensor_info);
    }

protected:
    std::unique_ptr<IImageDataFeeder> _feeder;
    unsigned int                      _width;
    unsigned int                      _height;
};

/** PPM Image loader concrete implementation */
class PPMLoader : public IImageLoader
{
public:
    /** Default Constructor */
    PPMLoader()
        : IImageLoader(), _fs()
    {
    }

    // Inherited methods overridden:
    bool is_open() override
    {
        return _fs.is_open();
    }
    void open(const std::string &filename) override
    {
        ARM_COMPUTE_ERROR_ON(is_open());
        try
        {
            _fs.exceptions(std::ifstream::failbit | std::ifstream::badbit);
            _fs.open(filename, std::ios::in | std::ios::binary);

            unsigned int max_val = 0;
            std::tie(_width, _height, max_val) = parse_ppm_header(_fs);

            ARM_COMPUTE_ERROR_ON_MSG_VAR(max_val >= 256, "2 bytes per colour channel not supported in file %s",
                                         filename.c_str());

            _feeder = std::make_unique<FileImageFeeder>(_fs);
        }
        catch(std::runtime_error &e)
        {
            ARM_COMPUTE_ERROR_VAR("Accessing %s: %s", filename.c_str(), e.what());
        }
    }
    void close() override
    {
        if(is_open())
        {
            _fs.close();
            _feeder = nullptr;
        }
        ARM_COMPUTE_ERROR_ON(is_open());
    }

protected:
    // Inherited methods overridden:
    void validate_info(const ITensorInfo *tensor_info) override
    {
        // Check if the file is large enough to fill the image
        const size_t current_position = _fs.tellg();
        _fs.seekg(0, std::ios_base::end);
        const size_t end_position = _fs.tellg();
        _fs.seekg(current_position, std::ios_base::beg);

        ARM_COMPUTE_ERROR_ON_MSG((end_position - current_position) < tensor_info->tensor_shape().total_size(),
                                 "Not enough data in file");
        ARM_COMPUTE_UNUSED(end_position, tensor_info);
    }

private:
    std::ifstream _fs;
};

/** Class to load the content of a JPEG file into an Image */
class JPEGLoader : public IImageLoader
{
private:
    /** Custom malloc deleter struct */
    struct malloc_deleter
    {
        void operator()(uint8_t *p) const
        {
            free(p);
        }
    };

public:
    /** Default Constructor */
    JPEGLoader()
        : IImageLoader(), _is_loaded(false), _data(nullptr)
    {
    }

    // Inherited methods overridden:
    bool is_open() override
    {
        return _is_loaded;
    }
    void open(const std::string &filename) override
    {
        int      bpp, width, height;
        uint8_t *rgb_image = stbi_load(filename.c_str(), &width, &height, &bpp, 3);
        if(rgb_image == NULL)
        {
            ARM_COMPUTE_ERROR_VAR("Accessing %s failed", filename.c_str());
        }
        else
        {
            _width     = width;
            _height    = height;
            _data      = std::unique_ptr<uint8_t, malloc_deleter>(rgb_image);
            _is_loaded = true;
            _feeder    = std::make_unique<MemoryImageFeeder>(_data.get());
        }
    }
    void close() override
    {
        if(is_open())
        {
            _width  = 0;
            _height = 0;
            release();
        }
        ARM_COMPUTE_ERROR_ON(is_open());
    }
    /** Explicitly Releases the memory of the loaded data */
    void release()
    {
        if(_is_loaded)
        {
            _data.reset();
            _is_loaded = false;
            _feeder    = nullptr;
        }
    }

private:
    bool _is_loaded;
    std::unique_ptr<uint8_t, malloc_deleter> _data;
};

/** Factory for generating appropriate image loader**/
class ImageLoaderFactory final
{
public:
    /** Create an image loader depending on the image type
     *
     * @param[in] filename File than needs to be loaded
     *
     * @return Image loader
     */
    static std::unique_ptr<IImageLoader> create(const std::string &filename)
    {
        ImageType type = arm_compute::utils::get_image_type_from_file(filename);
        switch(type)
        {
            case ImageType::PPM:
                return std::make_unique<PPMLoader>();
            case ImageType::JPEG:
                return std::make_unique<JPEGLoader>();
            case ImageType::UNKNOWN:
            default:
                return nullptr;
        }
    }
};
} // namespace utils
} // namespace arm_compute
#endif /* __UTILS_IMAGE_LOADER_H__*/
