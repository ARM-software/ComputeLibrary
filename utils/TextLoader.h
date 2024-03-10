#ifndef __UTILS_TEXT_LOADER_H__
#define __UTILS_TEXT_LOADER_H__

#include "arm_compute/core/Error.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"

#include "utils/Utils.h"

#include <cstdlib>
#include <memory>
#include <string>

namespace arm_compute
{

namespace utils
{

/** Text feeder interface */
class ITextDataFeeder
{
public:
    /** Virtual base destructor */
    virtual ~ITextDataFeeder() = default;
    /** Gets a character from an image feed */
    virtual uint8_t get() = 0;
    /** Gets character count from the file*/
    virtual uint8_t get_count() = 0;
    /** Feed a whole chuck to a destination pointer
     *
     * @param[out] dst      Destination pointer
     * @param[in]  row_size Row size in terms of bytes
     */
    virtual void get_chuck(uint8_t *dst, size_t chuck_size) = 0;
};
/** File Text feeder concrete implementation */
class FileTextFeeder : public ITextDataFeeder
{
public:
    /** Default constructor
     *
     * @param[in] fs Text file stream
     */
    FileTextFeeder(std::ifstream &fs) : _fs(fs)
    {
    }
    // Inherited overridden methods
    uint8_t get() override
    {
        return _fs.get();
    }
    uint8_t get_count() override
    {
        return _fs.gcount();
    }
    void get_chuck(uint8_t *dst, size_t chuck_size) override
    {
        ARM_COMPUTE_ERROR_ON(dst == nullptr);
        _fs.read(reinterpret_cast<std::fstream::char_type *>(dst), chuck_size);
    }

private:
    std::ifstream &_fs;
};

/** Text Loader Interface */
class ITextLoader
{
public:
    /** Default constructor */
    ITextLoader(): _feeder(nullptr),  _length(0)
    {
    }
    /** Virtual base deconstructor */
    virtual ~ITextLoader() = default;
    /** Return the length of the currently open text file. */
    unsigned int length() const
    {
        return _length;
    }
    /** Return true if the text file is currently open */
    virtual bool is_open() = 0;
    /** Open an text file and reads its metadata (Length)
     *
     * @param[in] filename File to open
     */
    virtual void open(const std::string &filename) = 0;
    /** Closes an text file */
    virtual void close() = 0;
    /** Initialise an text's metadata with the length of the text file currently open
     *
     * @param[out] text   Text to initialise
     * @param[in]  format Format to use for the text (Currently Utf-8)
     */
    template <typename T>
    void init_text(T &text, TextFormat format)
    {
        ARM_COMPUTE_ERROR_ON(!is_open());
        ARM_COMPUTE_ERROR_ON(format != TextFormat::UTF8);

        // Use the size of the input text
        TensorInfo text_info(_length, format);
        text.allocator()->init(text_info);
    }
    /** Fill an text tensor with the content of the currently open text file.
     *
     * @param[in,out] text Text tensor to fill (Must be allocated, and of matching dimensions with the opened text file).
     */
    template <typename T>
    void fill_text(T &text)
    {
        ARM_COMPUTE_ERROR_ON(!is_open());
        ARM_COMPUTE_ERROR_ON(text.info()->dimension(0) != _length );
        ARM_COMPUTE_ERROR_ON_FORMAT_NOT_IN(&text, TextFormat::UTF8);
        ARM_COMPUTE_ERROR_ON(_feeder.get() == nullptr);

        //unsigned char c = 0;

        /* read input from text data feeder */
        try
        {
            std::cout << _length <<std::endl;
            //c = _feeder->get_count();
            std::cout << "Word conuts";
            std::cout << _feeder->get_count() <<std::endl;
        }
        catch (const std::ifstream::failure &e)
        {
            ARM_COMPUTE_ERROR_VAR("Loading text file: %s", e.what());
        }
    }
protected:
    std::unique_ptr<ITextDataFeeder> _feeder;
    unsigned int _length;
};
    
class UTF8Loader : public ITextLoader
{
public:
    /** Default constructor */
    UTF8Loader(): ITextLoader(), _fs()
    {
    }
    // Inherited methods overridden:
    bool is_open() override
    {
        return _fs.is_open();
    }
    void set_length(unsigned int length)
    {
        _length = length;
    }
    void open(const std::string &filename) override
    {
        ARM_COMPUTE_ERROR_ON(is_open());
        try
        {
            _fs.exceptions(std::ifstream::failbit | std::ifstream::badbit);
            _fs.open(filename, std::ios::in | std::ios::binary);

            _feeder = std::make_unique<FileTextFeeder>(_fs);
        }
        catch (std::runtime_error &e)
        {
            ARM_COMPUTE_ERROR_VAR("Accessing %s: %s", filename.c_str(), e.what());
        }
    }
    void close() override
    {
        if (is_open())
        {
            _fs.close();
            _feeder = nullptr;
        }
        ARM_COMPUTE_ERROR_ON(is_open());
    }
private:
    std::ifstream _fs;
};


/** Factory for generating appropriate text loader**/
class TextLoaderFactory final
{
public:
    /** Create an text loader depending on the text type
     *
     * @param[in] filename File than needs to be loaded
     *
     * @return Text loader
     */
    static std::unique_ptr<ITextLoader> create(const std::string &filename)
    {
        // TODO 
        TextType type = arm_compute::utils::get_text_type_from_file(filename);
        switch (type)
        {
            case TextType::UTF8:
                return std::make_unique<UTF8Loader>();
            case TextType::UNKNOWN:
            default:
                return nullptr;
        }
    }
};

} // namespace utils

} // namespace arm_compute

#endif /* __UTILS_TEXT_LOADER_H__ */