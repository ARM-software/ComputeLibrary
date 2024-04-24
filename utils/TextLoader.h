#ifndef __UTILS_TEXT_LOADER_H__
#define __UTILS_TEXT_LOADER_H__

#include "arm_compute/core/Error.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"

#include "utils/Utils.h"
#include "utils/GraphUtils.h"

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

            std::tie(_length)  = parse_txt_header(_fs);

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

        unsigned char c = 0;

        /* read input from text data feeder */
        try
        {
            Window window;
            window.set(Window::DimX, Window::Dimension(0,_length,1));
            Iterator out(&text,window);
            execute_window_loop(
                window,
                [&](const Coordinates &)
                {
                    c = _feeder->get();
                    *out.ptr() = c;
                },
                out
            );
        }
        catch (const std::ifstream::failure &e)
        {
            ARM_COMPUTE_ERROR_VAR("Loading text file: %s", e.what());
        }
    }
private:
    std::ifstream _fs;
};

class TokenLoader : public ITextLoader
{
public:
    /** Default constructor */
    TokenLoader(): ITextLoader(), _fs()
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

            std::tie(_length)  = parse_txt_header(_fs);

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
    /** Fill an text tensor with the preprocessed token id  of the currently open text file.
     *
     * @param[in,out] text Text tensor to fill (Must be allocated, and of matching dimensions with the opened text file).
     */
    template <typename T>
    void fill_token_id(T &tensor, const std::string &vocab)
    {
        ARM_COMPUTE_ERROR_ON(!is_open());
        ARM_COMPUTE_ERROR_ON(_feeder.get() == nullptr);

        unsigned char c = 0;

        /* read input from text data feeder */
        try
        {
            std::vector<unsigned int> token_id;
            std::string input;
            for(int i = 0;i < _length ; i++)
            {
                input+= _feeder->get();
            }
            std::cout << input << std::endl;
            Window window;
            window.set(Window::DimX, Window::Dimension(0,_length,1));
            Iterator out(&tensor,window);
            execute_window_loop(
                window,
                [&](const Coordinates &)
                {
                    c = _feeder->get();
                    *out.ptr() = c;
                },
                out
            );
        }
        catch (const std::ifstream::failure &e)
        {
            ARM_COMPUTE_ERROR_VAR("Loading text file: %s", e.what());
        }
    }
private:
    void preprocess(ITensor &tensor)
    {
        /* Currently only support UTF-8 */
        if (tensor.info()->data_type() == DataType::F32)
        {
            const char pad_token[]          = u8"[PAD]";
            const char start_token[]        = u8"[CLS]";
            const char end_token[]          = u8"[SEP]";
            const char divide_helper[]      = u8" ";
            get_token<char,const char *,const char *,const char *>(tensor,std::move(pad_token),
                                                                        std::move(start_token),
                                                                        std::move(end_token),
                                                                        std::move(divide_helper));
        }
        else if (tensor.info()->data_type() == DataType::F16)
        {
            const char pad_token[]          = u8"[PAD]";
            const char start_token[]        = u8"[CLS]";
            const char end_token[]          = u8"[SEP]";
            const char divide_helper[]      = u8" ";
            get_token<char,const char *,const char *,const char *>(tensor,std::move(pad_token),
                                                                        std::move(start_token),
                                                                        std::move(end_token),
                                                                        std::move(divide_helper));
        }
        else if (tensor.info()->data_type() == DataType::U8)
        {
            const char pad_token[]          = u8"[PAD]";
            const char start_token[]        = u8"[CLS]";
            const char end_token[]          = u8"[SEP]";
            const char divide_helper[]      = u8" ";
            get_token<char,const char *,const char *,const char *>(tensor,std::move(pad_token),
                                                                        std::move(start_token),
                                                                        std::move(end_token),
                                                                        std::move(divide_helper));
        }
        else
        {
            ARM_COMPUTE_ERROR("NOT SUPPORTED!");
        }
    }

    template <typename T, typename... Args>
    void get_token(ITensor &tensor,Args &&... tokens)
    {
        const T * pad_token     = reinterpret_cast<const T *>(get_nth_elm<0>(tokens...));
        const T * start_token   = reinterpret_cast<const T *>(get_nth_elm<1>(tokens...));
        const T * end_token     = reinterpret_cast<const T *>(get_nth_elm<2>(tokens...));
        ARM_COMPUTE_UNUSED(pad_token);

        /** Read in */
        std::basic_string<T> buffer;
        Window window;
        window.set(Window::DimX, Window::Dimension(0,tensor.info()->dimension(0),1)); // Padding offset
        execute_window_loop(window,
                            [&](const Coordinates id){
                                buffer+= *reinterpret_cast<T *>(tensor.ptr_to_element(id));
                            });
        
        /** Sepreate into tokens and look up vocab list */
        std::map<std::basic_string<T>,int> token2id = get_token2id(_vocab_file);
        std::vector<unsigned int> text_ids;
        std::vector<std::basic_string<T>> tokens_vec;

        /* Split the text into words */
        std::basic_string<T> pat = R"([[:punct:]]|[[:alpha:]]+|[[:digit:]]+)";
        std::regex re(pat);
        std::smatch m;

        while (std::regex_search(buffer, m, re))
        {
            for (std::basic_string<T> x : m)
            {
                tokens_vec.push_back(x);
            }
            buffer = m.suffix();
        }

        // [CLS]
        text_ids.push_back(token2id[start_token]);
        
        // Input content
        find_longest_matching<T>(tokens_vec, token2id, text_ids);

        // [SEP]
        text_ids.push_back(token2id[end_token]);
        for(auto i : text_ids)std::cout << i << std::endl;
        /** Write back */
        tensor.info()->set_valid_region(tensor.info()->valid_region().set(0,0,text_ids.size()));
        window.use_tensor_dimensions(tensor.info()->tensor_shape());
        execute_window_loop(window,
                            [&](const Coordinates id){
                                *reinterpret_cast<unsigned int *>(tensor.ptr_to_element(id)) = text_ids[id[0]]; //Using dimesion x
                            });

    }
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