/*
   Copyright 2017 Leon Merten Lohse

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
   SOFTWARE.
*/

#include <complex>
#include <fstream>
#include <string>
#include <iostream>
#include <sstream>
#include <cstdint>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <regex>


namespace npy {

/* Compile-time test for byte order.
   If your compiler does not define these per default, you may want to define
   one of these constants manually. 
   Defaults to little endian order. */
#if defined(__BYTE_ORDER) && __BYTE_ORDER == __BIG_ENDIAN || \
    defined(__BIG_ENDIAN__) || \
    defined(__ARMEB__) || \
    defined(__THUMBEB__) || \
    defined(__AARCH64EB__) || \
    defined(_MIBSEB) || defined(__MIBSEB) || defined(__MIBSEB__)
const bool big_endian = true;
#else
const bool big_endian = false;
#endif


const char magic_string[] = "\x93NUMPY";
const size_t magic_string_length = 6;

const char little_endian_char = '<';
const char big_endian_char = '>';
const char no_endian_char = '|';

constexpr char host_endian_char = ( big_endian ? 
    big_endian_char : 
    little_endian_char );

inline void write_magic(std::ostream& ostream, unsigned char v_major=1, unsigned char v_minor=0) {
  ostream.write(magic_string, magic_string_length);
  ostream.put(v_major);
  ostream.put(v_minor);
}

inline void read_magic(std::istream& istream, unsigned char *v_major, unsigned char *v_minor) {
  char *buf = new char[magic_string_length+2];
  istream.read(buf, magic_string_length+2);

  if(!istream) {
      throw std::runtime_error("io error: failed reading file");
  }

  for (size_t i=0; i < magic_string_length; i++) {
    if(buf[i] != magic_string[i]) {
      throw std::runtime_error("this file do not have a valid npy format.");
    }
  }

  *v_major = buf[magic_string_length];
  *v_minor = buf[magic_string_length+1];
  delete[] buf;
}

// typestring magic
struct Typestring {
  private:
    char c_endian;
    char c_type;
    int  len;

  public:
    inline std::string str() {
      const size_t max_buflen = 16;
      char buf[max_buflen];
      std::sprintf(buf, "%c%c%u", c_endian, c_type, len);
      return std::string(buf);
    }

    Typestring(std::vector<float>& v) :c_endian {host_endian_char}, c_type {'f'}, len {sizeof(float)} {}
    Typestring(std::vector<double>& v) :c_endian {host_endian_char}, c_type {'f'}, len {sizeof(double)} {}
    Typestring(std::vector<long double>& v) :c_endian {host_endian_char}, c_type {'f'}, len {sizeof(long double)} {}

    Typestring(std::vector<char>& v) :c_endian {no_endian_char}, c_type {'i'}, len {sizeof(char)} {}
    Typestring(std::vector<short>& v) :c_endian {host_endian_char}, c_type {'i'}, len {sizeof(short)} {}
    Typestring(std::vector<int>& v) :c_endian {host_endian_char}, c_type {'i'}, len {sizeof(int)} {}
    Typestring(std::vector<long>& v) :c_endian {host_endian_char}, c_type {'i'}, len {sizeof(long)} {}
    Typestring(std::vector<long long>& v) :c_endian {host_endian_char}, c_type {'i'}, len {sizeof(long long)} {}

    Typestring(std::vector<unsigned char>& v) :c_endian {no_endian_char}, c_type {'u'}, len {sizeof(unsigned char)} {}
    Typestring(std::vector<unsigned short>& v) :c_endian {host_endian_char}, c_type {'u'}, len {sizeof(unsigned short)} {}
    Typestring(std::vector<unsigned int>& v) :c_endian {host_endian_char}, c_type {'u'}, len {sizeof(unsigned int)} {}
    Typestring(std::vector<unsigned long>& v) :c_endian {host_endian_char}, c_type {'u'}, len {sizeof(unsigned long)} {}
    Typestring(std::vector<unsigned long long>& v) :c_endian {host_endian_char}, c_type {'u'}, len {sizeof(unsigned long long)} {}

    Typestring(std::vector<std::complex<float>>& v) :c_endian {host_endian_char}, c_type {'c'}, len {sizeof(std::complex<float>)} {}
    Typestring(std::vector<std::complex<double>>& v) :c_endian {host_endian_char}, c_type {'c'}, len {sizeof(std::complex<double>)} {}
    Typestring(std::vector<std::complex<long double>>& v) :c_endian {host_endian_char}, c_type {'c'}, len {sizeof(std::complex<long double>)} {}
};

inline void parse_typestring( std::string typestring){
  std::regex re ("'([<>|])([ifuc])(\\d+)'");
  std::smatch sm;

  std::regex_match(typestring, sm, re );

  if ( sm.size() != 4 ) {
    throw std::runtime_error("invalid typestring");
  }
}

/* Helpers for the improvised parser */
inline std::string unwrap_s(std::string s, char delim_front, char delim_back) {
  if ((s.back() == delim_back) && (s.front() == delim_front))
    return s.substr(1, s.length()-2);
  else
    throw std::runtime_error("unable to unwrap");
}

inline std::string get_value_from_map(std::string mapstr) {
  size_t sep_pos = mapstr.find_first_of(":");
  if (sep_pos == std::string::npos)
    return "";

  return mapstr.substr(sep_pos+1);
}

inline void pop_char(std::string& s, char c) {
  if (s.back() == c)
    s.pop_back();
}

inline void ParseHeader(std::string header, std::string& descr, bool *fortran_order, std::vector<unsigned long>& shape) {
  /*
     The first 6 bytes are a magic string: exactly "x93NUMPY".

     The next 1 byte is an unsigned byte: the major version number of the file format, e.g. x01.

     The next 1 byte is an unsigned byte: the minor version number of the file format, e.g. x00. Note: the version of the file format is not tied to the version of the numpy package.

     The next 2 bytes form a little-endian unsigned short int: the length of the header data HEADER_LEN.

     The next HEADER_LEN bytes form the header data describing the array's format. It is an ASCII string which contains a Python literal expression of a dictionary. It is terminated by a newline ('n') and padded with spaces ('x20') to make the total length of the magic string + 4 + HEADER_LEN be evenly divisible by 16 for alignment purposes.

     The dictionary contains three keys:

     "descr" : dtype.descr
     An object that can be passed as an argument to the numpy.dtype() constructor to create the array's dtype.
     "fortran_order" : bool
     Whether the array data is Fortran-contiguous or not. Since Fortran-contiguous arrays are a common form of non-C-contiguity, we allow them to be written directly to disk for efficiency.
     "shape" : tuple of int
     The shape of the array.
     For repeatability and readability, this dictionary is formatted using pprint.pformat() so the keys are in alphabetic order.
   */

  // remove trailing newline
  if (header.back() != '\n')
    throw std::runtime_error("invalid header");
  header.pop_back();

  // remove all whitespaces
  header.erase(std::remove(header.begin(), header.end(), ' '), header.end());

  // unwrap dictionary
  header = unwrap_s(header, '{', '}');

  // find the positions of the 3 dictionary keys
  size_t keypos_descr = header.find("'descr'");
  size_t keypos_fortran = header.find("'fortran_order'");
  size_t keypos_shape = header.find("'shape'");

  // make sure all the keys are present
  if (keypos_descr == std::string::npos)
    throw std::runtime_error("missing 'descr' key");
  if (keypos_fortran == std::string::npos)
    throw std::runtime_error("missing 'fortran_order' key");
  if (keypos_shape == std::string::npos)
    throw std::runtime_error("missing 'shape' key");

  // Make sure the keys are in order.
  // Note that this violates the standard, which states that readers *must* not 
  // depend on the correct order here.
  // TODO: fix
  if (keypos_descr >= keypos_fortran || keypos_fortran >= keypos_shape)
    throw std::runtime_error("header keys in wrong order");

  // get the 3 key-value pairs
  std::string keyvalue_descr;
  keyvalue_descr = header.substr(keypos_descr, keypos_fortran - keypos_descr);
  pop_char(keyvalue_descr, ',');

  std::string keyvalue_fortran;
  keyvalue_fortran = header.substr(keypos_fortran, keypos_shape - keypos_fortran);
  pop_char(keyvalue_fortran, ',');

  std::string keyvalue_shape;
  keyvalue_shape = header.substr(keypos_shape, std::string::npos);
  pop_char(keyvalue_shape, ',');

  // get the values (right side of `:')
  std::string descr_s = get_value_from_map(keyvalue_descr);
  std::string fortran_s = get_value_from_map(keyvalue_fortran);
  std::string shape_s = get_value_from_map(keyvalue_shape);

  parse_typestring(descr_s);
  descr = unwrap_s(descr_s, '\'', '\'');

  // convert literal Python bool to C++ bool
  if (fortran_s == "True")
    *fortran_order = true;
  else if (fortran_s == "False")
    *fortran_order = false;
  else
    throw std::runtime_error("invalid fortran_order value");

  // parse the shape Python tuple ( x, y, z,)

  // first clear the vector
  shape.clear();
  shape_s = unwrap_s(shape_s, '(', ')');

  // a tokenizer would be nice... 
  size_t pos = 0;
  size_t pos_next;
  for(;;) {
    pos_next = shape_s.find_first_of(',', pos);
    std::string dim_s;
    if (pos_next != std::string::npos)
      dim_s = shape_s.substr(pos, pos_next - pos);
    else
      dim_s = shape_s.substr(pos);
    pop_char(dim_s, ',');
    if (dim_s.length() == 0) {
      if (pos_next != std::string::npos)
        throw std::runtime_error("invalid shape");
    }else{
      std::stringstream ss;
      ss << dim_s;
      unsigned long tmp;
      ss >> tmp;
      shape.push_back(tmp);
    }
    if (pos_next != std::string::npos)
      pos = ++pos_next;
    else
      break;
  }
}

inline void WriteHeader(std::ostream& out, const std::string& descr, bool fortran_order, unsigned int n_dims, const unsigned long shape[])
{
    std::ostringstream ss_header;
    std::string s_fortran_order;
    if (fortran_order)
      s_fortran_order = "True";
    else
      s_fortran_order = "False";

    std::ostringstream ss_shape;
    ss_shape << "(";
    for (unsigned int n=0; n < n_dims; n++){
      ss_shape << shape[n] << ", ";
    }
    ss_shape << ")";

    ss_header << "{'descr': '" << descr << "', 'fortran_order': " << s_fortran_order << ", 'shape': " << ss_shape.str() << " }";

    size_t header_len_pre = ss_header.str().length() + 1;
    size_t metadata_len = magic_string_length + 2 + 2 + header_len_pre;

    unsigned char version[2] = {1, 0};
    if (metadata_len >= 255*255) {
      metadata_len = magic_string_length + 2 + 4 + header_len_pre;
      version[0] = 2;
      version[1] = 0;
    }
    size_t padding_len = 16 - metadata_len % 16;
    std::string padding (padding_len, ' ');
    ss_header << padding;
    ss_header << '\n';

    std::string header = ss_header.str();

    // write magic
    write_magic(out, version[0], version[1]);

    // write header length
    if (version[0] == 1 && version[1] == 0) {
      char header_len_le16[2];
      uint16_t header_len = header.length();

      header_len_le16[0] = (header_len >> 0) & 0xff;
      header_len_le16[1] = (header_len >> 8) & 0xff;
      out.write(reinterpret_cast<char *>(header_len_le16), 2);
    }else{
      char header_len_le32[4];
      uint32_t header_len = header.length();

      header_len_le32[0] = (header_len >> 0) & 0xff;
      header_len_le32[1] = (header_len >> 8) & 0xff;
      header_len_le32[2] = (header_len >> 16) & 0xff;
      header_len_le32[3] = (header_len >> 24) & 0xff;
      out.write(reinterpret_cast<char *>(header_len_le32), 4);
    }

    out << header;
}

inline std::string read_header_1_0(std::istream& istream) {
    // read header length and convert from little endian
    char header_len_le16[2];
    istream.read(header_len_le16, 2);

    uint16_t header_length = (header_len_le16[0] << 0) | (header_len_le16[1] << 8);

    if((magic_string_length + 2 + 2 + header_length) % 16 != 0) {
        // TODO: display warning
    }

    char *buf = new char[header_length];
    istream.read(buf, header_length);
    std::string header (buf, header_length);
    delete[] buf;

    return header;
}

inline std::string read_header_2_0(std::istream& istream) {
    // read header length and convert from little endian
    char header_len_le32[4];
    istream.read(header_len_le32, 4);

    uint32_t header_length = (header_len_le32[0] <<  0) | (header_len_le32[1] <<  8)
                           | (header_len_le32[2] << 16) | (header_len_le32[3] <<  24);

    if((magic_string_length + 2 + 4 + header_length) % 16 != 0) {
      // TODO: display warning
    }

    char *buf = new char[header_length];
    istream.read(buf, header_length);
    std::string header (buf, header_length);
    delete[] buf;

    return header;
}

template<typename Scalar>
void SaveArrayAsNumpy( const std::string& filename, bool fortran_order, unsigned int n_dims, const unsigned long shape[], const std::vector<Scalar>& data)
{
    Typestring typestring_o {data};
    std::string typestring = typestring_o.str();

    std::ofstream stream( filename, std::ofstream::binary);
    if(!stream) {
        throw std::runtime_error("io error: failed to open a file.");
    }
    WriteHeader(stream, typestring, fortran_order, n_dims, shape);

    size_t size = 1;
    for (unsigned int i=0; i<n_dims; ++i)
      size *= shape[i];
    stream.write(reinterpret_cast<const char*>(&data[0]), sizeof(Scalar) * size);
}


/**

 */
template<typename Scalar>
void LoadArrayFromNumpy(const std::string& filename, std::vector<unsigned long>& shape, std::vector<Scalar>& data)
{
    std::ifstream stream(filename, std::ifstream::binary);
    if(!stream) {
        throw std::runtime_error("io error: failed to open a file.");
    }
    // check magic bytes an version number
    unsigned char v_major, v_minor;
    read_magic(stream, &v_major, &v_minor);

    std::string header;

    if(v_major == 1 && v_minor == 0){
      header = read_header_1_0(stream);
    }else if(v_major == 2 && v_minor == 0) {
      header = read_header_2_0(stream);
    }else{
       throw std::runtime_error("unsupported file format version");
    }

    // parse header
    bool fortran_order;
    std::string typestr;

    ParseHeader(header, typestr, &fortran_order, shape);

    // check if the typestring matches the given one
    Typestring typestring_o {data};
    std::string expect_typestr = typestring_o.str();
    if (typestr != expect_typestr) {
      throw std::runtime_error("formatting error: typestrings not matching");
    }

    // compute the data size based on the shape
    size_t total_size = 1;
    for(size_t i=0; i<shape.size(); ++i) {
        total_size *= shape[i];
    }
    data.resize(total_size);

    // read the data
    stream.read(reinterpret_cast<char*>(&data[0]), sizeof(Scalar)*total_size);
}

} // namespace npy
