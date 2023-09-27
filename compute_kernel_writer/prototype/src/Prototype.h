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

#ifndef CKW_PROTOTYPE_SRC_PROTOTYPE_H
#define CKW_PROTOTYPE_SRC_PROTOTYPE_H

#include "ckw/Error.h"
#include "ckw/TensorInfo.h"
#include "ckw/types/ConvertPolicy.h"
#include "ckw/types/DataType.h"
#include "ckw/types/Functions.h"
#include "ckw/types/GpuTargetLanguage.h"
#include "ckw/types/Operators.h"
#include "ckw/types/TensorSamplerTypes.h"

#include <algorithm>
#include <array>
#include <cassert> // assert (to be removed)
#include <chrono>
#include <cmath>
#include <cstdint> // int32_t
#include <functional>
#include <iostream> // cout (to be removed)
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace ckw
{
namespace prototype
{

// Dummy data structure for Size2D
using Size2D = std::vector<int32_t>;

// Dummy Status
using Status = void;

enum class ComponentType : int32_t
{
    Complex   = 0,
    Simple    = 1,
    Unfusable = 2
};

enum class GpuCompilationSpeed
{
    Fast = 0x00, // fast compilation may increase the latency of the network
    Slow = 0x01  // slow compilation may decrease the latency of the network
};

enum class GpuExtensions
{
    Fp16,
    Dot8,
    Mmul,
    FastMath
};

struct TensorInfo
{
    TensorShape      shape{{0}};
    DataType         data_type{DataType::Unknown};
    TensorDataLayout data_layout{TensorDataLayout::Nhwc};
    int32_t          id{-1};
};

struct ComponentAttribute
{
    GpuCompilationSpeed compilation_speed{GpuCompilationSpeed::Fast};
    bool                overwrite_tile{true};
};

inline std::string data_type_to_cl_type(DataType dt)
{
    switch (dt)
    {
        case DataType::Fp32:
            return "float";
        case DataType::Fp16:
            return "half";
        case DataType::Int8:
            return "char";
        case DataType::Uint8:
            return "uchar";
        case DataType::Uint16:
            return "ushort";
        case DataType::Int16:
            return "short";
        case DataType::Uint32:
            return "uint";
        case DataType::Int32:
            return "int";
        case DataType::Bool:
            return "bool";
        default:
            assert(false);
            return "";
    }
}

inline int32_t width_to_cl_vector_size(int32_t width)
{
    switch (width)
    {
        case 1:
            return 1;
        case 2:
            return 2;
        case 3:
            return 3;
        case 4:
            return 4;
        case 5:
        case 6:
        case 7:
        case 8:
            return 8;
        case 9:
        case 10:
        case 11:
        case 12:
        case 13:
        case 14:
        case 15:
        case 16:
            return 16;
        default:
            assert(false);
            return 0;
    }
}

inline std::string get_cl_data_type(DataType dt, int32_t width)
{
    std::string data_type;
    int32_t     w = width_to_cl_vector_size(width);
    data_type += data_type_to_cl_type(dt);
    if (w != 1)
    {
        data_type += std::to_string(w);
    }
    return data_type;
}

inline std::string to_opencl_store(int32_t vector_length)
{
    if (vector_length != 1)
    {
        return "vstore" + std::to_string(vector_length) + "(";
    }
    else
    {
        return "*(";
    }
}

struct TileInfo
{
    TileInfo()
    {
    }

    TileInfo(DataType dt) : dt(dt), w(1), h(1)
    {
    }

    TileInfo(DataType dt, int32_t width) : dt(dt), w(width), h(1)
    {
    }

    TileInfo(DataType dt, int32_t width, int32_t height) : dt(dt), w(width), h(height)
    {
    }

    DataType dt{DataType::Unknown}; // Data type of the tile
    int32_t  w{0};                  // Width (i.e. c0 - portion of the channels)
    int32_t  h{0};                  // Height (i.e. s0 - portion of the spatial dimensions)
};

inline std::ostream &operator<<(std::ostream &o, const TileInfo &a)
{
    o << a.w << " x " << a.h;
    return o;
}

struct DataTypeAsString
{
    std::string str{""};
    DataType    dt{DataType::Unknown};
    int32_t     size{1};
};

struct ValueAsString
{
    std::string      str{""};
    DataTypeAsString type{};
};

// https://stackoverflow.com/questions/51515378/storing-and-accessing-tile-properties-in-c
// A Tile is a collection of variables used to express a 2D data.
class IScalarTile
{
public:
    virtual ~IScalarTile() = default;

    /** Method to get the scalar variable from a tile
     * @param[in] x X coordinate on the width of the tile. If out-of-bound, the coordinate is clamped to the nearest valid edge
     * @param[in] y Y coordinate on the height of the tile. If out-of-bound, the coordinate is clamped to the nearest valid edge
     *
     * @return the scalar variable as a string
     */
    virtual ValueAsString scalar(int32_t x, int32_t y) const = 0;

    /** Method to get the list of underlying variable names used by the tile
     *
     * @return the list of variable names
     */
    virtual std::vector<ValueAsString> underlying_source_variables() const = 0;

    /** Method to get the name of the tile.
     *
     * @return the name of the tile
     */
    std::string name() const
    {
        return _basename;
    }

    /** Method to get the tile format
     *
     * @return the format
     */
    TileInfo format() const
    {
        return _format;
    }

    /** Method to know whether the tile is assignable or not (constant)
     *
     * @return true if the tile is assignable
     */
    virtual bool is_assignable() const = 0;

    /** Method to know whether the tile needs to be declared
     *
     * @return true if the tile needs to be declared in the code before being used
     */
    virtual bool need_declaration() const = 0;

protected:
    TileInfo    _format{};     // Tile format
    std::string _basename{""}; // Tile name
};

// A tile is a collection of variables used to express a 2D data. The variables are vectors in the GPU context.
// The vector size is given by the width of the tile. The number of vectors height by depth defines the number of vectors
class IVectorTile : public IScalarTile
{
public:
    virtual ~IVectorTile() = default;

    /** Method to get the vector variable from a tile. A vector is an ordered homogeneous collection of two or more scalars.
     *  The user can query the list of supported width for the vectors through preferred_vector_sizes().
     *
     * @param[in] y Y coordinate on the height of the tile. If out-of-bound, the coordinate is clamped to the nearest valid edge
     *
     * @return the vector variable as a string
     */
    virtual ValueAsString vector(int32_t y) const = 0;

    /** Method to get a vector variable from a tile. A vector is an ordered homogeneous collection of two or more scalars.
     *
     * @return the vector variable as a string
     */
    virtual ValueAsString vector(int32_t x_start, int32_t width, int32_t y) const = 0;
    /** Method to get the preferred vector sizes.
     *
     * @return a vector with the preferred vector sizes
     */
    //virtual std::vector<int32_t> preferred_vector_sizes() const = 0;
};

class ClTile : public IVectorTile
{
public:
    ClTile(const std::string &name, TileInfo format)
    {
        _format   = format;
        _basename = name;
    }

    ValueAsString scalar(int32_t x, int32_t y) const override
    {
        x = std::max(std::min(x, _format.w - 1), static_cast<int32_t>(0));
        y = std::max(std::min(y, _format.h - 1), static_cast<int32_t>(0));

        ValueAsString t;
        t.str       = build_variable_name(y);
        t.type.str  = get_cl_data_type(_format.dt, 1);
        t.type.dt   = _format.dt;
        t.type.size = 1;

        // Check required because if the width has only one element, we cannot use .s0
        if (_format.w != 1)
        {
            // Automatic broadcasting
            t.str += ".s" + std::to_string(x);
        }

        return t;
    }

    ValueAsString vector(int32_t y) const override
    {
        y = std::max(std::min(y, _format.h - 1), static_cast<int32_t>(0));

        ValueAsString t;
        t.str       = build_variable_name(y);
        t.type.str  = get_cl_data_type(_format.dt, _format.w);
        t.type.dt   = _format.dt;
        t.type.size = _format.w;
        return t;
    }

    ValueAsString vector(int32_t x_start, int32_t width, int32_t y) const override
    {
        y = std::max(std::min(y, _format.h - 1), static_cast<int32_t>(0));

        ValueAsString t;
        t.str       = build_variable_name(y);
        t.type.str  = get_cl_data_type(_format.dt, width);
        t.type.dt   = _format.dt;
        t.type.size = width;

        if (_format.w != 1)
        {
            t.str += ".s";
            for (int i = 0; i < width; ++i)
            {
                t.str += to_scalar_hex(x_start + i);
            }
        }
        return t;
    }

    std::vector<ValueAsString> underlying_source_variables() const override
    {
        std::vector<ValueAsString> vars;
        for (int32_t y = 0; y < _format.h; ++y)
        {
            ValueAsString t;
            t.str       = build_variable_name(y);
            t.type.str  = get_cl_data_type(_format.dt, _format.w);
            t.type.dt   = _format.dt;
            t.type.size = _format.w;
            vars.push_back(t);
        }
        return vars;
    }

    bool is_assignable() const override
    {
        return true;
    }

    bool need_declaration() const override
    {
        return true;
    }

private:
    std::string build_variable_name(int32_t y) const
    {
        std::string var_name = _basename;

        if (_format.h == 1)
        {
            return var_name;
        }
        else
        {
            var_name += "_";
            var_name += std::to_string(y);
        }

        return var_name;
    }

    std::string to_scalar_hex(int32_t x) const
    {
        switch (x)
        {
            case 0:
            case 1:
            case 2:
            case 3:
            case 4:
            case 5:
            case 6:
            case 7:
            case 8:
            case 9:
                return std::to_string(x);
            case 10:
                return "A";
            case 11:
                return "B";
            case 12:
                return "C";
            case 13:
                return "D";
            case 14:
                return "E";
            case 15:
                return "F";
            default:
                std::cout << "Unsupported hexadecimal value" << std::endl;
                assert(false);
                return "";
        }
    }
};

// Unique features: It contains values in the form of string. The name used for this object is misleading since the variables can change the value over time.
class ClConstantTile : public IVectorTile
{
public:
    ClConstantTile(const std::vector<std::vector<std::string>> &in, DataType dt)
    {
        _format.w  = in[0].size();
        _format.h  = in.size();
        _format.dt = dt;

        _data = std::vector<std::vector<std::string>>(_format.h, std::vector<std::string>(_format.w));

        for (int32_t y = 0; y < _format.h; ++y)
        {
            for (int32_t x = 0; x < _format.w; ++x)
            {
                _data[y][x] = in[y][x];
            }
        }
    }

    ValueAsString scalar(int32_t x, int32_t y) const override
    {
        x = std::max(std::min(x, _format.w - 1), static_cast<int32_t>(0));
        y = std::max(std::min(y, _format.h - 1), static_cast<int32_t>(0));

        ValueAsString t;
        t.str       = _data[y][x];
        t.type.str  = get_cl_data_type(_format.dt, 1);
        t.type.dt   = _format.dt;
        t.type.size = 1;

        return t;
    }

    ValueAsString vector(int32_t y) const override
    {
        y = std::max(std::min(y, _format.h - 1), static_cast<int32_t>(0));

        return vector(0, _format.w, y);
    }

    ValueAsString vector(int32_t x_start, int32_t width, int32_t y) const override
    {
        y = std::max(std::min(y, _format.h - 1), static_cast<int32_t>(0));

        ValueAsString t;
        t.str       = "";
        t.type.str  = get_cl_data_type(_format.dt, width);
        t.type.dt   = _format.dt;
        t.type.size = width;

        if (width > 1)
        {
            t.str += "((" + get_cl_data_type(_format.dt, width) + ")(";
        }

        int32_t x = x_start;
        for (; x < width - 1; ++x)
        {
            t.str += scalar(x, y).str;
            t.str += ", ";
        }
        t.str += scalar(x, y).str;

        if (width > 1)
        {
            t.str += "))";
        }

        return t;
    }

    std::vector<ValueAsString> underlying_source_variables() const override
    {
        std::vector<ValueAsString> vars;

        for (int32_t y = 0; y < _format.h; ++y)
        {
            for (int32_t x = 0; x < _format.w; ++x)
            {
                ValueAsString t;
                t.str       = _data[y][x];
                t.type.str  = get_cl_data_type(_format.dt, 1);
                t.type.dt   = _format.dt;
                t.type.size = 1;
                vars.push_back(t);
            }
        }

        return vars;
    }

    bool is_assignable() const override
    {
        return false;
    }

    bool need_declaration() const override
    {
        return false;
    }

private:
    std::vector<std::vector<std::string>> _data{};
};

enum class TensorComponentIndex : int32_t
{
    IndexMask = 0x0000000f,
};

enum class TensorComponentGroup : int32_t
{
    OffsetFirstElement = 0x00000100,
    Stride             = 0x00001000,
    Dimension          = 0x00010000,
    FoldedDimension    = 0x00100000,
    Constant           = 0x01000000
};

inline std::string to_string(TensorComponentType x)
{
    switch (x)
    {
        case TensorComponentType::Unknown:
            return "Unknown";
        case TensorComponentType::OffsetFirstElement:
            return "OffsetFirstElement";
        case TensorComponentType::Stride1:
            return "Stride1";
        case TensorComponentType::Stride2:
            return "Stride2";
        case TensorComponentType::Stride3:
            return "Stride3";
        case TensorComponentType::Stride4:
            return "Stride4";
        case TensorComponentType::Dim0:
            return "Dim0";
        case TensorComponentType::Dim1:
            return "Dim1";
        case TensorComponentType::Dim2:
            return "Dim2";
        case TensorComponentType::Dim3:
            return "Dim3";
        case TensorComponentType::Dim4:
            return "Dim4";
        case TensorComponentType::Dim1xDim2:
            return "Dim1xDim2";
        case TensorComponentType::Dim1xDim2xDim3:
            return "Dim1xDim2xDim3";
        default:
            assert(false);
            return "";
    }
}

class ITensorArgument
{
public:
    virtual ~ITensorArgument() = default;

    /** Method to get the tensor component as a string
     *
     * @param[in] x tensor component to query
     *
     * @return  the tensor component as a string
     */
    virtual std::string component(TensorComponentType x) = 0;

    /** Method to get the tensor component type declaration as a string
     *
     * @return  the tensor component type declaration as a string
     */
    virtual std::string component_type_declaration() const = 0;

    /** Method to get the tensor component data type
     *
     * @return  the tensor component data type
     */
    virtual DataType component_data_type() const = 0;

    /** Method to get the tensor component declarations
     *
     * @return a vector containing the tensor component declarations
     */
    virtual std::vector<TensorComponentType> component_declarations() const = 0;

    /** Method to get the name of the tensor argument.
     *
     * @return the name of the tensor argument
     */
    std::string name() const
    {
        return _basename;
    }

    /** Method to get the tensor format
     *
     * @return the format
     */
    TensorInfo format() const
    {
        return _format;
    }

protected:
    TensorInfo  _format{};
    std::string _basename{};
};

enum class GpuTensorStorage : int32_t
{
    Unknown          = 0x0000,
    BufferUint8Ptr   = 0x0012,
    Image2dReadOnly  = 0x0020,
    Image2dWriteOnly = 0x0021,
    Image3dReadOnly  = 0x0030,
    Image3dWriteOnly = 0x0031
};

inline GpuTensorStorage to_gpu_tensor_storage(TensorStorageType s)
{
    switch (s)
    {
        case TensorStorageType::Unknown:
            return GpuTensorStorage::Unknown;

        case TensorStorageType::BufferUint8Ptr:
            return GpuTensorStorage::BufferUint8Ptr;

        case TensorStorageType::Texture2dReadOnly:
            return GpuTensorStorage::Image2dReadOnly;

        case TensorStorageType::Texture2dWriteOnly:
            return GpuTensorStorage::Image2dWriteOnly;

        default:
            assert(false);
            return GpuTensorStorage::Unknown;
    }
}

inline TensorStorageType to_tensor_storage(GpuTensorStorage s)
{
    switch (s)
    {
        case GpuTensorStorage::Unknown:
            return TensorStorageType::Unknown;

        case GpuTensorStorage::BufferUint8Ptr:
            return TensorStorageType::BufferUint8Ptr;

        case GpuTensorStorage::Image2dReadOnly:
            return TensorStorageType::Texture2dReadOnly;

        case GpuTensorStorage::Image2dWriteOnly:
            return TensorStorageType::Texture2dWriteOnly;

        default:
            assert(false);
            return TensorStorageType::Unknown;
    }
}

class IGpuTensorArgument : public ITensorArgument
{
public:
    virtual ~IGpuTensorArgument() = default;

    /** Method to get the tensor storage, which is the underlying storage used to keep the data memory
     *
     * @param[in] x tensor storage to query
     *
     * @return  the tensor storage as a string
     */
    virtual std::string storage(GpuTensorStorage x) = 0;

    /** Method to get the tensor storage type declaration as a string
     *
     * @param[in] x tensor component to query
     *
     * @return  the tensor storage type declaration as a string
     */
    virtual std::string storage_type_declaration(GpuTensorStorage x) const = 0;

    /** Method to get the tensor storage declarations
     *
     * @return a vector containing the tensor storage declarations
     */
    virtual std::vector<GpuTensorStorage> storage_declarations() const = 0;
};

class ClTensorArgument : public IGpuTensorArgument
{
public:
    ClTensorArgument(const std::string &name, const TensorInfo &x, bool return_by_value_when_possible)
    {
        _basename                      = name;
        _format                        = x;
        _return_by_value_when_possible = return_by_value_when_possible;
    }

    // Methods to override
    std::string component(TensorComponentType x) override
    {
        if ((static_cast<int32_t>(x) & static_cast<int32_t>(TensorComponentGroup::Constant)))
        {
            int32_t idx = static_cast<int32_t>(x) & static_cast<int32_t>(TensorComponentIndex::IndexMask);
            return std::to_string(idx - 1);
        }

        if (_return_by_value_when_possible)
        {
            if ((static_cast<int32_t>(x) & static_cast<int32_t>(TensorComponentGroup::Dimension)))
            {
                int32_t idx = static_cast<int32_t>(x) & static_cast<int32_t>(TensorComponentIndex::IndexMask);
                return std::to_string(_format.shape[idx]);
            }

            if ((static_cast<int32_t>(x) & static_cast<int32_t>(TensorComponentGroup::FoldedDimension)))
            {
                switch (x)
                {
                    case TensorComponentType::Dim1xDim2:
                        return std::to_string(_format.shape[1] * _format.shape[2]);
                    case TensorComponentType::Dim1xDim2xDim3:
                        return std::to_string(_format.shape[1] * _format.shape[2] * _format.shape[2]);
                    default:
                        std::cout << "Unsupported folded dimension" << std::endl;
                        assert(false);
                }
            }
        }

        if (std::find(_components_required.begin(), _components_required.end(), x) == _components_required.end())
        {
            _components_required.push_back(x);
        }

        return build_component_name(x);
    }

    std::string component_type_declaration() const override
    {
        return "int";
    };

    DataType component_data_type() const override
    {
        return DataType::Int32;
    }

    std::string storage(GpuTensorStorage x) override
    {
        if (std::find(_storage_required.begin(), _storage_required.end(), x) == _storage_required.end())
        {
            _storage_required.push_back(x);
        }

        return build_storage_name(x);
    }

    std::string storage_type_declaration(GpuTensorStorage x) const override
    {
        switch (x)
        {
            case GpuTensorStorage::BufferUint8Ptr:
                return "__global uchar*";
            case GpuTensorStorage::Image2dReadOnly:
                return "__read_only image2d_t";
            case GpuTensorStorage::Image2dWriteOnly:
                return "__write_only image2d_t";
            case GpuTensorStorage::Image3dReadOnly:
                return "__read_only image3d_t ";
            case GpuTensorStorage::Image3dWriteOnly:
                return "__write_only image3d_t ";
            default:
                std::cout << "Unsupported storage" << std::endl;
                assert(false);
                return "";
        }
    };

    std::vector<GpuTensorStorage> storage_declarations() const override
    {
        return _storage_required;
    }

    std::vector<TensorComponentType> component_declarations() const override
    {
        return _components_required;
    }

private:
    std::string build_storage_name(GpuTensorStorage x) const
    {
        std::string var_name = _basename;

        switch (x)
        {
            case GpuTensorStorage::BufferUint8Ptr:
                return var_name + "_ptr";
            case GpuTensorStorage::Image2dReadOnly:
            case GpuTensorStorage::Image2dWriteOnly:
                return var_name + "_img2d";
            case GpuTensorStorage::Image3dReadOnly:
            case GpuTensorStorage::Image3dWriteOnly:
                return var_name + "_img3d";
            default:
                std::cout << "Unsupported storage" << std::endl;
                assert(false);
        }

        return var_name;
    }

    std::string build_component_name(TensorComponentType x) const
    {
        std::string var_name = _basename;

        switch (x)
        {
            case TensorComponentType::OffsetFirstElement:
                return var_name + "_offset_first_element";
            case TensorComponentType::Stride1:
                return var_name + "_stride1";
            case TensorComponentType::Stride2:
                return var_name + "_stride2";
            case TensorComponentType::Stride3:
                return var_name + "_stride3";
            case TensorComponentType::Dim0:
                return var_name + "_dim0";
            case TensorComponentType::Dim1:
                return var_name + "_dim1";
            case TensorComponentType::Dim2:
                return var_name + "_dim2";
            case TensorComponentType::Dim3:
                return var_name + "_dim3";
            case TensorComponentType::Dim1xDim2:
                return var_name + "_dim1xdim2";
            case TensorComponentType::Dim1xDim2xDim3:
                return var_name + "_dim1xdim2xdim3";
            default:
                std::cout << "Unsupported component" << std::endl;
                assert(false);
        }

        return var_name;
    }

    bool                             _return_by_value_when_possible{false};
    std::vector<GpuTensorStorage>    _storage_required{};
    std::vector<TensorComponentType> _components_required{};
};

/**
 * @brief Data structure that contains the declared tiles by the components.
 * The registry is a linear data structure that follows the similar principle of the stack. The user can use the @p increment_registry_level() method to
 * increase the level of the stack (0 when it starts). When the user uses the @p decrement_registry_level() method, the registry decreases the level of the stack
 * and remove (pop) all the tiles from the level above.
 * When a tile is declared on the level 0, it is a global tile. A global tile is visible in all parts of the code.
 * Since different components may use the same name to define a tile, the registry adopts the IdSpace concept, an @p id to prevent name collisions
 * when declaring tiles among different components.
 *
 */
class GpuTileRegistry
{
public:
    enum class RegistryTileType
    {
        Tile,
        Link
    };

    using RegistryIdSpace  = int32_t;
    using RegistryLevel    = int32_t;
    using RegistryTileName = std::string;

    struct RegistryTileTableEntry
    {
        RegistryLevel                registry_level{0};
        std::unique_ptr<IVectorTile> tile_object{nullptr};
    };

    struct RegistryTileTypeTableEntry
    {
        RegistryTileType tile_type{RegistryTileType::Tile};
        RegistryTileName tile_name{};
        RegistryIdSpace  registry_idspace{0};
        RegistryLevel    registry_level{0};
    };

    using RegistryTileTable     = std::map<RegistryIdSpace, std::map<RegistryTileName, RegistryTileTableEntry>>;
    using RegistryTileTypeTable = std::map<RegistryIdSpace, std::map<RegistryTileName, RegistryTileTypeTableEntry>>;

    /**
     * @brief Construct a new Gpu Tile Registry object
     *
     */
    GpuTileRegistry()
    {
        _language = GpuTargetLanguage::Unknown;
    }

    /**
     * @brief Construct a new Gpu Tile Registry object providing the Gpu programming language
     *
     * @param[in] language Gpu programming language to use
     */
    GpuTileRegistry(GpuTargetLanguage language)
    {
        _language = language;
    }

    /**
     * @brief Default destructor. Destroy the Gpu Tile Registry object
     *
     */
    ~GpuTileRegistry() = default;

    /**
     * @brief Set the working IdSpace for the tile registry. IdSpace is used to prevent name collisions when declaring tiles.
     *        Therefore, the IdSpace should be set before declaring any tiles.
     *
     * @param[in] id The IdSpace id
     */
    void set_IdSpace(int32_t id)
    {
        _IdSpace = id;
    }

    /**
     * @brief Get the current working IdSpace for the tile registry. IdSpace is used to prevent name collisions when declaring tiles
     *
     * @return The IdSpace id
     */
    int32_t IdSpace() const
    {
        return _IdSpace;
    }

    /**
     * @brief Gets all the IdSpace declarations defined in the tile registry.
     *
     * @return all the IdSpace declarations defined in the tile registry as std::vector<int32_t>. It returns an empty vector if there are no IdSpace declarations.
     */
    std::vector<int32_t> IdSpace_declarations() const
    {
        std::vector<int32_t> x;

        auto it = _frags.begin();

        while (it != _frags.end())
        {
            x.push_back(it->first);

            it++;
        }

        return x;
    }

    /**
     * @brief Declare a tile from a previously created tile
     */
    void insert(const std::string &name, const IVectorTile *frag)
    {
        assert(_language == GpuTargetLanguage::OpenCL);
        const int32_t     key_IdSpace  = _IdSpace;
        const std::string key_var_name = name;
        const std::string var_name     = frag->name();
        TileInfo          format       = frag->format();

        // First check whether a tile with the same name exists
        IVectorTile *result = (*this)[key_var_name];
        assert(result == nullptr);
        if (result == nullptr)
        {
            std::unique_ptr<ClTile> tile = std::make_unique<ClTile>(var_name, format);

            _frags[key_IdSpace][key_var_name].tile_object    = std::move(tile);
            _frags[key_IdSpace][key_var_name].registry_level = _registry_level;

            _frag_types[key_IdSpace][key_var_name].tile_type        = RegistryTileType::Link;
            _frag_types[key_IdSpace][key_var_name].tile_name        = key_var_name;
            _frag_types[key_IdSpace][key_var_name].registry_idspace = _IdSpace;
            _frag_types[key_IdSpace][key_var_name].registry_level   = _registry_level;
        }
    }

    /**
     * @brief Declare a tile with TileInfo. The tile will be stored in the IdSpace set with @p set_IdSpace()
     *
     * @note The reference name used for declaring the tile should not be previously used in the IdSpace
     *
     * @param[in] name   Reference name for the tile. The reference name can be used to retrieve the tile stored in the registry.
     * @param[in] format Tile format use to use
     */
    void insert(const std::string &name, const TileInfo &format)
    {
        assert(_language == GpuTargetLanguage::OpenCL);
        const int32_t     key_IdSpace  = _IdSpace;
        const std::string key_var_name = name;
        const std::string var_name     = generate_tile_name(name);

        // First check whether a tile with the same name exists
        IVectorTile *result = (*this)[key_var_name];
        assert(result == nullptr);
        if (result == nullptr)
        {
            std::unique_ptr<ClTile> tile                     = std::make_unique<ClTile>(var_name, format);
            _frags[key_IdSpace][key_var_name].tile_object    = std::move(tile);
            _frags[key_IdSpace][key_var_name].registry_level = _registry_level;

            _frag_types[key_IdSpace][key_var_name].tile_type        = RegistryTileType::Tile;
            _frag_types[key_IdSpace][key_var_name].tile_name        = key_var_name;
            _frag_types[key_IdSpace][key_var_name].registry_idspace = _IdSpace;
            _frag_types[key_IdSpace][key_var_name].registry_level   = _registry_level;
        }
    }

    /**
     * @brief Declare a constant tile. The content of the tile is passed as a vector of std::string
     *
     * @note The reference name used for declaring the tile should not be previously used in the IdSpace
     *
     * @param[in] name Reference name for the tile. The reference name can be used to retrieve the tile stored in the registry.
     * @param[in] in   A 3D std::vector of std::string. From the 3D std::vector we can know the dimensions for the tile
     * @param[in] dt   The data type for the elements stored in the 3D std::vector as std::string. It is user's responsibilty to ensure
     *                 that the data type is aligned with the content of the std::string.
     */
    void insert(const std::string &name, const std::vector<std::vector<std::string>> &in, DataType dt)
    {
        assert(_language == GpuTargetLanguage::OpenCL);
        const int32_t     key_IdSpace  = _IdSpace;
        const std::string key_var_name = name;

        // First check whether a tile with the same name exists
        IVectorTile *result = (*this)[key_var_name];
        assert(result == nullptr);
        if (result == nullptr)
        {
            std::unique_ptr<ClConstantTile> tile             = std::make_unique<ClConstantTile>(in, dt);
            _frags[key_IdSpace][key_var_name].tile_object    = std::move(tile);
            _frags[key_IdSpace][key_var_name].registry_level = _registry_level;

            _frag_types[key_IdSpace][key_var_name].tile_type        = RegistryTileType::Tile;
            _frag_types[key_IdSpace][key_var_name].tile_name        = key_var_name;
            _frag_types[key_IdSpace][key_var_name].registry_idspace = _IdSpace;
            _frag_types[key_IdSpace][key_var_name].registry_level   = _registry_level;
        }
    }

    /**
     * @brief Declare an anonymous constant tile. The content of the tile is passed as a vector of std::string
     *
     * @note This method can be used to declare temporary tiles that need to be accessed only once.
     *
     * @param[in] in   A 3D std::vector of std::string. From the 3D std::vector we can know the dimensions for the tile
     * @param[in] dt   The data type for the elements stored in the 3D std::vector as std::string. It is user responsibilty to ensure
     *                 that the data type is aligned with what passed with the std::string.
     *
     * @return IVectorTile* the anonymous constant tile
     */
    IVectorTile *insert(const std::vector<std::vector<std::string>> &in, DataType dt)
    {
        assert(_language == GpuTargetLanguage::OpenCL);
        const int32_t     key_IdSpace  = _IdSpace;
        const std::string key_var_name = "_" + std::to_string(_anonymous_frag_count++);

        // First check whether a tile with the same name exists
        IVectorTile *result = (*this)[key_var_name];
        assert(result == nullptr);
        if (result == nullptr)
        {
            std::unique_ptr<ClConstantTile> tile             = std::make_unique<ClConstantTile>(in, dt);
            _frags[key_IdSpace][key_var_name].tile_object    = std::move(tile);
            _frags[key_IdSpace][key_var_name].registry_level = _registry_level;

            _frag_types[key_IdSpace][key_var_name].tile_type        = RegistryTileType::Tile;
            _frag_types[key_IdSpace][key_var_name].tile_name        = key_var_name;
            _frag_types[key_IdSpace][key_var_name].registry_idspace = _IdSpace;
            _frag_types[key_IdSpace][key_var_name].registry_level   = _registry_level;
        }

        return (*this)[key_var_name];
    }

    /**
     * @brief Get the tile from the registry. This method searches the tile in the IdSpace provided by the user
     *
     * @param[in] name         The name of the tile to retrieve
     * @param[in] IdSpace The IdSpace id where to search the tile
     *
     * @return IVectorTile* The tile
     */
    IVectorTile *get(const std::string &name, int32_t IdSpace)
    {
        const int32_t     key_IdSpace  = IdSpace;
        const std::string key_var_name = name;

        IVectorTile *result         = nullptr;
        auto         search_IdSpace = _frags.find(key_IdSpace);
        if (search_IdSpace != _frags.end())
        {
            auto search_tile = _frags[key_IdSpace].find(key_var_name);
            if (search_tile != _frags[key_IdSpace].end())
            {
                result = search_tile->second.tile_object.get();
                assert(result != nullptr);
            }
        }

        return result;
    }

    /**
     * @brief Get the tile from the registry. This method searches the tile in the IdSpace set with @p set_IdSpace()
     *
     * @param[in] name The name of the tile to retrieve
     *
     * @return IVectorTile* The tile
     */
    IVectorTile *operator[](const std::string &name)
    {
        return get(name, _IdSpace);
    }

    /**
     * @brief Check whether the tile in the in the IdSpace provided by the user exists
     *
     * @param[in] name         Name of the tile to search for
     * @param[in] IdSpace The IdSpace id where to search the tile
     *
     * @return true if the tile exists
     * @return false if the tile does not exist
     */
    bool has_tile(const std::string &name, int32_t IdSpace) const
    {
        const int32_t     key_IdSpace  = IdSpace;
        const std::string key_var_name = name;

        // IVectorTile* result = nullptr;
        auto search_IdSpace = _frags.find(key_IdSpace);

        return search_IdSpace != _frags.end();
    }

    /**
     * @brief Check whether the tile within the current IdSpace exists
     *
     * @param[in] name Name of the tile to search for
     *
     * @return true if the tile exists
     * @return false if the tile does not exist
     */
    bool has_tile(const std::string &name) const
    {
        return has_tile(name, _IdSpace);
    }

    /**
     * @brief Get all the tiles declared within the IdSpace provided by the user
     *
     * @param[in] IdSpace IdSpace where to retrieve all the declared tiles
     *
     * @return std::vector<IVectorTile*> A vector with all the declared tiles in the IdSpace provided by the user
     */
    std::vector<IVectorTile *> tile_declarations(int32_t IdSpace)
    {
        std::vector<IVectorTile *> tiles;

        std::map<RegistryTileName, RegistryTileTypeTableEntry>::iterator it = _frag_types[IdSpace].begin();

        while (it != _frag_types[IdSpace].end())
        {
            // The following line should be enabled. However, we cannot at this stage
            // because it used to retrieve the output tile produced by each component.
            // However, this method should NOT be used to retrieve the output tile
            //if(it->second.tile_type == RegistryTileType::Tile)
            {
                tiles.push_back(get(it->second.tile_name, it->second.registry_idspace));
            }
            it++;
        }

        return tiles;
    }

    /**
     * @brief Increase the level of stack.
     *
     */
    void increment_registry_level()
    {
        _registry_level++;
    }

    /**
     * @brief Remove all the tiles declared at the current stack level and decrease the level of the stack.
     *
     */
    void decrement_registry_level()
    {
        assert(_registry_level >= 0);

        // Remove all variables in the local scope
        std::map<RegistryTileName, RegistryTileTableEntry>::iterator it = _frags[_IdSpace].begin();

        while (it != _frags[_IdSpace].end())
        {
            if (it->second.registry_level == _registry_level)
            {
                it = _frags[_IdSpace].erase(it);
            }
            else
            {
                it++;
            }
        }

        std::map<RegistryTileName, RegistryTileTypeTableEntry>::iterator it_type = _frag_types[_IdSpace].begin();

        while (it_type != _frag_types[_IdSpace].end())
        {
            if (it_type->second.registry_level == _registry_level)
            {
                it_type = _frag_types[_IdSpace].erase(it_type);
            }
            else
            {
                it_type++;
            }
        }

        _registry_level--;
    }

    /**
     * @brief Get the level of the stack
     *
     */
    int32_t level() const
    {
        return _registry_level;
    }

private:
    // This method ensures that the key is unique among different components
    std::string generate_tile_name(const std::string &name)
    {
        assert(_IdSpace >= 0);
        if (_registry_level == 0)
        {
            return "_G" + std::to_string(_IdSpace) + "_" + name;
        }
        else
        {
            return name;
        }
    }

    RegistryTileTable     _frags{};
    RegistryTileTypeTable _frag_types{};
    RegistryLevel         _registry_level{0};
    RegistryIdSpace       _IdSpace{-1};
    int32_t               _anonymous_frag_count{0};              // Counter used to create the anonymous tiles
    GpuTargetLanguage     _language{GpuTargetLanguage::Unknown}; // Gpu programming language
};

using TensorEntry = std::unique_ptr<IGpuTensorArgument>;

/**
 * @brief Data structure that contains the tensors consumed by the components.
 * Since different components may use the same name as reference for a tensor, the registry adopts the IdSpace concept, an @p id to prevent name collisions
 * when declaring tensors among different components.
 *
 */
class GpuTensorArgumentRegistry
{
public:
    /**
     * @brief Construct a new Gpu Tensor Registry object
     *
     */
    GpuTensorArgumentRegistry()
    {
        _language = GpuTargetLanguage::Unknown;
    }

    /**
     * @brief Construct a new Gpu Tensor Registry object
     *
     * @param[in] language Gpu programming language to use
     */
    GpuTensorArgumentRegistry(GpuTargetLanguage language)
    {
        _language = language;
    }

    /**
     * @brief Default destructor. Destroy the Gpu Tensor Registry object
     *
     */
    ~GpuTensorArgumentRegistry() = default;

    /**
     * @brief Set the working IdSpace for the tensor registry. IdSpace is used to prevent name collisions when declaring tensors.
     *        Therefore, the IdSpace should be set before declaring any tensors.
     *
     * @param[in] id The IdSpace id
     */
    void set_IdSpace(int32_t id)
    {
        _IdSpace = id;
    }

    /**
     * @brief Get the current working IdSpace for the tensor registry. IdSpace is used to prevent name collisions when declaring tensors
     *
     * @return The IdSpace id
     */
    int32_t IdSpace() const
    {
        return _IdSpace;
    }

    /**
     * @brief Gets all the IdSpace declarations defined in the tensor registry.
     *
     * @return all the IdSpace declarations defined in the tensor registry as std::vector<int32_t>. It returns an empty vector if there are no IdSpace declarations.
     */
    std::vector<int32_t> IdSpace_declarations() const
    {
        std::vector<int32_t> x;

        auto it = _refs.begin();

        while (it != _refs.end())
        {
            x.push_back(it->first);

            it++;
        }

        return x;
    }

    /**
     * @brief Declare a tensor with TensorInfo. The tensor will be stored in the IdSpace set with @p set_IdSpace()
     *
     * @note The reference name used for declaring the tensor should not be previously used in the IdSpace
     *
     * @param[in] name                          Reference name for the tensor. The reference name can be used to retrieve the tensor stored in the registry.
     * @param[in] x                             Pair of tensor info and tensor id
     * @param[in] return_by_value_when_possible True if we want the value stored in the tensor components
     */
    void insert(const std::string &name, const TensorInfo &x, bool return_by_value_when_possible)
    {
        assert(_language == GpuTargetLanguage::OpenCL);
        const int32_t     key_IdSpace  = _IdSpace;
        const int32_t     tensor_id    = x.id;
        const std::string key_var_name = name;
        const std::string var_name     = generate_tensor_name(name, tensor_id);

        // First, check whether the tensor has already a reference. If so, trigger an assert
        assert(!has_tensor_argument(name));

        // Check whether a tensor with that tensorID exists
        auto result = _tensor_arguments.find(tensor_id);
        if (result == _tensor_arguments.end())
        {
            // It means that we haven't added a tensor with that tensor_id yet. Create a IGpuTensorArgument before creating the reference
            std::unique_ptr<ClTensorArgument> arg =
                std::make_unique<ClTensorArgument>(var_name, x, return_by_value_when_possible);
            _tensor_arguments[tensor_id] = std::move(arg);
        }

        _refs[key_IdSpace][key_var_name] = tensor_id;
    }

    /**
     * @brief Get the tensor from the registry. This method searches the tensor in the IdSpace set with @p set_IdSpace()
     *
     * @param[in] name The name of the tensor to retrieve
     *
     * @return IGpuTensor* The tensor
     */
    IGpuTensorArgument *operator[](const std::string &name)
    {
        const int32_t     key_IdSpace  = _IdSpace;
        const std::string key_var_name = name;

        IGpuTensorArgument *result         = nullptr;
        auto                search_IdSpace = _refs.find(key_IdSpace);
        if (search_IdSpace != _refs.end())
        {
            auto search_tensor_id = _refs[key_IdSpace].find(key_var_name);

            if (search_tensor_id != _refs[key_IdSpace].end())
            {
                const int32_t tensor_id              = search_tensor_id->second;
                auto          search_tensor_argument = _tensor_arguments.find(tensor_id);
                if (search_tensor_argument != _tensor_arguments.end())
                {
                    result = search_tensor_argument->second.get();
                }
                assert(result != nullptr);
            }
        }

        return result;
    }

    /**
     * @brief Get all the tensors declared in the IdSpace provided by the user
     *
     * @return std::vector<IGpuTensorArgument*> A vector with all the declared tensors
     */
    std::vector<IGpuTensorArgument *> tensor_argument_declarations()
    {
        std::vector<IGpuTensorArgument *> args;

        auto it = _tensor_arguments.begin();

        while (it != _tensor_arguments.end())
        {
            args.push_back(it->second.get());
            it++;
        }

        return args;
    }

    /**
     * @brief Check whether the tensor argument in the IdSpace set with @p set_IdSpace() exists
     *
     * @param[in] name Name of the tensor argument to search for
     *
     * @return true if the tensor argument exists
     * @return false if the tensor argument does not exist
     */
    bool has_tensor_argument(const std::string &name)
    {
        const int32_t     key_IdSpace  = _IdSpace;
        const std::string key_var_name = name;

        auto search_IdSpace = _refs.find(key_IdSpace);

        if (search_IdSpace != _refs.end())
        {
            auto search_tensor_id = _refs[key_IdSpace].find(key_var_name);

            return search_tensor_id != _refs[key_IdSpace].end();
        }
        else
        {
            return false;
        }
    }

    /**
     * @brief Check whether the tensor argument is in the the IdSpace provided by the user
     *
     * @param[in] name    Name of the tensor argument to search for
     * @param[in] IdSpace The IdSpace id where to search the tensor argument
     *
     * @return true if the tile exists
     * @return false if the tile does not exist
     */
    bool has_tensor_argument(const std::string &name, int32_t IdSpace)
    {
        const int32_t     key_IdSpace  = IdSpace;
        const std::string key_var_name = name;

        auto search_IdSpace = _refs.find(key_IdSpace);

        if (search_IdSpace != _refs.end())
        {
            auto search_tensor_id = _refs[key_IdSpace].find(key_var_name);

            return search_tensor_id != _refs[key_IdSpace].end();
        }
        else
        {
            return false;
        }
    }

private:
    // This method ensures that the key is unique among different components
    std::string generate_tensor_name(const std::string &name, int32_t tensor_id)
    {
        assert(tensor_id >= 0);

        return name + std::to_string(tensor_id);
    }

    std::map<int32_t, TensorEntry>                    _tensor_arguments{};
    std::map<int32_t, std::map<std::string, int32_t>> _refs{};
    int32_t                                           _IdSpace{-1};
    GpuTargetLanguage                                 _language{GpuTargetLanguage::Unknown}; // Gpu programming language
};

enum class OpType : int32_t
{
    Elementwise = 0x0000,
    Relational  = 0x1000,
    Algebra     = 0x2000
};

inline std::string to_string(AssignmentOp op)
{
    switch (op)
    {
        case AssignmentOp::Decrement:
            return "-=";
        case AssignmentOp::Increment:
            return "+=";
        default:
            assert(false);
            return "";
    }
}

inline std::string to_string(UnaryOp op)
{
    switch (op)
    {
        case UnaryOp::LogicalNot:
            return "!";
        case UnaryOp::BitwiseNot:
            return "~";
        case UnaryOp::Negate:
            return "-";
        default:
            assert(false);
            return "";
    }
}

inline std::string to_string(BinaryOp op)
{
    switch (op)
    {
        case BinaryOp::Add:
            return "+";
        case BinaryOp::Sub:
            return "-";
        case BinaryOp::Mul:
            return "*";
        case BinaryOp::Div:
            return "/";
        case BinaryOp::Mod:
            return "%";
        case BinaryOp::Equal:
            return "==";
        case BinaryOp::Less:
            return "<";
        case BinaryOp::LessEqual:
            return "<=";
        case BinaryOp::Greater:
            return ">";
        case BinaryOp::GreaterEqual:
            return ">=";
        case BinaryOp::LogicalAnd:
            return "&&";
        case BinaryOp::LogicalOr:
            return "||";
        case BinaryOp::BitwiseXOR:
            return "^";
        default:
            assert(false);
            return "";
    }
}

inline std::string binary_op_string(BinaryOp op)
{
    switch (op)
    {
        case BinaryOp::Add:
            return "add";
        case BinaryOp::Sub:
            return "sub";
        case BinaryOp::Mul:
            return "mul";
        case BinaryOp::Div:
            return "div";
        case BinaryOp::Mod:
            return "mod";
        case BinaryOp::Equal:
            return "eq";
        case BinaryOp::Less:
            return "gt";
        case BinaryOp::LessEqual:
            return "gteq";
        case BinaryOp::Greater:
            return "lt";
        case BinaryOp::GreaterEqual:
            return "lte";
        default:
            assert(false);
            return "";
    }
}

enum class OperandType : int32_t
{
    Unknown              = 0x00000000,
    ScalarFp32           = 0x00001011, // Immediate scalar tile
    ScalarFp16           = 0x00001012, // Immediate scalar tile
    ScalarInt32          = 0x00001021, // Immediate scalar tile
    ScalarInt16          = 0x00001022, // Immediate scalar tile
    ScalarInt8           = 0x00001024, // Immediate scalar tile
    ScalarUInt32         = 0x00001031, // Immediate scalar tile
    ScalarUInt16         = 0x00001032, // Immediate scalar tile
    ScalarUInt8          = 0x00001034, // Immediate scalar tile
    ScalarBool           = 0x00001041, // Immediate scalar tile
    ScalarTile           = 0x00001050, // Scalar from a tile
    Tile                 = 0x00010000, // Tile
    TensorStride1        = 0x00100001, // Tensor component
    TensorStride2        = 0x00100002, // Tensor component
    TensorStride3        = 0x00100003, // Tensor component
    TensorStride4        = 0x00100004, // Tensor component
    TensorDim0           = 0x00100010, // Tensor component
    TensorDim1           = 0x00100020, // Tensor component
    TensorDim2           = 0x00100030, // Tensor component
    TensorDim3           = 0x00100040, // Tensor component
    TensorDim4           = 0x00100050, // Tensor component
    TensorC              = 0x00100010, // Tensor component
    TensorW              = 0x00100020, // Tensor component
    TensorH              = 0x00100030, // Tensor component
    TensorD              = 0x00100040, // Tensor component
    TensorN              = 0x00100050, // Tensor component
    TensorDim1xDim2      = 0x00100100, // Tensor component
    TensorDim1xDim2xDim3 = 0x00100200, // Tensor component
    TensorWxH            = 0x00100300, // Tensor component
    TensorWxHxD          = 0x00100400, // Tensor component
    TensorDataOffset     = 0x00100500, // Tensor component
};

struct ScalarTileCoord
{
    ScalarTileCoord()
    {
    }

    ScalarTileCoord(int32_t x0, int32_t y0) : x(x0), y(y0)
    {
    }

    int32_t x{-1};
    int32_t y{-1};
};

/**
 * @brief Operand class. This object is used to pass the operands to the operations performed by the writer.
 * Operand can be of three types:
 * -# Scalar immediate: constant expression
 * -# Tile: A tile
 * -# Tensor component: A component (scalar) of a tensor
 *
 */
class Operand
{
public:
    Operand(const std::string &val)
    {
        _str  = val;
        _type = OperandType::Tile;
    }

    Operand(const std::string &val, const ScalarTileCoord &coord)
    {
        _str   = val;
        _type  = OperandType::ScalarTile;
        _coord = coord;
    }

    Operand(const std::string &val, OperandType type)
    {
        _str  = val;
        _type = type;
    }

    Operand(const Operand &t)
    {
        _str  = t.value();
        _type = t.type();
    }

    Operand &operator=(const Operand &t)
    {
        _str   = t.value();
        _type  = t.type();
        _coord = t.scalar_tile_coordinate();
        return *this;
    }

    std::string value() const
    {
        return _str;
    }

    OperandType type() const
    {
        return _type;
    }

    ScalarTileCoord scalar_tile_coordinate() const
    {
        return _coord;
    }

private:
    std::string     _str{};
    OperandType     _type{OperandType::Unknown};
    ScalarTileCoord _coord{};
};

using GpuSamplerTensorStorage = GpuTensorStorage;

struct GpuSampler
{
    GpuSampler() = default;

    TensorSamplerFormat       format{TensorSamplerFormat::Unknown};
    GpuSamplerTensorStorage   storage{GpuSamplerTensorStorage::Unknown};
    TensorSamplerAddressModeX address_mode_x{TensorSamplerAddressModeX::Unknown};
    TensorSamplerAddressModeY address_mode_y{TensorSamplerAddressModeY::Unknown};
    TensorSamplerAddressModeZ address_mode_z{TensorSamplerAddressModeZ::Unknown};
};

inline GpuSampler create_simple_sampler(
    const TensorInfo *tensor_info_id, GpuSampler sampler, int32_t step_x, int32_t step_y, int32_t step_z)
{
    CKW_UNUSED(step_x, step_y, step_z);

    auto tensor = tensor_info_id->shape;

    GpuSampler dst_sampler;
    dst_sampler.format         = sampler.format;
    dst_sampler.storage        = GpuSamplerTensorStorage::BufferUint8Ptr;
    dst_sampler.address_mode_x = sampler.address_mode_x;
    dst_sampler.address_mode_y = sampler.address_mode_y;
    dst_sampler.address_mode_z = sampler.address_mode_z;

    int32_t dim_x = 0;
    int32_t dim_y = 0;
    int32_t dim_z = 0;

    switch (sampler.format)
    {
        case TensorSamplerFormat::C_W_H:
            dim_x = tensor[0];
            dim_y = tensor[1];
            dim_z = tensor[2];
            break;
        case TensorSamplerFormat::C_WH_1:
            dim_x = tensor[0];
            dim_y = tensor[1] * tensor[2];
            dim_z = 1;
            break;
        default:
            std::cout << "Unsupported tensor format" << std::endl;
            assert(false);
            break;
    }

    if (dim_x == 1)
    {
        assert(step_x == 1);
        dst_sampler.address_mode_x = TensorSamplerAddressModeX::None;
    }

    if (dim_y == 1)
    {
        assert(step_y == 1);
        dst_sampler.address_mode_y = TensorSamplerAddressModeY::None;
    }

    if (dim_z == 1)
    {
        assert(step_z == 1);
        dst_sampler.address_mode_z = TensorSamplerAddressModeZ::None;
    }

    return dst_sampler;
}

class GpuOutputSampler
{
public:
    GpuOutputSampler() = default;

    /**
     * @brief Method used to initialize the GpuOutputSampler. The GpuOutputSampler can be initialized only once
     *        by the root component. Once initialized, all simpler components will need to used this sampler
     *        or a broadcasted version of it
     *
     * @param[in] sampler GpuSampler
     * @param[in] step_x  Increment step in the X direction. Not necessarily it is the same of n0 of tile!
     * @param[in] step_y  Increment step in the Y direction. Not necessarily it is the same of m0 of tile!
     * @param[in] step_z  Increment step in the Z direction. Not necessarily it is the same of d0 of tile!
     */
    void initialize(const TensorInfo       *tensor_info_id,
                    GpuSamplerTensorStorage tensor_storage,
                    TensorSamplerFormat     tensor_format,
                    int32_t                 step_x,
                    int32_t                 step_y,
                    int32_t                 step_z)
    {
        assert(_is_initialized == false);

        _step_x         = step_x;
        _step_y         = step_y;
        _step_z         = step_z;
        _tensor_info_id = tensor_info_id;
        _sampler        = create_sampler(tensor_storage, tensor_format);
        _is_initialized = true;
    };

    GpuSampler sampler() const
    {
        return _sampler;
    };

    int32_t step_x() const
    {
        return _step_x;
    };

    int32_t step_y() const
    {
        return _step_y;
    };

    int32_t step_z() const
    {
        return _step_z;
    };

private:
    GpuSampler create_sampler(GpuSamplerTensorStorage tensor_storage, TensorSamplerFormat tensor_format)
    {
        // Output can only be in output mode
        assert(tensor_storage != GpuSamplerTensorStorage::Image2dReadOnly);
        assert(tensor_storage != GpuSamplerTensorStorage::Image3dReadOnly);

        auto tensor = _tensor_info_id->shape;

        GpuSampler sampler;
        sampler.format         = tensor_format;
        sampler.storage        = tensor_storage;
        sampler.address_mode_x = TensorSamplerAddressModeX::None;
        sampler.address_mode_y = TensorSamplerAddressModeY::None;
        sampler.address_mode_z = TensorSamplerAddressModeZ::None;

        // In the case of texture, we do not need any special checks at the border
        if (tensor_storage == GpuSamplerTensorStorage::BufferUint8Ptr)
        {
            int32_t dim_x = 0;
            int32_t dim_y = 0;
            int32_t dim_z = 0;

            switch (tensor_format)
            {
                case TensorSamplerFormat::C_W_H:
                    dim_x = tensor[0];
                    dim_y = tensor[1];
                    dim_z = tensor[2];
                    break;
                case TensorSamplerFormat::C_WH_1:
                    dim_x = tensor[0];
                    dim_y = tensor[1] * tensor[2];
                    dim_z = 1;
                    break;
                default:
                    std::cout << "Unsupported tensor format" << std::endl;
                    assert(false);
                    break;
            }

            if ((dim_x % _step_x) != 0 && dim_x != 1)
            {
                sampler.address_mode_x = TensorSamplerAddressModeX::OverlappingMin;
            }

            if ((dim_y % _step_y) != 0 && dim_y != 1)
            {
                sampler.address_mode_y = TensorSamplerAddressModeY::ClampToMaxEdgeOnly;
            }

            if ((dim_z % _step_z) != 0 && dim_z != 1)
            {
                sampler.address_mode_z = TensorSamplerAddressModeZ::ClampToMaxEdgeOnly;
            }
        }

        return sampler;
    }

    GpuSampler        _sampler{}; // GpuSampler
    int32_t           _step_x{1};
    int32_t           _step_y{1};
    int32_t           _step_z{1};
    const TensorInfo *_tensor_info_id{nullptr};
    bool              _is_initialized{false};
};

/**
 * @brief Tensor operand class. This object is used to pass the operands as tensor to the operations performed by the writer.
 */
class TensorOperand
{
public:
    TensorOperand(const std::string &val, GpuSampler sampler) : _str(val), _sampler(sampler)
    {
    }

    TensorOperand &operator=(const TensorOperand &t)
    {
        _str     = t.value();
        _sampler = t.sampler();
        return *this;
    }

    std::string value() const
    {
        return _str;
    }

    GpuSampler sampler() const
    {
        return _sampler;
    }

private:
    std::string _str{};
    GpuSampler  _sampler{};
};

/**
 * @brief Data structure that contains all the necessary information to write the Gpu kernel with the Gpu kernel Writer
 *        This data structure must be initialized before being passed to the Gpu Kernel Writer
 *
 */
class GpuKernelWriterDataHolder
{
public:
    /**
     * @brief Construct a new Gpu Kernel Data object. In this phase, we should also store
     *        the GPU target and target specific capabilities (extensions). For now, we just initialize the
     *        programming language
     *
     * @param[in] language Gpu programming language to use
     */
    GpuKernelWriterDataHolder(GpuTargetLanguage language)
        : tiles(language), arguments(language), code(""), _language(language)
    {
    }

    /**
     * @brief Get the Gpu programming language used
     *
     * @return GpuTargetLanguage the Gpu programming language
     */
    GpuTargetLanguage programming_language() const
    {
        return _language;
    }

    /**
     * @brief @ref GpuTileRegistry
     *
     */
    GpuTileRegistry tiles{};
    /**
     * @brief @ref GpuTensorArgumentRegistry
     *
     */
    GpuTensorArgumentRegistry arguments{};
    /**
     * @brief @ref GpuOutputSampler.
     *
     */
    GpuOutputSampler output_sampler{};
    /**
     * @brief Source code
     *
     */
    std::string code{};

    // GpuExtensionRegistry extensions{};
private:
    GpuTargetLanguage _language;
};

struct LWS
{
    int32_t x{1};
    int32_t y{1};
    int32_t z{1};
};

/**
 * @brief Utility class used to get the tile from the operand. If the operand is not a tile, @ref OperandUnpacker
 *        declare an anonymous tile in the tile registry.
 */
class OperandUnpacker
{
public:
    OperandUnpacker(GpuTileRegistry &tiles, GpuTensorArgumentRegistry &arguments) : _tiles(tiles), _arguments(arguments)
    {
        // Increase the level of the stack to allocate possible temporary tiles
        _tiles.increment_registry_level();
    };

    ~OperandUnpacker()
    {
        // Decrease the level of the stack to deallocate any temporary tiles
        _tiles.decrement_registry_level();
    }

    IVectorTile *unpack(const Operand &src)
    {
        // Get the tile
        if (src.type() == OperandType::Tile)
        {
            assert(_tiles.has_tile(src.value()));
            return _tiles[src.value()];
        }
        // Create an anonymous tile with a constant
        else if (static_cast<int32_t>(src.type()) & 0x00001000)
        {
            if (src.type() == OperandType::ScalarTile)
            {
                ScalarTileCoord coord = src.scalar_tile_coordinate();
                assert(_tiles.has_tile(src.value()));
                assert(coord.x >= 0);
                assert(coord.y >= 0);
                auto val = _tiles[src.value()]->scalar(coord.x, coord.y);
                return _tiles.insert({{{val.str}}}, val.type.dt);
            }
            else
            {
                return _tiles.insert({{{src.value()}}}, to_tile_data_type(src.type()));
            }
        }
        // Create an anonymous tile with the tensor component
        else
        {
            assert(_arguments.has_tensor_argument(src.value()));
            auto              x   = _arguments[src.value()];
            const std::string val = x->component(to_tensor_component(src.type()));
            const DataType    dt  = x->component_data_type();
            return _tiles.insert({{{val}}}, dt);
        }
    }

private:
    DataType to_tile_data_type(OperandType x)
    {
        return static_cast<DataType>(static_cast<int32_t>(x) & 0x00ff);
    }

    TensorComponentType to_tensor_component(OperandType x)
    {
        switch (x)
        {
            case OperandType::TensorDim0:
                return TensorComponentType::Dim0;
            case OperandType::TensorDim1:
                return TensorComponentType::Dim1;
            case OperandType::TensorDim2:
                return TensorComponentType::Dim2;
            case OperandType::TensorDim3:
                return TensorComponentType::Dim3;
            case OperandType::TensorDim4:
                return TensorComponentType::Dim4;
            case OperandType::TensorStride1:
                return TensorComponentType::Stride1;
            case OperandType::TensorStride2:
                return TensorComponentType::Stride2;
            case OperandType::TensorStride3:
                return TensorComponentType::Stride3;
            case OperandType::TensorStride4:
                return TensorComponentType::Stride4;
            case OperandType::TensorDim1xDim2:
                return TensorComponentType::Dim1xDim2;
            case OperandType::TensorDim1xDim2xDim3:
                return TensorComponentType::Dim1xDim2xDim3;
            case OperandType::TensorDataOffset:
                return TensorComponentType::OffsetFirstElement;
            default:
                assert(false);
                return TensorComponentType::Unknown;
        }
    }

    GpuTileRegistry           &_tiles;
    GpuTensorArgumentRegistry &_arguments;
};

/**
 * @brief Utility class used to get the tensor argument from the operand. If the operand is not a tile, @ref OperandUnpacker
 *        declare an anonymous tile in the tile registry.
 *        Tensor dimension reduction aims for reducing the tensor data dimension while keeping data's tensor structure.
 */
class TensorOperandUnpacker
{
public:
    TensorOperandUnpacker(GpuTensorArgumentRegistry &arguments) : _arguments(arguments){};

    IGpuTensorArgument *unpack(const TensorOperand &src)
    {
        assert(_arguments.has_tensor_argument(src.value()));
        return _arguments[src.value()];
    }

private:
    GpuTensorArgumentRegistry &_arguments;
};

/**
 * @brief The GpuKernel will be used in three occasions (stages):
 * #- Compilation stage
 * #- Tuning stage
 * #- Dispatch stage
 */
struct GpuKernel
{
    // Compilation stage
    std::string                code{};            // Source code, required for the compilation stage
    std::vector<GpuExtensions> list_extensions{}; // Extensions, required for the compilation stage
    // Tuning stage
    std::string      config_id{}; // Unique id, required for the tuning stage
    std::vector<LWS> list_lws{};  // LWS to test, required for the tuning stage
    // Dispatch stage
    GpuOutputSampler output_sampler{}; // GpuOutputSampler, required for the dispatch stage
    std::vector<std::pair<int32_t, GpuTensorStorage>>
        list_tensor_storages; // List of tensor storages, required for the dispatch stage
    std::vector<std::pair<int32_t, TensorComponentType>>
        list_tensor_components; // List of tensor components (width, stride,..), required for the dispatch stage)
};

// Generate all extension pragmas (hardcoded for now)
inline std::string generate_extensions()
{
    std::string ext = R"(
#if defined(cl_khr_fp16)
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif // defined(cl_khr_fp16)

#if defined(cl_arm_integer_dot_product_int8)
#pragma OPENCL EXTENSION cl_arm_integer_dot_product_int8 : enable
#endif // defined(cl_arm_integer_dot_product_int8)

#if defined(cl_arm_integer_dot_product_accumulate_int8)
#pragma OPENCL EXTENSION cl_arm_integer_dot_product_accumulate_int8 : enable
#endif // defined(cl_arm_integer_dot_product_accumulate_int8)

#if defined(cl_arm_printf)
#pragma OPENCL EXTENSION cl_arm_printf : enable
#endif // defined(cl_arm_printf);
)";
    return ext;
}

// This function should produce an object with the source
inline std::string generate_code(GpuKernelWriterDataHolder &in, const std::string &name)
{
    std::string code;
    code += generate_extensions();
    code += "__kernel void ";
    code += name;
    code += "(\n";

    auto IdSpaces = in.arguments.IdSpace_declarations();

    std::vector<std::string> arg_str;

    auto tensor_args = in.arguments.tensor_argument_declarations();

    for (auto &i : tensor_args)
    {
        // For each tensor used, get the storage and tensor components
        auto storages   = i->storage_declarations();
        auto components = i->component_declarations();

        for (auto &y : storages)
        {
            std::string str;
            str += i->storage_type_declaration(y);
            str += " ";
            str += i->storage(y);
            arg_str.push_back(str);
        }

        for (auto &y : components)
        {
            std::string str;
            str += i->component_type_declaration();
            str += " ";
            str += i->component(y);
            arg_str.push_back(str);
        }
    }

    for (size_t i = 0; i < arg_str.size(); ++i)
    {
        code += arg_str[i];
        if (i + 1 < arg_str.size())
        {
            code += ",\n";
        }
    }

    code += ")\n";
    code += "{\n";
    code += in.code;
    code += "}\n";

    return code;
}

/**
 * @brief This class is responsible to map a N-Tensor to a 3d tensor. The mapper needs the GpuSampler to know
 * how to reduce the dimensionality of a tensor
 *
 */
class GpuTensor3dMapper
{
public:
    GpuTensor3dMapper(IGpuTensorArgument *tensor, GpuSampler sampler) : _sampler(sampler), _tensor(tensor){};

    std::string tensor_component_x() const
    {
        const auto format = _sampler.format;
        switch (format)
        {
            case TensorSamplerFormat::C_WH_1:
            case TensorSamplerFormat::C_W_H:
                return _tensor->component(TensorComponentType::Dim0);
            default:
                std::cout << "Unsupported tensor format" << std::endl;
                assert(false);
                return "";
        }
    }

    std::string tensor_component_y() const
    {
        const auto format = _sampler.format;
        switch (format)
        {
            case TensorSamplerFormat::C_WH_1:
                return _tensor->component(TensorComponentType::Dim1xDim2);
            case TensorSamplerFormat::C_W_H:
                return _tensor->component(TensorComponentType::Dim1);
            default:
                std::cout << "Unsupported tensor format" << std::endl;
                assert(false);
                return "";
        }
    }

    std::string tensor_component_z() const
    {
        const auto format = _sampler.format;
        switch (format)
        {
            case TensorSamplerFormat::C_WH_1:
                return "1";
            case TensorSamplerFormat::C_W_H:
                return _tensor->component(TensorComponentType::Dim2);
            default:
                std::cout << "Unsupported tensor format" << std::endl;
                assert(false);
                return "";
        }
    }

    std::string tensor_component_stride_y() const
    {
        const auto format = _sampler.format;
        switch (format)
        {
            case TensorSamplerFormat::C_WH_1:
            case TensorSamplerFormat::C_W_H:
                return _tensor->component(TensorComponentType::Stride1);
            default:
                std::cout << "Unsupported tensor format" << std::endl;
                assert(false);
                return "";
        }
    }

    std::string tensor_component_stride_z() const
    {
        const auto format = _sampler.format;
        switch (format)
        {
            case TensorSamplerFormat::C_WH_1:
                return "0";
            case TensorSamplerFormat::C_W_H:
                return _tensor->component(TensorComponentType::Stride2);
            default:
                std::cout << "Unsupported tensor format" << std::endl;
                assert(false);
                return "";
        }
    }

    std::string tensor_component_stride_batch() const
    {
        const auto format = _sampler.format;
        switch (format)
        {
            case TensorSamplerFormat::C_WH_1:
            case TensorSamplerFormat::C_W_H:
                return _tensor->component(TensorComponentType::Stride3);
            default:
                std::cout << "Unsupported tensor format" << std::endl;
                assert(false);
                return "";
        }
    }

    bool is_one_component_x() const
    {
        auto       t      = _tensor->format();
        const auto format = _sampler.format;
        switch (format)
        {
            case TensorSamplerFormat::C_WH_1:
            case TensorSamplerFormat::C_W_H:
                return t.shape[0] == 1;
            default:
                std::cout << "Unsupported tensor format" << std::endl;
                assert(false);
                return "";
        }
    }

    bool is_one_component_y() const
    {
        auto       t      = _tensor->format();
        const auto format = _sampler.format;
        switch (format)
        {
            case TensorSamplerFormat::C_WH_1:
                return (t.shape[1] * t.shape[2]) == 1;
            case TensorSamplerFormat::C_W_H:
                return t.shape[1] == 1;
            default:
                std::cout << "Unsupported tensor format" << std::endl;
                assert(false);
                return "";
        }
    }

    bool is_one_component_z() const
    {
        auto       t      = _tensor->format();
        const auto format = _sampler.format;
        switch (format)
        {
            case TensorSamplerFormat::C_WH_1:
                return true;
            case TensorSamplerFormat::C_W_H:
                return t.shape[2] == 1;
            default:
                std::cout << "Unsupported tensor format" << std::endl;
                assert(false);
                return "";
        }
    }

    bool is_one_component_batch() const
    {
        auto       t      = _tensor->format();
        const auto format = _sampler.format;
        switch (format)
        {
            case TensorSamplerFormat::C_WH_1:
            case TensorSamplerFormat::C_W_H:
                return t.shape[3] == 1;
            default:
                std::cout << "Unsupported tensor format" << std::endl;
                assert(false);
                return "";
        }
    }

    GpuSampler gpu_sampler() const
    {
        return _sampler;
    }

    IGpuTensorArgument *tensor_argument() const
    {
        return _tensor;
    }

private:
    GpuSampler          _sampler;
    IGpuTensorArgument *_tensor;
};

struct GpuKernelWriterAttribute
{
    bool return_tensor_component_by_value{false};
};

enum class RoundingMode
{
    None,
    Rte,
    Rtz,
    Rtp,
    Rtn
};

// https://llvm.org/docs/tutorial/MyFirstLanguageFrontend/LangImpl05.html
class IGpuKernelWriter
{
public:
    virtual ~IGpuKernelWriter() = default;

    virtual void set_IdSpace(int32_t id) = 0;

    virtual void import_tile(const std::string &dst, const IVectorTile *src) = 0;

    virtual void declare_argument(const std::string &name, const TensorInfo &tensor) = 0;

    virtual void declare_tile(const std::string &name, const TileInfo &info) = 0;

    virtual void
    declare_const_tile(const std::string &name, const std::vector<std::vector<std::string>> &in, DataType dt) = 0;

    virtual void write_text(const std::string &x) = 0;

    virtual void compound_statement_begin() = 0;

    virtual void compound_statement_end() = 0;

    // Operations
    virtual void op_get_global_id(const Operand &dst_var, int32_t dim) = 0;

    virtual void
    op_get_global_coord(const Operand &dst, const Operand &step, const TensorOperand &tensor, int32_t dim) = 0;

    virtual void op_get_global_batch(const Operand &dst, const TensorOperand &tensor) = 0;

    virtual void op_get_global_size(const Operand &dst_var, int32_t dim) = 0;

    virtual void op_unary_expression(const Operand &dst, UnaryOp op, const Operand &src) = 0;

    virtual void op_binary_expression(const Operand &dst, const Operand &lhs, BinaryOp op, const Operand &rhs) = 0;

    virtual void op_assign(const Operand &dst_name, const Operand &src_name) = 0;

    virtual void
    op_unary_elementwise_function(const Operand &dst_name, UnaryFunction func, const Operand &src_name) = 0;

    virtual void op_binary_elementwise_function(const Operand &dst_name,
                                                BinaryFunction func,
                                                const Operand &first_name,
                                                const Operand &second_name) = 0;

    virtual void op_ternary_elementwise_function(const Operand  &dst_name,
                                                 TernaryFunction func,
                                                 const Operand  &first_name,
                                                 const Operand  &second_name,
                                                 const Operand  &third_name) = 0;

    virtual void op_if_header(const Operand &lhs, BinaryOp op, const Operand &rhs) = 0;

    virtual void op_else_if_header(const Operand &lhs, BinaryOp op, const Operand &rhs) = 0;

    virtual void op_else_header() = 0;

    virtual void op_for_loop_header(const Operand &var_name,
                                    BinaryOp       cond_op,
                                    const Operand &cond_value,
                                    const Operand &update_var,
                                    AssignmentOp   update_op,
                                    const Operand &update_value) = 0;

    virtual void op_load_indirect(const TensorOperand &tensor,
                                  const Operand       &dst,
                                  const Operand       &x,
                                  const Operand       &y_indirect,
                                  const Operand       &z,
                                  const Operand       &b = Operand("0", OperandType::ScalarInt32)) = 0;

    virtual void op_load_immediate(const TensorOperand &tensor,
                                   const Operand       &dst,
                                   const Operand       &x,
                                   const Operand       &y,
                                   const Operand       &z,
                                   const Operand       &b          = Operand("0", OperandType::ScalarInt32),
                                   const Operand       &dilation_y = Operand("1", OperandType::ScalarInt32)) = 0;

    virtual void op_store_immediate(const TensorOperand &tensor,
                                    const Operand       &src,
                                    const Operand       &x,
                                    const Operand       &y,
                                    const Operand       &z,
                                    const Operand       &b = Operand("0", OperandType::ScalarInt32)) = 0;

    virtual void op_cast_expression(const Operand &dst, const Operand &src, ConvertPolicy policy) = 0;

    virtual void op_return() = 0;

    // Utils
    // It is the process of converting
    virtual void util_get_indirect_buffer(const Operand       &dst,
                                          const TensorOperand &tensor,
                                          const Operand       &x,
                                          const Operand       &y,
                                          const Operand       &x_off,
                                          const Operand       &y_off) = 0;
};

enum class GpuLoadStoreType
{
    Load  = 1,
    Store = 2
};

class IGpuLoadStoreHelperWriter
{
public:
    IGpuLoadStoreHelperWriter(IGpuKernelWriter *x, GpuTensor3dMapper mapper, GpuLoadStoreType type)
        : _writer(x), _mapper(mapper), _type(type)
    {
    }

    IGpuLoadStoreHelperWriter(const IGpuLoadStoreHelperWriter &) = default;

    IGpuLoadStoreHelperWriter &operator=(const IGpuLoadStoreHelperWriter &) = default;

    virtual ~IGpuLoadStoreHelperWriter() = default;

    virtual void initialize(IVectorTile *dst, IVectorTile *x, IVectorTile *z, IVectorTile *b) = 0;

    virtual void write(const std::pair<int32_t, std::string> &y) = 0;

    virtual void finalize() = 0;

protected:
    IGpuKernelWriter *_writer;
    GpuTensor3dMapper _mapper;
    GpuLoadStoreType  _type;
};

class ClLoadStoreBufferHelperWriter : public IGpuLoadStoreHelperWriter
{
public:
    ClLoadStoreBufferHelperWriter(IGpuKernelWriter *x, const GpuTensor3dMapper &mapper, GpuLoadStoreType type)
        : IGpuLoadStoreHelperWriter(x, mapper, type)
    {
    }

    ClLoadStoreBufferHelperWriter(const ClLoadStoreBufferHelperWriter &) = default;

    ClLoadStoreBufferHelperWriter &operator=(const ClLoadStoreBufferHelperWriter &) = default;

    static bool validate(IGpuKernelWriter *x, GpuTensor3dMapper mapper, GpuLoadStoreType type, IVectorTile *dst)
    {
        CKW_UNUSED(x, type, dst);

        if (mapper.gpu_sampler().storage != GpuSamplerTensorStorage::BufferUint8Ptr)
        {
            return false;
        }
        return true;
    }

    void initialize(IVectorTile *dst, IVectorTile *x, IVectorTile *z, IVectorTile *b) override
    {
        assert(validate(_writer, _mapper, _type, dst));

        _dst           = dst;
        _ls_width_full = dst->format().w;

        _coord_x      = x->scalar(0, 0).str;
        _coord_z      = z->scalar(0, 0).str;
        _coord_b      = b->scalar(0, 0).str;
        _coord_orig_z = _coord_z;

        out_of_bound_initialize_x(_coord_x);
        out_of_bound_initialize_z(_coord_z);

        /*
        meaning of else:
        - x: partial load/store
        - y: no load/store operation
        - z: no load/store operation
        if(x)
        {
            if(z)
            {
                if(y)
                {
                    // full load/store width
                }
                else
                {
                    // no load/store
                }
            }
            else
            {
                // no load/store
            }
        }
        else
        {
            if(z)
            {
                if(y)
                {
                    // partial load/store width
                }
                else
                {
                    // no load/store
                }
            }
            else
            {
                // no load/store
            }
        }
        */
    }

    void write(const std::pair<int32_t, std::string> &y) override
    {
        int32_t     idx_y   = y.first;
        std::string coord_y = y.second;

        // The only check required is on Y.
        out_of_bound_initialize_y(coord_y);

        const std::string dst     = _dst->vector(idx_y).str;
        const std::string address = to_ls_buffer_address(_coord_x, coord_y, _coord_z, _coord_b);
        const std::string ls_buf  = to_ls_buffer(_type, _ls_width_full, dst, address);

        _writer->write_text(ls_buf);
        _writer->write_text(";\n");

        out_of_bound_finalize_y(dst);

        // The left over load/store will be written in the finalize stage
        if (_ls_width_part.size() != 0)
        {
            int32_t w = 0;
            for (auto &p : _ls_width_part)
            {
                const std::string dst0    = _dst->vector(w, p, idx_y).str;
                const std::string coord_x = _coord_x + " + " + std::to_string(w);
                const std::string address = to_ls_buffer_address(coord_x, coord_y, _coord_z, _coord_b);
                const std::string ls_buf0 = to_ls_buffer(_type, p, dst0, address);
                _leftovers_x.push_back(std::make_pair(std::make_pair(dst0, coord_y), ls_buf0));

                w += p;
            }
        }
    }

    void finalize() override
    {
        out_of_bound_finalize_z();
        out_of_bound_finalize_x();
    }

private:
    IVectorTile                                                             *_dst{nullptr};
    int32_t                                                                  _ls_width_full{0};
    std::vector<int32_t>                                                     _ls_width_part{};
    std::vector<std::pair<std::pair<std::string, std::string>, std::string>> _leftovers_x{};
    std::string                                                              _coord_x{};
    std::string                                                              _coord_z{};
    std::string                                                              _coord_orig_z{};
    std::string                                                              _coord_b{};

    void out_of_bound_initialize_x(std::string &coord)
    {
        if (_mapper.gpu_sampler().address_mode_x == TensorSamplerAddressModeX::OverlappingMin)
        {
            auto tensor_format = _mapper.tensor_argument()->format();
            auto shape         = tensor_format.shape;

            _ls_width_part = decompose_leftover_ls_vector_width(shape[0] % _ls_width_full);
            if (_ls_width_part.size() != 0)
            {
                _writer->write_text("if(" + coord + " > 0)\n");
                _writer->compound_statement_begin();
            }
        }
    };

    void out_of_bound_finalize_x()
    {
        if (_mapper.gpu_sampler().address_mode_x == TensorSamplerAddressModeX::OverlappingMin)
        {
            if (_ls_width_part.size() != 0)
            {
                _writer->compound_statement_end();
                _writer->write_text("else\n");
                _writer->compound_statement_begin();

                out_of_bound_initialize_z(_coord_orig_z);
                for (auto &i : _leftovers_x)
                {
                    out_of_bound_initialize_y(i.first.second);
                    _writer->write_text(i.second);
                    _writer->write_text(";\n");
                    out_of_bound_finalize_y(i.first.first);
                }
                out_of_bound_finalize_z();
                _writer->compound_statement_end();
            }
        }
    };

    void out_of_bound_initialize_y(std::string &coord)
    {
        std::string max = "";

        const auto address_mode_y = _mapper.gpu_sampler().address_mode_y;

        switch (address_mode_y)
        {
            case TensorSamplerAddressModeY::Skip:
            case TensorSamplerAddressModeY::ClampToBorder:
                // NOTE: This line should not be moved outside of the switch statement.
                // The reason for that is because when we query the component, the component is marked as used
                // and added to the list of arguments of the kernel. Since, not in all cases this component is required,
                // we should request the component only when used
                max = _mapper.tensor_component_y();
                _writer->write_text("if((" + coord + " >= 0) && (" + coord + " < " + max + "))\n");
                _writer->compound_statement_begin();
                break;
            case TensorSamplerAddressModeY::SkipMinEdgeOnly:
            case TensorSamplerAddressModeY::ClampToBorderMinEdgeOnly:
                _writer->write_text("if(" + coord + " >= 0)\n");
                _writer->compound_statement_begin();
                break;
            case TensorSamplerAddressModeY::SkipMaxEdgeOnly:
            case TensorSamplerAddressModeY::ClampToBorderMaxEdgeOnly:
                max = _mapper.tensor_component_y();
                _writer->write_text("if(" + coord + " < " + max + ")\n");
                _writer->compound_statement_begin();
                break;
            case TensorSamplerAddressModeY::ClampToNearest:
                max   = _mapper.tensor_component_y();
                coord = "clamp(" + coord + ", 0, " + max + " - 1)";
                break;
            case TensorSamplerAddressModeY::ClampToMaxEdgeOnly:
                max   = _mapper.tensor_component_y();
                coord = "min(" + coord + ", " + max + " - 1)";
                break;
            case TensorSamplerAddressModeY::ClampToMinEdgeOnly:
                coord = "max(" + coord + ", 0)";
                break;
            case TensorSamplerAddressModeY::None:
                break;
            default:
                std::cout << "Unsupported address mode for write_out_of_bound_check_yz" << std::endl;
                assert(false);
        }
    };

    void out_of_bound_finalize_y(const std::string &dst)
    {
        const auto address_mode_y = _mapper.gpu_sampler().address_mode_y;

        switch (address_mode_y)
        {
            case TensorSamplerAddressModeY::ClampToBorder:
            case TensorSamplerAddressModeY::ClampToBorderMaxEdgeOnly:
            case TensorSamplerAddressModeY::ClampToBorderMinEdgeOnly:
            case TensorSamplerAddressModeY::Skip:
            case TensorSamplerAddressModeY::SkipMaxEdgeOnly:
            case TensorSamplerAddressModeY::SkipMinEdgeOnly:
                _writer->compound_statement_end();
                break;
            case TensorSamplerAddressModeY::None:
                break;

            default:
                assert(false);
        }

        switch (address_mode_y)
        {
            case TensorSamplerAddressModeY::ClampToBorder:
            case TensorSamplerAddressModeY::ClampToBorderMinEdgeOnly:
            case TensorSamplerAddressModeY::ClampToBorderMaxEdgeOnly:
                _writer->write_text("else\n");
                _writer->compound_statement_begin();
                _writer->write_text(dst);
                _writer->write_text(" = 0.0f;\n");
                _writer->compound_statement_end();
                break;
            case TensorSamplerAddressModeY::None:
                break;

            default:
                assert(false);
        }
    };

    void out_of_bound_initialize_z(std::string &coord)
    {
        std::string max = "";

        const auto address_mode_z = _mapper.gpu_sampler().address_mode_z;

        switch (address_mode_z)
        {
            case TensorSamplerAddressModeZ::Skip:
                max = _mapper.tensor_component_z();
                _writer->write_text("if((" + coord + " >= 0) && (" + coord + " < " + max + "))\n");
                _writer->compound_statement_begin();
                break;
            case TensorSamplerAddressModeZ::SkipMinEdgeOnly:
                _writer->write_text("if(" + coord + " >= 0)\n");
                _writer->compound_statement_begin();
                break;
            case TensorSamplerAddressModeZ::SkipMaxEdgeOnly:
                max = _mapper.tensor_component_z();
                _writer->write_text("if(" + coord + " < " + max + ")\n");
                _writer->compound_statement_begin();
                break;
            case TensorSamplerAddressModeZ::ClampToNearest:
                max   = _mapper.tensor_component_z();
                coord = "clamp(" + coord + ", 0, " + max + " - 1)";
                break;
            case TensorSamplerAddressModeZ::ClampToMaxEdgeOnly:
                max   = _mapper.tensor_component_z();
                coord = "min(" + coord + ", " + max + " - 1)";
                break;
            case TensorSamplerAddressModeZ::ClampToMinEdgeOnly:
                coord = "max(" + coord + ", 0)";
                break;
            case TensorSamplerAddressModeZ::None:
                break;
            default:
                std::cout << "Unsupported address mode for write_out_of_bound_check_yz" << std::endl;
                assert(false);
        }
    };

    void out_of_bound_finalize_z()
    {
        const auto address_mode_z = _mapper.gpu_sampler().address_mode_z;

        switch (address_mode_z)
        {
            case TensorSamplerAddressModeZ::Skip:
            case TensorSamplerAddressModeZ::SkipMinEdgeOnly:
            case TensorSamplerAddressModeZ::SkipMaxEdgeOnly:
                _writer->compound_statement_end();
                break;
            case TensorSamplerAddressModeZ::None:
                break;

            default:
                assert(false);
        }
    };

    std::vector<int32_t> decompose_leftover_ls_vector_width(int32_t ls_leftover_vector_width) const
    {
        std::vector<int32_t> x;

        switch (ls_leftover_vector_width)
        {
            case 0:
                break;
            case 1:
            case 2:
            case 3:
            case 4:
            case 8:
            case 16:
                x.push_back(ls_leftover_vector_width);
                break;
            case 5:
                x.push_back(4);
                x.push_back(1);
                break;
            case 6:
                x.push_back(4);
                x.push_back(2);
                break;
            case 7:
                x.push_back(4);
                x.push_back(3);
                break;
            case 9:
                x.push_back(8);
                x.push_back(1);
                break;
            case 10:
                x.push_back(8);
                x.push_back(2);
                break;
            case 11:
                x.push_back(8);
                x.push_back(3);
                break;
            case 12:
                x.push_back(8);
                x.push_back(4);
                break;
            case 13:
                x.push_back(8);
                x.push_back(4);
                x.push_back(1);
                break;
            case 14:
                x.push_back(8);
                x.push_back(4);
                x.push_back(2);
                break;
            case 15:
                x.push_back(8);
                x.push_back(4);
                x.push_back(3);
                break;

            default:
                assert(false);
        }
        return x;
    }

    std::string
    to_ls_buffer(GpuLoadStoreType type, int32_t vector_width, const std::string &data, const std::string &address)
    {
        switch (type)
        {
            case GpuLoadStoreType::Load:
                if (vector_width != 1)
                {
                    return data + " = vload" + std::to_string(vector_width) + "(0, " + address + ")";
                }
                else
                {
                    return data + " = *(" + address + ")";
                }
                break;
            case GpuLoadStoreType::Store:
                if (vector_width != 1)
                {
                    return "vstore" + std::to_string(vector_width) + "(" + data + ", 0, " + address + ")";
                }
                else
                {
                    return "*(" + address + ") = " + data;
                }
                break;
            default:
                std::cout << "Unsupported GpuLoadStoreType" << std::endl;
                assert(false);
                return "";
        }
    }

    std::string
    to_ls_buffer_address(const std::string &x, const std::string &y, const std::string &z, const std::string &b) const
    {
        auto tensor_storage = static_cast<GpuTensorStorage>(_mapper.gpu_sampler().storage);
        assert(tensor_storage == GpuTensorStorage::BufferUint8Ptr);
        const std::string ptr_buf  = _mapper.tensor_argument()->storage(tensor_storage);
        const std::string dst_type = get_cl_data_type(_dst->format().dt, 1);

        std::string address;
        address += "(__global ";
        address += dst_type;
        address += "*)(";
        address += ptr_buf;
        if (x != "0" && (_mapper.is_one_component_x() != true))
        {
            address += " + (";
            address += x + ") * sizeof(" + dst_type + ")";
        }
        if (y != "0")
        {
            const std::string stride_y = _mapper.tensor_component_stride_y();
            address += " + (";
            address += y + ")";
            address += " * ";
            address += stride_y;
        }
        if (z != "0" && (_mapper.is_one_component_z() != true))
        {
            const std::string stride_z = _mapper.tensor_component_stride_z();
            address += " + (";
            address += z + ")";
            address += " * ";
            address += stride_z;
        }
        if (b != "0" && (_mapper.is_one_component_batch() != true))
        {
            const std::string stride_b = _mapper.tensor_component_stride_batch();
            address += " + (";
            address += b + ")";
            address += " * ";
            address += stride_b;
        }
        address += ")";
        return address;
    }
};

class ClLoadStoreImage2dHelperWriter : public IGpuLoadStoreHelperWriter
{
public:
    static bool validate(IGpuKernelWriter *x, const GpuTensor3dMapper &mapper, GpuLoadStoreType type, IVectorTile *dst)
    {
        CKW_UNUSED(x);

        if (dst->format().w != 4)
        {
            return false;
        }
        if (mapper.gpu_sampler().address_mode_x != TensorSamplerAddressModeX::None)
        {
            return false;
        }
        if (mapper.gpu_sampler().address_mode_z != TensorSamplerAddressModeZ::None)
        {
            return false;
        }
        if (mapper.gpu_sampler().storage != GpuSamplerTensorStorage::Image2dReadOnly && type == GpuLoadStoreType::Load)
        {
            return false;
        }
        if (mapper.gpu_sampler().storage != GpuSamplerTensorStorage::Image2dWriteOnly &&
            type == GpuLoadStoreType::Store)
        {
            return false;
        }
        if ((dst->format().dt != DataType::Fp32) && (dst->format().dt != DataType::Fp16))
        {
            return false;
        }
        return true;
        /*
        - x: Only GpuSamplerAddressModeX::None is supported and vector length = 4
        - z: Only GpuSamplerAddressModeZ::None is supported
        */
    }

    ClLoadStoreImage2dHelperWriter(IGpuKernelWriter *x, const GpuTensor3dMapper &mapper, GpuLoadStoreType type)
        : IGpuLoadStoreHelperWriter(x, mapper, type)
    {
    }

    ClLoadStoreImage2dHelperWriter(const ClLoadStoreImage2dHelperWriter &) = default;

    ClLoadStoreImage2dHelperWriter &operator=(const ClLoadStoreImage2dHelperWriter &) = default;

    void initialize(IVectorTile *dst, IVectorTile *x, IVectorTile *z, IVectorTile *b) override
    {
        assert(validate(_writer, _mapper, _type, dst));

        _dst           = dst;
        _ls_width_full = dst->format().w;
        _coord_x       = x->scalar(0, 0).str;
        _coord_z       = z->scalar(0, 0).str;
        _coord_b       = b->scalar(0, 0).str;

        /*
        if(y)
        {
            // full load/store width
        }
        else
        {
            // no load/store
        }
        */
    }

    void write(const std::pair<int32_t, std::string> &y) override
    {
        int32_t     idx_y   = y.first;
        std::string coord_y = y.second;

        // The only check required is on Y.
        out_of_bound_initialize_y(coord_y);

        const std::string dst     = _dst->vector(idx_y).str;
        const std::string sampler = to_ls_image2d_sampler();
        const std::string coord   = to_ls_image2d_coord(_coord_x, coord_y, _coord_z, _coord_b);
        const std::string ls_buf  = to_ls_image2d(_type, _ls_width_full, dst, sampler, coord);

        _writer->write_text(ls_buf);
        _writer->write_text(";\n");

        out_of_bound_finalize_y(dst);
    }

    void finalize() override
    {
    }

private:
    IVectorTile *_dst{nullptr};
    int32_t      _ls_width_full{0};
    std::string  _coord_x{};
    std::string  _coord_z{};
    std::string  _coord_b{};

    void out_of_bound_initialize_y(std::string &coord)
    {
        std::string max = "";

        const auto address_mode_y = _mapper.gpu_sampler().address_mode_y;

        switch (address_mode_y)
        {
            case TensorSamplerAddressModeY::Skip:
                max = _mapper.tensor_component_y();
                _writer->write_text("if((" + coord + " >= 0) && (" + coord + " < " + max + "))\n");
                _writer->compound_statement_begin();
                break;
            case TensorSamplerAddressModeY::SkipMinEdgeOnly:
                _writer->write_text("if(" + coord + " >= 0)\n");
                _writer->compound_statement_begin();
                break;
            case TensorSamplerAddressModeY::SkipMaxEdgeOnly:
                max = _mapper.tensor_component_y();
                _writer->write_text("if(" + coord + " < " + max + ")\n");
                _writer->compound_statement_begin();
                break;
            case TensorSamplerAddressModeY::ClampToBorder:
            case TensorSamplerAddressModeY::ClampToBorderMinEdgeOnly:
            case TensorSamplerAddressModeY::ClampToBorderMaxEdgeOnly:
            case TensorSamplerAddressModeY::ClampToNearest:
            case TensorSamplerAddressModeY::ClampToMaxEdgeOnly:
            case TensorSamplerAddressModeY::ClampToMinEdgeOnly:
            case TensorSamplerAddressModeY::None:
                break;
            default:
                std::cout << "Unsupported address mode for write_out_of_bound_check_y" << std::endl;
                assert(false);
        }
    };

    void out_of_bound_finalize_y(const std::string &dst)
    {
        CKW_UNUSED(dst);

        const auto address_mode_y = _mapper.gpu_sampler().address_mode_y;

        switch (address_mode_y)
        {
            case TensorSamplerAddressModeY::Skip:
            case TensorSamplerAddressModeY::SkipMinEdgeOnly:
            case TensorSamplerAddressModeY::SkipMaxEdgeOnly:
                _writer->compound_statement_end();
                break;

            default:
                assert(false);
        }
    };

    std::string to_ls_image2d(GpuLoadStoreType   type,
                              int32_t            vector_width,
                              const std::string &data,
                              const std::string &sampler,
                              const std::string &coord)
    {
        CKW_UNUSED(vector_width);

        auto              tensor_storage = static_cast<GpuTensorStorage>(_mapper.gpu_sampler().storage);
        const std::string image2d_obj    = _mapper.tensor_argument()->storage(tensor_storage);
        const std::string post_fix       = _dst->format().dt == DataType::Fp32 ? "f" : "h";

        switch (type)
        {
            case GpuLoadStoreType::Load:
                return data + " = read_image" + post_fix + "(" + image2d_obj + ", " + sampler + ", " + coord + ")";
                break;
            case GpuLoadStoreType::Store:
                return "write_image" + post_fix + "(" + image2d_obj + ", " + coord + ", " + data + ")";
            default:
                assert(false);
                std::cout << "Unsupported GpuLoadStoreType" << std::endl;
                assert(false);
                return "";
        }
    }

    std::string to_ls_image2d_sampler() const
    {
        const auto address_mode_y = _mapper.gpu_sampler().address_mode_y;

        switch (address_mode_y)
        {
            case TensorSamplerAddressModeY::None:
                return "CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST";
            case TensorSamplerAddressModeY::Skip:
            case TensorSamplerAddressModeY::SkipMinEdgeOnly:
            case TensorSamplerAddressModeY::SkipMaxEdgeOnly:
            case TensorSamplerAddressModeY::ClampToBorder:
            case TensorSamplerAddressModeY::ClampToBorderMinEdgeOnly:
            case TensorSamplerAddressModeY::ClampToBorderMaxEdgeOnly:
                return "CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST";
            case TensorSamplerAddressModeY::ClampToNearest:
            case TensorSamplerAddressModeY::ClampToMaxEdgeOnly:
            case TensorSamplerAddressModeY::ClampToMinEdgeOnly:
                return "CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST";
            default:
                std::cout << "Unsupported address_mode_coord" << std::endl;
                assert(false);
                return "";
        }
    }

    std::string
    to_ls_image2d_coord(const std::string &x, const std::string &y, const std::string &z, const std::string &b) const
    {
        std::string coord_x = "(" + x + ") >> 2";
        std::string coord_y = "(";

        if (y != "0")
        {
            coord_y += y;
        }
        if (z != "0" && (_mapper.is_one_component_z() != true))
        {
            const std::string dim = _mapper.tensor_component_y();
            coord_y += " + (";
            coord_y += z + ")";
            coord_y += " * ";
            coord_y += dim;
        }
        if (b != "0" && (_mapper.is_one_component_batch() != true))
        {
            const std::string dim0 = _mapper.tensor_component_y();
            const std::string dim1 = _mapper.tensor_component_z();
            coord_y += " + (";
            coord_y += b + ")";
            coord_y += " * ";
            coord_y += dim0;
            coord_y += " * ";
            coord_y += dim1;
        }
        coord_y += ")";
        return "(int2)(" + coord_x + ", " + coord_y + ")";
    }
};

/** IGpuLoadStoreHelperWriter factory class */
class ClLoadStoreHelperWriterFactory final
{
public:
    /** Static method to call the IGpuLoadStoreHelperWriter class accordingly with the tensor storage set in the mapper
     *
     *
     * @return IGpuLoadStoreHelperWriter
     */
    static std::unique_ptr<IGpuLoadStoreHelperWriter>
    create(IGpuKernelWriter *x, const GpuTensor3dMapper &mapper, GpuLoadStoreType type)
    {
        const auto tensor_storage = mapper.gpu_sampler().storage;
        switch (tensor_storage)
        {
            case GpuSamplerTensorStorage::BufferUint8Ptr:
                return std::make_unique<ClLoadStoreBufferHelperWriter>(x, mapper, type);
            case GpuSamplerTensorStorage::Image2dReadOnly:
            case GpuSamplerTensorStorage::Image2dWriteOnly:
                return std::make_unique<ClLoadStoreImage2dHelperWriter>(x, mapper, type);
            default:
                std::cout << "Unsupported Gpu tensor storage" << std::endl;
                assert(false);
                return nullptr;
        }
    }
};

// This utility method needs to go in utils.h
inline bool is_tile_scalar(const IVectorTile *x)
{
    return x->format().w == 1 && x->format().h == 1;
}

class ClKernelWriter : public IGpuKernelWriter
{
public:
    ClKernelWriter(GpuKernelWriterAttribute *attr, GpuKernelWriterDataHolder *x)
    {
        _data = x;
        _attr = attr;
    }

    ClKernelWriter(const ClKernelWriter &) = default;

    ClKernelWriter &operator=(const ClKernelWriter &) = default;

    // A IdSpaced ID is a term used to describe a fragment that is registered in ICode to ensure
    // there are no conflicts or ambiguity in the code
    void set_IdSpace(int32_t id) override
    {
        _data->tiles.set_IdSpace(id);
        _data->arguments.set_IdSpace(id);
    }

    void import_tile(const std::string &dst_name, const IVectorTile *src) override
    {
        _data->tiles.insert(dst_name, src);
    }

    void declare_argument(const std::string &name, const TensorInfo &tensor) override
    {
        assert(_data->arguments[name] == nullptr);
        _data->arguments.insert(name, tensor, _attr->return_tensor_component_by_value);
    }

    void declare_tile(const std::string &name, const TileInfo &format) override
    {
        assert(_data->tiles[name] == nullptr);
        _data->tiles.insert(name, format);

        IVectorTile *x = _data->tiles[name];

        for (auto &t : x->underlying_source_variables())
        {
            _data->code += t.type.str + " " + t.str + ";\n";
        }
    }

    void
    declare_const_tile(const std::string &name, const std::vector<std::vector<std::string>> &in, DataType dt) override
    {
        assert(_data->tiles[name] == nullptr);
        _data->tiles.insert(name, in, dt);
        // Note: A constant does not need to be declared in the code
    }

    void write_text(const std::string &x) override
    {
        _data->code += x;
    }

    void compound_statement_begin() override
    {
        _data->tiles.increment_registry_level();
        _data->code += "{\n";
    }

    void compound_statement_end() override
    {
        _data->tiles.decrement_registry_level();
        _data->code += "}\n";
    }

    void op_get_global_id(const Operand &dst_var, int32_t dim) override
    {
        assert(dst_var.type() == OperandType::Tile);
        assert(_data->tiles.has_tile(dst_var.value()));
        assert(_data->tiles[dst_var.value()]->format().w == 1 &&
               _data->tiles[dst_var.value()]->format().h == 1); // It must be a scalar variable

        auto var = _data->tiles[dst_var.value()];

        _data->code += var->scalar(0, 0).str;
        _data->code += " = get_global_id(";
        _data->code += std::to_string(dim);
        _data->code += ");\n";
    };

    void op_get_global_coord(const Operand       &o_dst,
                             const Operand       &o_step,
                             const TensorOperand &o_tensor,
                             int32_t              dim) override
    {
        OperandUnpacker operands(_data->tiles, _data->arguments);
        auto            dst  = operands.unpack(o_dst);
        auto            step = operands.unpack(o_step);

        // Validation: Check that x, y and z are scalar

        TensorOperandUnpacker tensor_operands(_data->arguments);
        auto                  tensor      = tensor_operands.unpack(o_tensor);
        auto                  gpu_sampler = o_tensor.sampler();

        GpuTensor3dMapper mapper(tensor, gpu_sampler);

        switch (dim)
        {
            case 0:
                if (mapper.is_one_component_x())
                {
                    _data->code += dst->scalar(0, 0).str;
                    _data->code += " = 0;\n";
                }
                else
                {
                    if (mapper.gpu_sampler().address_mode_x == TensorSamplerAddressModeX::OverlappingMin)
                    {
                        // Validation: Check: fixed tensor shape
                        // TO BE CHANGED
                        _data->code += dst->scalar(0, 0).str;
                        _data->code += " = get_global_id(0) * ";
                        _data->code += step->scalar(0, 0).str;
                        _data->code += ";\n";
                    }
                    else
                    {
                        _data->code += dst->scalar(0, 0).str;
                        _data->code += " = get_global_id(0) * ";
                        _data->code += step->scalar(0, 0).str;
                        _data->code += ";\n";
                    }
                }
                break;
            case 1:
                if (mapper.is_one_component_y())
                {
                    _data->code += dst->scalar(0, 0).str;
                    _data->code += " = 0;\n";
                }
                else
                {
                    if (mapper.gpu_sampler().address_mode_y == TensorSamplerAddressModeY::OverlappingMin)
                    {
                    }
                    else
                    {
                        _data->code += dst->scalar(0, 0).str;
                        _data->code += " = get_global_id(1) * ";
                        _data->code += step->scalar(0, 0).str;
                        _data->code += ";\n";
                    }
                }
                break;
            case 2:
                if (mapper.is_one_component_z())
                {
                    _data->code += dst->scalar(0, 0).str;
                    _data->code += " = 0;\n";
                }
                else
                {
                    _data->code += dst->scalar(0, 0).str;
                    _data->code += " = get_global_id(2) * ";
                    _data->code += step->scalar(0, 0).str;
                    _data->code += ";\n";
                }
                break;
            default:
                break;
        }
    };

    void op_get_global_batch(const Operand &o_dst, const TensorOperand &o_tensor) override
    {
        OperandUnpacker    operands(_data->tiles, _data->arguments);
        const IVectorTile *dst = operands.unpack(o_dst);

        TensorOperandUnpacker tensor_operands(_data->arguments);
        IGpuTensorArgument   *tensor      = tensor_operands.unpack(o_tensor);
        auto                  gpu_sampler = o_tensor.sampler();

        GpuTensor3dMapper mapper(tensor, gpu_sampler);

        if (mapper.is_one_component_batch())
        {
            _data->code += dst->scalar(0, 0).str;
            _data->code += " = 0;\n";
        }
        else
        {
            std::cout << "Unsupported batched computation" << std::endl;
            assert(false);
        }
    };

    void op_get_global_size(const Operand &dst_var, int32_t dim) override
    {
        assert(dst_var.type() == OperandType::Tile);
        assert(_data->tiles.has_tile(dst_var.value()));
        assert(_data->tiles[dst_var.value()]->format().w == 1 &&
               _data->tiles[dst_var.value()]->format().h == 1); // It must be a scalar variable

        auto var = _data->tiles[dst_var.value()];

        _data->code += var->scalar(0, 0).str;
        _data->code += " = get_global_size(";
        _data->code += std::to_string(dim);
        _data->code += ");\n";
    }

    void op_unary_expression(const Operand &dst_name, UnaryOp op, const Operand &src_name) override
    {
        OperandUnpacker    operands(_data->tiles, _data->arguments);
        const IVectorTile *src = operands.unpack(src_name);
        const IVectorTile *dst = operands.unpack(dst_name);

        const int32_t     dst_w = dst->format().w;
        const int32_t     dst_h = dst->format().h;
        const int32_t     src_w = src->format().w;
        const std::string dt    = dst->underlying_source_variables()[0].type.str;

        const bool broadcast_src_x = dst_w != 1 && src_w == 1;

        const std::string src_prefix = broadcast_src_x ? "(" + dt + ")" : "";

        // Broadcasting on Y is automatic
        for (int32_t y = 0; y < dst_h; ++y)
        {
            _data->code += dst->vector(y).str;
            _data->code += " = ";
            _data->code += to_string(op);
            _data->code += src_prefix + src->vector(y).str;
            _data->code += ";\n";
        }
    }

    void op_binary_expression(const Operand &dst_name,
                              const Operand &lhs_name,
                              BinaryOp       op,
                              const Operand &rhs_name) override
    {
        OperandUnpacker    operands(_data->tiles, _data->arguments);
        const IVectorTile *lhs = operands.unpack(lhs_name);
        const IVectorTile *rhs = operands.unpack(rhs_name);
        const IVectorTile *dst = operands.unpack(dst_name);

        const int32_t dst_w = dst->format().w;
        const int32_t dst_h = dst->format().h;
        assert(lhs != nullptr);
        const int32_t lhs_w = lhs->format().w;
        const int32_t rhs_w = rhs->format().w;

        if (op == BinaryOp::MatMul_Nt_T)
        {
            assert((dst->format().dt == DataType::Fp32) || (dst->format().dt == DataType::Fp16));
            for (int32_t y = 0; y < dst_h; ++y)
            {
                for (int32_t x = 0; x < dst_w; ++x)
                {
                    for (int32_t k = 0; k < lhs_w; ++k)
                    {
                        _data->code += dst->scalar(x, y).str;
                        _data->code += " = fma(";
                        _data->code += lhs->scalar(k, y).str;
                        _data->code += ", ";
                        _data->code += rhs->scalar(k, x).str;
                        _data->code += ", ";
                        _data->code += dst->scalar(x, y).str;
                        _data->code += ");\n";
                    }
                }
            }

            return;
        }

        const bool broadcast_lhs_x = dst_w != 1 && lhs_w == 1;
        const bool broadcast_rhs_x = dst_w != 1 && rhs_w == 1;

        const std::string lhs_prefix =
            broadcast_lhs_x ? "(" + dst->underlying_source_variables()[0].type.str + ")" : "";
        const std::string rhs_prefix =
            broadcast_rhs_x ? "(" + dst->underlying_source_variables()[0].type.str + ")" : "";
        const std::string op_str = to_string(op);

        // Broadcasting on Y is automatic
        for (int32_t y = 0; y < dst_h; ++y)
        {
            _data->code += dst->vector(y).str;
            _data->code += " = ";
            _data->code += lhs_prefix + lhs->vector(y).str;
            _data->code += " ";
            _data->code += op_str;
            _data->code += " ";
            _data->code += rhs_prefix + rhs->vector(y).str;
            _data->code += ";\n";
        }
    };

    void op_cast_expression(const Operand &o_dst, const Operand &o_src, ConvertPolicy policy) override
    {
        OperandUnpacker    operands(_data->tiles, _data->arguments);
        const IVectorTile *src = operands.unpack(o_src);
        const IVectorTile *dst = operands.unpack(o_dst);
        // const int32_t dst_w  = dst->format().w;
        const int32_t     dst_h    = dst->format().h;
        const std::string dt       = dst->underlying_source_variables()[0].type.str;
        const bool        is_float = (dst->format().dt == DataType::Fp32) || (dst->format().dt == DataType::Fp16);
        const std::string sat      = ((policy == ConvertPolicy::Saturate && !is_float) ? "_sat" : "");

        // Broadcasting on Y is automatic
        for (int32_t y = 0; y < dst_h; ++y)
        {
            _data->code += dst->vector(y).str;
            _data->code += " = convert_" + dt + sat + "(";
            _data->code += src->vector(y).str;
            _data->code += ");\n";
        }
    };

    void op_assign(const Operand &dst_name, const Operand &src_name) override
    {
        OperandUnpacker    operands(_data->tiles, _data->arguments);
        const IVectorTile *src = operands.unpack(src_name);
        const IVectorTile *dst = operands.unpack(dst_name);

        const int32_t     dst_w = dst->format().w;
        const int32_t     dst_h = dst->format().h;
        const int32_t     src_w = src->format().w;
        const std::string dt    = dst->underlying_source_variables()[0].type.str;

        const bool broadcast_src_x = dst_w != 1 && src_w == 1;

        const std::string src_prefix = broadcast_src_x ? "(" + dt + ")" : "";

        // Broadcasting on Y is automatic
        for (int32_t y = 0; y < dst_h; ++y)
        {
            _data->code += dst->vector(y).str;
            _data->code += " = ";
            _data->code += src_prefix + src->vector(y).str;
            _data->code += ";\n";
        }
    }

    void op_unary_elementwise_function(const Operand &dst_name, UnaryFunction func, const Operand &src_name) override
    {
        OperandUnpacker    operands(_data->tiles, _data->arguments);
        const IVectorTile *src = operands.unpack(src_name);
        const IVectorTile *dst = operands.unpack(dst_name);

        const int32_t     dst_h = dst->format().h;
        const std::string dt    = dst->underlying_source_variables()[0].type.str;

        // Always perform an explicit cast. This automatically covers at least the 2 scenarios:
        // 1. Widen a scalar into a vector type. This enables scalar-vector broadcasting
        // 2. Ensure non-ambiguity over function overloads.
        //    E.g. a constant tile may be accidentally initialized with a double literal. By casting it to single float,
        //    it avoids ambiguous function calls
        const std::string src_prefix = "(" + dt + ")";

        // Broadcasting on Y is automatic
        for (int32_t y = 0; y < dst_h; ++y)
        {
            _data->code += dst->vector(y).str;
            _data->code += " = ";

            switch (func)
            {
                case UnaryFunction::Exp:
                    _data->code += "exp(";
                    break;
                case UnaryFunction::Tanh:
                    _data->code += "tanh(";
                    break;
                case UnaryFunction::Sqrt:
                    _data->code += "sqrt(";
                    break;
                case UnaryFunction::Erf:
                    _data->code += "erf(";
                    break;
                case UnaryFunction::Fabs:
                    _data->code += "fabs(";
                    break;
                case UnaryFunction::Log:
                    _data->code += "log(";
                    break;
                case UnaryFunction::SizeOf:
                    _data->code += "sizeof(";
                    break;
                case UnaryFunction::Round:
                    _data->code += "round(";
                    break;
                case UnaryFunction::Floor:
                    _data->code += "floor(";
                    break;
                default:
                    CKW_ASSERT_MSG(false, "Unexpected UnaryFunction used.");
            }

            _data->code += src_prefix + src->vector(y).str;
            _data->code += ");\n";
        }
    }

    void op_binary_elementwise_function(const Operand &dst_name,
                                        BinaryFunction func,
                                        const Operand &first_name,
                                        const Operand &second_name) override
    {
        OperandUnpacker    operands(_data->tiles, _data->arguments);
        const IVectorTile *first  = operands.unpack(first_name);
        const IVectorTile *second = operands.unpack(second_name);
        const IVectorTile *dst    = operands.unpack(dst_name);

        const int32_t     dst_h        = dst->format().h;
        const auto        datatype     = dst->underlying_source_variables()[0].type;
        const std::string datatype_str = datatype.str;

        // Always perform an explicit cast. See similar comments in op_unary_elementwise_function
        const std::string first_prefix  = "(" + datatype_str + ")";
        const std::string second_prefix = "(" + datatype_str + ")";

        const bool is_float = (datatype.dt == DataType::Fp32 || datatype.dt == DataType::Fp16);

        // Broadcasting on Y is automatic
        for (int32_t y = 0; y < dst_h; ++y)
        {
            _data->code += dst->vector(y).str;
            _data->code += " = ";

            switch (func)
            {
                case BinaryFunction::Min:
                    _data->code += is_float ? "fmin(" : "min(";
                    break;
                case BinaryFunction::Max:
                    _data->code += is_float ? "fmax(" : "max(";
                    break;
                default:
                    CKW_ASSERT_MSG(false, "Unexpected BinaryFunction used.");
            }

            _data->code += first_prefix + first->vector(y).str;
            _data->code += ", ";
            _data->code += second_prefix + second->vector(y).str;
            _data->code += ");\n";
        }
    }

    void op_ternary_elementwise_function(const Operand  &dst_name,
                                         TernaryFunction func,
                                         const Operand  &first_name,
                                         const Operand  &second_name,
                                         const Operand  &third_name) override
    {
        OperandUnpacker    operands(_data->tiles, _data->arguments);
        const IVectorTile *first  = operands.unpack(first_name);
        const IVectorTile *second = operands.unpack(second_name);
        const IVectorTile *third  = operands.unpack(third_name);
        const IVectorTile *dst    = operands.unpack(dst_name);

        const int32_t     dst_h = dst->format().h;
        const std::string dt    = dst->underlying_source_variables()[0].type.str;

        // Always perform an explicit cast. See similar comments in op_unary_elementwise_function
        const std::string first_prefix  = "(" + dt + ")";
        const std::string second_prefix = "(" + dt + ")";
        const std::string third_prefix  = "(" + dt + ")";

        // Broadcasting on Y is automatic
        for (int32_t y = 0; y < dst_h; ++y)
        {
            _data->code += dst->vector(y).str;
            _data->code += " = ";

            switch (func)
            {
                case TernaryFunction::Select:
                    _data->code += "select(";
                    break;
                case TernaryFunction::Clamp:
                    _data->code += "clamp(";
                    break;
                default:
                    CKW_ASSERT_MSG(false, "Unexpected TernaryFunction used.");
            }

            _data->code += first_prefix + first->vector(y).str;
            _data->code += ", ";
            _data->code += second_prefix + second->vector(y).str;
            _data->code += ", ";
            _data->code += third_prefix + third->vector(y).str;
            _data->code += ");\n";
        }
    }

    void op_if_header(const Operand &o_lhs, BinaryOp op, const Operand &o_rhs) override
    {
        OperandUnpacker    operands(_data->tiles, _data->arguments);
        const IVectorTile *lhs = operands.unpack(o_lhs);
        const IVectorTile *rhs = operands.unpack(o_rhs);

        assert(is_tile_scalar(lhs));
        assert(is_tile_scalar(rhs));

        _data->code += "if(";
        _data->code += lhs->scalar(0, 0).str;
        _data->code += " ";
        _data->code += to_string(op);
        _data->code += " ";
        _data->code += rhs->scalar(0, 0).str;
        _data->code += ")\n";
    }

    void op_else_if_header(const Operand &o_lhs, BinaryOp op, const Operand &o_rhs) override
    {
        _data->code += "else ";
        op_if_header(o_lhs, op, o_rhs);
    }

    void op_else_header() override
    {
        _data->code += "else\n";
    }

    void op_for_loop_header(const Operand &var_name,
                            BinaryOp       cond_op,
                            const Operand &cond_value_name,
                            const Operand &update_var_name,
                            AssignmentOp   update_op,
                            const Operand &update_value_name) override
    {
        OperandUnpacker    operands(_data->tiles, _data->arguments);
        const IVectorTile *var          = operands.unpack(var_name);
        const IVectorTile *cond_value   = operands.unpack(cond_value_name);
        const IVectorTile *update_var   = operands.unpack(update_var_name);
        const IVectorTile *update_value = operands.unpack(update_value_name);

        const int32_t dst_w = var->format().w;
        const int32_t dst_h = var->format().h;

        // It must be a scalar variable
        CKW_UNUSED(dst_w, dst_h);
        assert(dst_w == 1);
        assert(dst_h == 1);

        _data->code += "for(; ";
        _data->code += var->scalar(0, 0).str;
        _data->code += " ";
        _data->code += to_string(cond_op);
        _data->code += " " + cond_value->scalar(0, 0).str + "; ";
        _data->code += update_var->scalar(0, 0).str;
        _data->code += " ";
        _data->code += to_string(update_op);
        _data->code += " " + update_value->scalar(0, 0).str + ")";
        _data->code += "\n";
    }

    void op_load_immediate(const TensorOperand &o_tensor,
                           const Operand       &o_dst,
                           const Operand       &o_x,
                           const Operand       &o_y,
                           const Operand       &o_z,
                           const Operand       &o_batch_idx,
                           const Operand       &dilation_y) override
    {
        OperandUnpacker operands(_data->tiles, _data->arguments);

        // Not const as it requires changes to 'load_writer'.
        IVectorTile *dst   = operands.unpack(o_dst);
        IVectorTile *x     = operands.unpack(o_x);
        IVectorTile *y     = operands.unpack(o_y);
        IVectorTile *z     = operands.unpack(o_z);
        IVectorTile *dil_y = operands.unpack(dilation_y);
        IVectorTile *b     = operands.unpack(o_batch_idx);

        TensorOperandUnpacker tensor_operands(_data->arguments);
        IGpuTensorArgument   *tensor      = tensor_operands.unpack(o_tensor);
        auto                  gpu_sampler = o_tensor.sampler();

        GpuTensor3dMapper mapper(tensor, gpu_sampler);

        auto load_writer = ClLoadStoreHelperWriterFactory::create(this, mapper, GpuLoadStoreType::Load);

        // Initialize the constant part
        load_writer->initialize(dst, x, z, b);

        for (int i = 0; i < dst->format().h; ++i)
        {
            std::string coord_y = y->scalar(0, 0).str + " + " + std::to_string(i);
            if (dil_y->scalar(0, 0).str != "1")
            {
                coord_y += " * " + dil_y->scalar(0, 0).str;
            }
            load_writer->write(std::make_pair(i, coord_y));
        }

        load_writer->finalize();
    }

    void op_load_indirect(const TensorOperand &o_tensor,
                          const Operand       &o_dst,
                          const Operand       &o_x,
                          const Operand       &o_indirect_h,
                          const Operand       &o_z,
                          const Operand       &o_batch_idx) override
    {
        OperandUnpacker operands(_data->tiles, _data->arguments);

        // Not const as it requires changes to 'load_writer'.
        IVectorTile *dst   = operands.unpack(o_dst);
        IVectorTile *x     = operands.unpack(o_x);
        IVectorTile *y_ind = operands.unpack(o_indirect_h);
        IVectorTile *z     = operands.unpack(o_z);
        IVectorTile *b     = operands.unpack(o_batch_idx);

        TensorOperandUnpacker tensor_operands(_data->arguments);
        IGpuTensorArgument   *tensor      = tensor_operands.unpack(o_tensor);
        auto                  gpu_sampler = o_tensor.sampler();

        GpuTensor3dMapper mapper(tensor, gpu_sampler);

        auto load_writer = ClLoadStoreHelperWriterFactory::create(this, mapper, GpuLoadStoreType::Load);

        // Initialize the constant part
        load_writer->initialize(dst, x, z, b);

        for (int i = 0; i < dst->format().h; ++i)
        {
            load_writer->write(std::make_pair(i, y_ind->scalar(0, i).str));
        }

        load_writer->finalize();
    }

    void op_store_immediate(const TensorOperand &tensor_name,
                            const Operand       &src_name,
                            const Operand       &x_name,
                            const Operand       &y_name,
                            const Operand       &z_name,
                            const Operand       &batch_index_name) override
    {
        OperandUnpacker operands(_data->tiles, _data->arguments);

        // Not const as it requires changes to 'load_writer'.
        IVectorTile *src = operands.unpack(src_name);
        IVectorTile *x   = operands.unpack(x_name);
        IVectorTile *y   = operands.unpack(y_name);
        IVectorTile *z   = operands.unpack(z_name);
        IVectorTile *b   = operands.unpack(batch_index_name);

        TensorOperandUnpacker tensor_operands(_data->arguments);
        IGpuTensorArgument   *tensor      = tensor_operands.unpack(tensor_name);
        auto                  gpu_sampler = tensor_name.sampler();

        GpuTensor3dMapper mapper(tensor, gpu_sampler);

        auto store_writer = ClLoadStoreHelperWriterFactory::create(this, mapper, GpuLoadStoreType::Store);

        // Initialize the constant part
        store_writer->initialize(src, x, z, b);

        int32_t tile_h = src->format().h;

        for (int m0 = tile_h - 1; m0 >= 0; m0--)
        {
            store_writer->write(std::make_pair(m0, y->scalar(0, 0).str + " + " + std::to_string(m0)));
        }

        store_writer->finalize();
    }

    void op_return() override
    {
        _data->code += "return;\n";
    }

    void util_get_indirect_buffer(const Operand       &o_dst,
                                  const TensorOperand &o_tensor,
                                  const Operand       &o_x,
                                  const Operand       &o_y,
                                  const Operand       &o_x_off,
                                  const Operand       &o_y_off) override
    {
        OperandUnpacker    operands(_data->tiles, _data->arguments);
        const IVectorTile *dst   = operands.unpack(o_dst);
        const IVectorTile *x     = operands.unpack(o_x);
        const IVectorTile *y     = operands.unpack(o_y);
        const IVectorTile *x_off = operands.unpack(o_x_off);
        const IVectorTile *y_off = operands.unpack(o_y_off);

        TensorOperandUnpacker tensor_operands(_data->arguments);
        IGpuTensorArgument   *tensor = tensor_operands.unpack(o_tensor);

        assert(dst->format().w == 1);
        assert(x->format().w == 1);
        assert(y->format().w == 1);
        assert(x_off->format().w == 1);
        assert(y_off->format().w == 1);
        assert(dst->format().dt == DataType::Int32);
        assert(x->format().dt == DataType::Int32);
        assert(y->format().dt == DataType::Int32);
        assert(x_off->format().dt == DataType::Int32);
        assert(y_off->format().dt == DataType::Int32);

        const std::string width  = tensor->component(TensorComponentType::Dim1);
        const std::string height = tensor->component(TensorComponentType::Dim2);
        const std::string wxh    = tensor->component(TensorComponentType::Dim1xDim2);
        /*
        int x_s;
        int y_s;
        x_s = (xi_0 + x_k);
        y_s = (yi_0 + y_k);
        mi_0 = x_s + y_s * width + b * widthxheight;
        mi_0 = select(-1, mi_0, x_s >= 0);
        mi_0 = select(-1, mi_0, y_s >= 0);
        mi_0 = select(-1, mi_0, x_s < 128);
        mi_0 = select(-1, mi_0, y_s < 128);
        */
        compound_statement_begin();
        declare_tile("_x_s", TileInfo(DataType::Int32));
        declare_tile("_y_s", TileInfo(DataType::Int32));
        auto x_s = operands.unpack(Operand("_x_s"));
        auto y_s = operands.unpack(Operand("_y_s"));
        for (int i = 0; i < dst->format().h; ++i)
        {
            // x_s = (xi_0 + x_k);
            // y_s = (yi_0 + y_k);
            _data->code += x_s->scalar(0, i).str;
            _data->code += " = (";
            _data->code += x->scalar(0, i).str;
            _data->code += " + ";
            _data->code += x_off->scalar(0, i).str;
            _data->code += ");\n";
            _data->code += y_s->scalar(0, i).str;
            _data->code += " = (";
            _data->code += y->scalar(0, i).str;
            _data->code += " + ";
            _data->code += y_off->scalar(0, i).str;
            _data->code += ");\n";
            // mi_0 = x_s + y_s * width;
            _data->code += dst->scalar(0, i).str;
            _data->code += " = ";
            _data->code += x_s->scalar(0, i).str;
            _data->code += " + ";
            _data->code += y_s->scalar(0, i).str;
            _data->code += " * " + width + ";\n";
            // mi_0 = select(wxh, mi_0, x_s >= 0);
            _data->code += dst->scalar(0, i).str;
            _data->code += " = select(-1, ";
            _data->code += dst->scalar(0, i).str;
            _data->code += ", ";
            _data->code += x_s->scalar(0, i).str;
            _data->code += " >= 0);\n";
            // mi_0 = select(wxh, mi_0, x_s < width);
            _data->code += dst->scalar(0, i).str;
            _data->code += " = select(-1, ";
            _data->code += dst->scalar(0, i).str;
            _data->code += ", ";
            _data->code += x_s->scalar(0, i).str;
            _data->code += " < ";
            _data->code += width + ");\n";
            // mi_0 = select(wxh, mi_0, y_s >= 0);
            _data->code += dst->scalar(0, i).str;
            _data->code += " = select(-1, ";
            _data->code += dst->scalar(0, i).str;
            _data->code += ", ";
            _data->code += y_s->scalar(0, i).str;
            _data->code += " >= 0);\n";
            // mi_0 = select(wxh, mi_0, y_s < height);
            _data->code += dst->scalar(0, i).str;
            _data->code += " = select(-1, ";
            _data->code += dst->scalar(0, i).str;
            _data->code += ", ";
            _data->code += y_s->scalar(0, i).str;
            _data->code += " < ";
            _data->code += height + ");\n";
        }
        compound_statement_end();
    }

private:
    GpuKernelWriterDataHolder *_data{nullptr};
    GpuKernelWriterAttribute  *_attr{nullptr};
};

/** IGpuKernelWriter factory class */
class GpuKernelWriterFactory final
{
public:
    /** Static method to call the IGpuKernelWriter class accordingly with the Gpu programming language
     *
     * @param[in] gpu GPU target
     *
     * @return IGpuKernelWriter
     */
    static std::unique_ptr<IGpuKernelWriter> create(GpuKernelWriterAttribute *attr, GpuKernelWriterDataHolder *x)
    {
        switch (x->programming_language())
        {
            case GpuTargetLanguage::OpenCL:
                return std::make_unique<ClKernelWriter>(attr, x);
            default:
                std::cout << "Unsupported Gpu programming language" << std::endl;
                assert(false);
                return nullptr;
        }
    }
};

inline int32_t
adjust_step(TensorSamplerFormat tensor_format, int32_t step, const TensorInfo *tensor_info_id, int32_t idx)
{
    auto tensor = tensor_info_id->shape;

    int32_t dim[3] = {0};

    switch (tensor_format)
    {
        case TensorSamplerFormat::C_W_H:
            dim[0] = tensor[0];
            dim[1] = tensor[1];
            dim[2] = tensor[2];
            break;
        case TensorSamplerFormat::C_WH_1:
            dim[0] = tensor[0];
            dim[1] = tensor[1] * tensor[2];
            dim[2] = 1;
            break;
        default:
            std::cout << "Unsupported tensor format" << std::endl;
            assert(false);
            break;
    }

    return std::min(step, dim[idx]);
}

} // namespace prototype
} // namespace ckw

#endif // CKW_PROTOTYPE_SRC_PROTOTYPE_H
