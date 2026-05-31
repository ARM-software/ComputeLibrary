//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include <tuple>
#include <unordered_map>

namespace kai::test {

namespace impl {
/// Naive combination of two hash values
constexpr size_t hash_combine(size_t lhs, size_t rhs) noexcept {
    return lhs ^ (rhs << 3);
}

/// Tuple hash recursive case, combines hash of Ith hash with Ith - 1 hash.
/// Defaults to last element in tuple
template <typename Tuple, std::size_t Index = std::tuple_size<Tuple>::value - 1>
struct TupleHash {
    static size_t combine(const Tuple& tuple) noexcept {
        using EType = std::tuple_element_t<Index, Tuple>;
        return hash_combine(
            std::hash<EType>{}(std::get<Index>(tuple)),  // hash of current element
            TupleHash<Tuple, Index - 1>::combine(tuple)  // hash of previous elements
        );
    }
};

/// TupleHash base case, returns hash of first value
template <typename Tuple>
struct TupleHash<Tuple, 0> {
    static size_t combine(const Tuple& tuple) noexcept {
        using EType = std::tuple_element_t<0, Tuple>;
        return std::hash<EType>{}(std::get<0>(tuple));
    }
};
}  // namespace impl

struct TupleHash {
    template <typename... Args>
    std::size_t operator()(const std::tuple<Args...>& tuple) const noexcept {
        return impl::TupleHash<std::tuple<Args...>>::combine(tuple);
    }
};

/// Cached reference cache
///
/// The user need to specialize and implement the private `generate_reference()` function, which
/// will then be used by the cache mechanism to generate cached reference data
///
///   template <>
///   CacheData ReferenceGenerator<YourTestID, YourTestData>::generate_reference(const YourTestID& k) {
///      ...
///   }
///
/// And then use the `getV()` helper function to retrieve test reference data
///
///   const auto& test_data = getV<YourTestID, YourTestData>(test_id);
///
/// Notice that current implementation can be quite memory intensive, but the positive tradeoff
/// is its ease of use
template <typename K, typename V>
struct ReferenceGenerator final {
    static ReferenceGenerator& getRG() {
        static ReferenceGenerator rg;
        return rg;
    }

    const V& get_test_reference(const K& test_id) {
        if (const auto itr = m_data.find(test_id); itr != end(m_data)) {
            return itr->second;
        }

        return m_data[test_id] = generate_reference(test_id);
    }

private:
    ReferenceGenerator() = default;

    std::unordered_map<K, V, TupleHash> m_data;

    V generate_reference(const K& test_id);
};

/// Main accessor function for retrieving reference test values from test identifier.
///
/// This can also be used as a shim between test data creation and usage
template <typename K, typename V, typename RG = ReferenceGenerator<K, V>>
inline const V& getV(const K& k) {
    auto& rg = ReferenceGenerator<K, V>::getRG();
    return rg.get_test_reference(k);
}

};  // namespace kai::test
