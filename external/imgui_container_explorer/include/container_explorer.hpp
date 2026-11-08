#pragma once

#include <algorithm>
#include <array>
#include <cfloat>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <memory>
#include <ranges>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "imgui.h"

#if defined(__cpp_impl_reflection) && __cpp_impl_reflection >= 202500L && __has_include(<meta>)
#include <meta>
#define CONTAINER_EXPLORER_HAS_CPP26_REFLECTION 1
#else
#define CONTAINER_EXPLORER_HAS_CPP26_REFLECTION 0
#endif

#if !CONTAINER_EXPLORER_HAS_CPP26_REFLECTION && defined(__clang__)
#if __has_builtin(__builtin_dump_struct)
#define CONTAINER_EXPLORER_HAS_CLANG_DUMP_STRUCT 1
#endif
#endif
#ifndef CONTAINER_EXPLORER_HAS_CLANG_DUMP_STRUCT
#define CONTAINER_EXPLORER_HAS_CLANG_DUMP_STRUCT 0
#endif

#if !CONTAINER_EXPLORER_HAS_CPP26_REFLECTION && defined(_WIN32) && defined(_MSC_VER) && !defined(__clang__)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <dbghelp.h>
#pragma comment(lib, "dbghelp.lib")
#undef min
#undef max
#undef near
#undef far
#define CONTAINER_EXPLORER_HAS_MSVC_PDB_REFLECTION 1
#else
#define CONTAINER_EXPLORER_HAS_MSVC_PDB_REFLECTION 0
#endif

#if __has_include(<glm/glm.hpp>)
#include <glm/glm.hpp>
#define CONTAINER_EXPLORER_HAS_GLM 1
#else
#define CONTAINER_EXPLORER_HAS_GLM 0
#endif

// Optional trait to associate an SOA reference struct with a value struct for field name reflection.
// Users may either specialize `container_explorer_soa_value_type<RefStruct>` or define
// `using value_type = ValueStruct;` (or `using soa_value_type = ValueStruct;`) inside `RefStruct`.
template <typename T>
struct container_explorer_soa_value_type
{
    using type = void;
};

namespace container_explorer_detail
{
    // -------------------------------------------------------------------------
    // GLM Type Traits
    // -------------------------------------------------------------------------
    template <typename T>
    struct glm_vector_info
    {
        static constexpr bool is_vector = false;
    };

    template <typename T>
    struct glm_matrix_info
    {
        static constexpr bool is_matrix = false;
    };

#if CONTAINER_EXPLORER_HAS_GLM
    template <glm::length_t Length, typename T, glm::qualifier Qualifier>
    struct glm_vector_info<glm::vec<Length, T, Qualifier>>
    {
        static constexpr bool is_vector = true;
        static constexpr glm::length_t length = Length;
        using value_type = T;
    };

    template <glm::length_t Cols, glm::length_t Rows, typename T, glm::qualifier Qualifier>
    struct glm_matrix_info<glm::mat<Cols, Rows, T, Qualifier>>
    {
        static constexpr bool is_matrix = true;
        static constexpr glm::length_t cols = Cols;
        static constexpr glm::length_t rows = Rows;
        using col_type = glm::vec<Rows, T, Qualifier>;
        using value_type = T;
    };
#endif

    // -------------------------------------------------------------------------
    // Wrapper & Tuple Traits
    // -------------------------------------------------------------------------

    // Detects proxy wrappers like soa::RefWrapper<TReference> that inherit from
    // their first template type argument (e.g. struct RefWrapper : TReference).
    template <typename T>
    struct inherited_wrapper_traits
    {
        static constexpr bool is_wrapper = false;
        using underlying_type = T;
    };

    template <template <typename...> class Wrapper, typename First, typename... Rest>
        requires (std::is_class_v<Wrapper<First, Rest...>> &&
                  std::is_class_v<First> &&
                  std::is_base_of_v<First, Wrapper<First, Rest...>> &&
                  !std::is_same_v<First, Wrapper<First, Rest...>>)
    struct inherited_wrapper_traits<Wrapper<First, Rest...>>
    {
        static constexpr bool is_wrapper = true;
        using underlying_type = First;
    };

    template <typename T>
    struct is_reference_wrapper : std::false_type {};

    template <typename U>
    struct is_reference_wrapper<std::reference_wrapper<U>> : std::true_type {};

    template <typename T>
    inline constexpr bool is_reference_wrapper_v = is_reference_wrapper<std::remove_cvref_t<T>>::value;

    template <typename T>
    struct is_std_array : std::false_type {};

    template <typename U, std::size_t N>
    struct is_std_array<std::array<U, N>> : std::true_type {};

    template <typename T>
    inline constexpr bool is_std_array_v = is_std_array<std::remove_cvref_t<T>>::value;

    template <typename T>
    struct is_std_pair : std::false_type {};

    template <typename A, typename B>
    struct is_std_pair<std::pair<A, B>> : std::true_type {};

    template <typename T>
    inline constexpr bool is_std_pair_v = is_std_pair<std::remove_cvref_t<T>>::value;

    template <typename T>
    struct is_std_tuple : std::false_type {};

    template <typename... Ts>
    struct is_std_tuple<std::tuple<Ts...>> : std::true_type {};

    template <typename T>
    inline constexpr bool is_std_tuple_v = is_std_tuple<std::remove_cvref_t<T>>::value;

    // Detects proxy boolean references such as std::vector<bool>::reference
    template <typename T>
    concept BoolProxy =
        !std::is_fundamental_v<std::remove_cvref_t<T>> &&
        !std::is_enum_v<std::remove_cvref_t<T>> &&
        !std::is_pointer_v<std::remove_cvref_t<T>> &&
        !inherited_wrapper_traits<std::remove_cvref_t<T>>::is_wrapper &&
        std::convertible_to<T, bool> &&
        requires(T proxy, bool b) {
            proxy = b;
        };

    // -------------------------------------------------------------------------
    // Leaf Widget Detection
    // -------------------------------------------------------------------------
    template <typename T>
    constexpr bool is_leaf_type()
    {
        using Clean = std::remove_cvref_t<T>;
        if constexpr (std::is_fundamental_v<Clean> ||
                      std::is_enum_v<Clean> ||
                      std::is_pointer_v<Clean> ||
                      std::is_same_v<Clean, std::string> ||
                      std::is_same_v<Clean, std::string_view> ||
                      glm_vector_info<Clean>::is_vector ||
                      BoolProxy<T>)
        {
            return true;
        }
        else
        {
            return false;
        }
    }

    template <typename T>
    inline constexpr bool is_leaf_type_v = is_leaf_type<T>();

    // -------------------------------------------------------------------------
    // Top-Level Aggregate Arity Detection (handles reference members without brace elision)
    // -------------------------------------------------------------------------
    struct AnyTopMember
    {
        template <typename T>
        constexpr operator T&() const noexcept;
    };

    template <typename T, std::size_t... Is>
    constexpr auto test_top_init(std::index_sequence<Is...>)
        -> decltype(T{((void)Is, AnyTopMember{})...}, true)
    {
        return true;
    }

    template <typename T, std::size_t... Is>
    constexpr bool test_top_init(...)
    {
        return false;
    }

    template <typename T, std::size_t N = 32>
    constexpr std::size_t find_top_aggregate_arity()
    {
        if constexpr (test_top_init<T>(std::make_index_sequence<N>{}))
        {
            return N;
        }
        else if constexpr (N > 0)
        {
            return find_top_aggregate_arity<T, N - 1>();
        }
        else
        {
            return 0;
        }
    }

    template <typename T>
    constexpr std::size_t aggregate_field_count()
    {
        using Clean = std::remove_cvref_t<T>;
#if CONTAINER_EXPLORER_HAS_CPP26_REFLECTION
        constexpr auto ctx = std::meta::access_context::current();
        return std::meta::nonstatic_data_members_of(^^Clean, ctx).size();
#else
        return find_top_aggregate_arity<Clean>();
#endif
    }

    // -------------------------------------------------------------------------
    // Associated SOA Value Struct Resolution (for reflecting reference member names)
    // -------------------------------------------------------------------------
    template <typename T>
    struct template_value_counterpart
    {
        using type = void;
    };

    template <template <typename...> class Tmpl, typename... Args>
        requires ((std::is_reference_v<Args> || ...) &&
                  requires { sizeof(Tmpl<std::remove_cvref_t<Args>...>); })
    struct template_value_counterpart<Tmpl<Args...>>
    {
        using type = Tmpl<std::remove_cvref_t<Args>...>;
    };

    template <template <template <typename> class> class Tmpl, template <typename> class Meta>
        requires (!std::is_same_v<Tmpl<Meta>, Tmpl<std::type_identity_t>> &&
                  requires { sizeof(Tmpl<std::type_identity_t>); })
    struct template_value_counterpart<Tmpl<Meta>>
    {
        using type = Tmpl<std::type_identity_t>;
    };

    template <typename T, typename Candidate>
    consteval bool is_valid_schema_candidate()
    {
        using Clean = std::remove_cvref_t<T>;
        using Cand = std::remove_cvref_t<Candidate>;
        if constexpr (!std::is_void_v<Cand> &&
                      !std::is_same_v<Clean, Cand> &&
                      !is_leaf_type_v<Cand> &&
                      std::is_aggregate_v<Cand>)
        {
            return aggregate_field_count<Cand>() == aggregate_field_count<Clean>();
        }
        else
        {
            return false;
        }
    }

    template <typename T>
    consteval auto select_schema_type()
    {
        using Clean = std::remove_cvref_t<T>;
        if constexpr (!std::is_void_v<typename container_explorer_soa_value_type<Clean>::type> &&
                      is_valid_schema_candidate<Clean, typename container_explorer_soa_value_type<Clean>::type>())
        {
            return std::type_identity<std::remove_cvref_t<typename container_explorer_soa_value_type<Clean>::type>>{};
        }
        else if constexpr (requires { typename Clean::soa_value_type; })
        {
            if constexpr (is_valid_schema_candidate<Clean, typename Clean::soa_value_type>())
                return std::type_identity<std::remove_cvref_t<typename Clean::soa_value_type>>{};
            else
                return std::type_identity<Clean>{};
        }
        else if constexpr (requires { typename Clean::value_type; })
        {
            if constexpr (is_valid_schema_candidate<Clean, typename Clean::value_type>())
                return std::type_identity<std::remove_cvref_t<typename Clean::value_type>>{};
            else
                return std::type_identity<Clean>{};
        }
        else if constexpr (!std::is_void_v<typename template_value_counterpart<Clean>::type> &&
                           is_valid_schema_candidate<Clean, typename template_value_counterpart<Clean>::type>())
        {
            return std::type_identity<typename template_value_counterpart<Clean>::type>{};
        }
        else
        {
            return std::type_identity<Clean>{};
        }
    }

    template <typename T>
    using resolve_schema_type_t = typename decltype(select_schema_type<T>())::type;

    // -------------------------------------------------------------------------
    // Self-Contained Aggregate Reflection (Zero Boost.PFR Dependency)
    // -------------------------------------------------------------------------
    template <typename T>
    extern const T fake_object_v;

    template <typename T>
    consteval const T& fake_object() noexcept
    {
        return fake_object_v<T>;
    }

#define CONTAINER_EXPLORER_COMMA() ,
#define CONTAINER_EXPLORER_EMPTY()
#define CONTAINER_EXPLORER_BIND(i) a##i
#define CONTAINER_EXPLORER_RET(i) if constexpr (I == (i)) return std::addressof(a##i);
#define CONTAINER_EXPLORER_TYPE_RET(i) if constexpr (I == (i)) return std::type_identity<std::remove_cvref_t<decltype(a##i)>>{};
#define CONTAINER_EXPLORER_IS_VAL_RET(i) if constexpr (I == (i)) return std::is_const_v<decltype(a##i)> && requires { std::addressof(a##i); };

#define CONTAINER_EXPLORER_REP_1(M, S)  M(0)
#define CONTAINER_EXPLORER_REP_2(M, S)  CONTAINER_EXPLORER_REP_1(M, S) S() M(1)
#define CONTAINER_EXPLORER_REP_3(M, S)  CONTAINER_EXPLORER_REP_2(M, S) S() M(2)
#define CONTAINER_EXPLORER_REP_4(M, S)  CONTAINER_EXPLORER_REP_3(M, S) S() M(3)
#define CONTAINER_EXPLORER_REP_5(M, S)  CONTAINER_EXPLORER_REP_4(M, S) S() M(4)
#define CONTAINER_EXPLORER_REP_6(M, S)  CONTAINER_EXPLORER_REP_5(M, S) S() M(5)
#define CONTAINER_EXPLORER_REP_7(M, S)  CONTAINER_EXPLORER_REP_6(M, S) S() M(6)
#define CONTAINER_EXPLORER_REP_8(M, S)  CONTAINER_EXPLORER_REP_7(M, S) S() M(7)
#define CONTAINER_EXPLORER_REP_9(M, S)  CONTAINER_EXPLORER_REP_8(M, S) S() M(8)
#define CONTAINER_EXPLORER_REP_10(M, S) CONTAINER_EXPLORER_REP_9(M, S) S() M(9)
#define CONTAINER_EXPLORER_REP_11(M, S) CONTAINER_EXPLORER_REP_10(M, S) S() M(10)
#define CONTAINER_EXPLORER_REP_12(M, S) CONTAINER_EXPLORER_REP_11(M, S) S() M(11)
#define CONTAINER_EXPLORER_REP_13(M, S) CONTAINER_EXPLORER_REP_12(M, S) S() M(12)
#define CONTAINER_EXPLORER_REP_14(M, S) CONTAINER_EXPLORER_REP_13(M, S) S() M(13)
#define CONTAINER_EXPLORER_REP_15(M, S) CONTAINER_EXPLORER_REP_14(M, S) S() M(14)
#define CONTAINER_EXPLORER_REP_16(M, S) CONTAINER_EXPLORER_REP_15(M, S) S() M(15)
#define CONTAINER_EXPLORER_REP_17(M, S) CONTAINER_EXPLORER_REP_16(M, S) S() M(16)
#define CONTAINER_EXPLORER_REP_18(M, S) CONTAINER_EXPLORER_REP_17(M, S) S() M(17)
#define CONTAINER_EXPLORER_REP_19(M, S) CONTAINER_EXPLORER_REP_18(M, S) S() M(18)
#define CONTAINER_EXPLORER_REP_20(M, S) CONTAINER_EXPLORER_REP_19(M, S) S() M(19)
#define CONTAINER_EXPLORER_REP_21(M, S) CONTAINER_EXPLORER_REP_20(M, S) S() M(20)
#define CONTAINER_EXPLORER_REP_22(M, S) CONTAINER_EXPLORER_REP_21(M, S) S() M(21)
#define CONTAINER_EXPLORER_REP_23(M, S) CONTAINER_EXPLORER_REP_22(M, S) S() M(22)
#define CONTAINER_EXPLORER_REP_24(M, S) CONTAINER_EXPLORER_REP_23(M, S) S() M(23)
#define CONTAINER_EXPLORER_REP_25(M, S) CONTAINER_EXPLORER_REP_24(M, S) S() M(24)
#define CONTAINER_EXPLORER_REP_26(M, S) CONTAINER_EXPLORER_REP_25(M, S) S() M(25)
#define CONTAINER_EXPLORER_REP_27(M, S) CONTAINER_EXPLORER_REP_26(M, S) S() M(26)
#define CONTAINER_EXPLORER_REP_28(M, S) CONTAINER_EXPLORER_REP_27(M, S) S() M(27)
#define CONTAINER_EXPLORER_REP_29(M, S) CONTAINER_EXPLORER_REP_28(M, S) S() M(28)
#define CONTAINER_EXPLORER_REP_30(M, S) CONTAINER_EXPLORER_REP_29(M, S) S() M(29)
#define CONTAINER_EXPLORER_REP_31(M, S) CONTAINER_EXPLORER_REP_30(M, S) S() M(30)
#define CONTAINER_EXPLORER_REP_32(M, S) CONTAINER_EXPLORER_REP_31(M, S) S() M(31)

#define CONTAINER_EXPLORER_IS_VAL_CASE(K) \
    else if constexpr (N == (K)) { \
        const auto& [CONTAINER_EXPLORER_REP_##K(CONTAINER_EXPLORER_BIND, CONTAINER_EXPLORER_COMMA)] = fake_object<Clean>(); \
        CONTAINER_EXPLORER_REP_##K(CONTAINER_EXPLORER_IS_VAL_RET, CONTAINER_EXPLORER_EMPTY) \
    }

    template <typename T, std::size_t I, std::size_t N>
    consteval bool is_value_member_at()
    {
        using Clean = std::remove_cvref_t<T>;
        if constexpr (N == 0) {
            return false;
        }
        CONTAINER_EXPLORER_IS_VAL_CASE(1)  CONTAINER_EXPLORER_IS_VAL_CASE(2)  CONTAINER_EXPLORER_IS_VAL_CASE(3)  CONTAINER_EXPLORER_IS_VAL_CASE(4)
        CONTAINER_EXPLORER_IS_VAL_CASE(5)  CONTAINER_EXPLORER_IS_VAL_CASE(6)  CONTAINER_EXPLORER_IS_VAL_CASE(7)  CONTAINER_EXPLORER_IS_VAL_CASE(8)
        CONTAINER_EXPLORER_IS_VAL_CASE(9)  CONTAINER_EXPLORER_IS_VAL_CASE(10) CONTAINER_EXPLORER_IS_VAL_CASE(11) CONTAINER_EXPLORER_IS_VAL_CASE(12)
        CONTAINER_EXPLORER_IS_VAL_CASE(13) CONTAINER_EXPLORER_IS_VAL_CASE(14) CONTAINER_EXPLORER_IS_VAL_CASE(15) CONTAINER_EXPLORER_IS_VAL_CASE(16)
        CONTAINER_EXPLORER_IS_VAL_CASE(17) CONTAINER_EXPLORER_IS_VAL_CASE(18) CONTAINER_EXPLORER_IS_VAL_CASE(19) CONTAINER_EXPLORER_IS_VAL_CASE(20)
        CONTAINER_EXPLORER_IS_VAL_CASE(21) CONTAINER_EXPLORER_IS_VAL_CASE(22) CONTAINER_EXPLORER_IS_VAL_CASE(23) CONTAINER_EXPLORER_IS_VAL_CASE(24)
        CONTAINER_EXPLORER_IS_VAL_CASE(25) CONTAINER_EXPLORER_IS_VAL_CASE(26) CONTAINER_EXPLORER_IS_VAL_CASE(27) CONTAINER_EXPLORER_IS_VAL_CASE(28)
        CONTAINER_EXPLORER_IS_VAL_CASE(29) CONTAINER_EXPLORER_IS_VAL_CASE(30) CONTAINER_EXPLORER_IS_VAL_CASE(31) CONTAINER_EXPLORER_IS_VAL_CASE(32)
        return false;
    }

#undef CONTAINER_EXPLORER_IS_VAL_CASE

#define CONTAINER_EXPLORER_TYPE_CASE(K) \
    else if constexpr (N == (K)) { \
        const auto& [CONTAINER_EXPLORER_REP_##K(CONTAINER_EXPLORER_BIND, CONTAINER_EXPLORER_COMMA)] = fake_object<Clean>(); \
        CONTAINER_EXPLORER_REP_##K(CONTAINER_EXPLORER_TYPE_RET, CONTAINER_EXPLORER_EMPTY) \
    }

    template <typename T, std::size_t I, std::size_t N>
    consteval auto field_type_at()
    {
        using Clean = std::remove_cvref_t<T>;
        if constexpr (N == 0) {
            return std::type_identity<void>{};
        }
        CONTAINER_EXPLORER_TYPE_CASE(1)  CONTAINER_EXPLORER_TYPE_CASE(2)  CONTAINER_EXPLORER_TYPE_CASE(3)  CONTAINER_EXPLORER_TYPE_CASE(4)
        CONTAINER_EXPLORER_TYPE_CASE(5)  CONTAINER_EXPLORER_TYPE_CASE(6)  CONTAINER_EXPLORER_TYPE_CASE(7)  CONTAINER_EXPLORER_TYPE_CASE(8)
        CONTAINER_EXPLORER_TYPE_CASE(9)  CONTAINER_EXPLORER_TYPE_CASE(10) CONTAINER_EXPLORER_TYPE_CASE(11) CONTAINER_EXPLORER_TYPE_CASE(12)
        CONTAINER_EXPLORER_TYPE_CASE(13) CONTAINER_EXPLORER_TYPE_CASE(14) CONTAINER_EXPLORER_TYPE_CASE(15) CONTAINER_EXPLORER_TYPE_CASE(16)
        CONTAINER_EXPLORER_TYPE_CASE(17) CONTAINER_EXPLORER_TYPE_CASE(18) CONTAINER_EXPLORER_TYPE_CASE(19) CONTAINER_EXPLORER_TYPE_CASE(20)
        CONTAINER_EXPLORER_TYPE_CASE(21) CONTAINER_EXPLORER_TYPE_CASE(22) CONTAINER_EXPLORER_TYPE_CASE(23) CONTAINER_EXPLORER_TYPE_CASE(24)
        CONTAINER_EXPLORER_TYPE_CASE(25) CONTAINER_EXPLORER_TYPE_CASE(26) CONTAINER_EXPLORER_TYPE_CASE(27) CONTAINER_EXPLORER_TYPE_CASE(28)
        CONTAINER_EXPLORER_TYPE_CASE(29) CONTAINER_EXPLORER_TYPE_CASE(30) CONTAINER_EXPLORER_TYPE_CASE(31) CONTAINER_EXPLORER_TYPE_CASE(32)
        else {
            return std::type_identity<void>{};
        }
    }

#undef CONTAINER_EXPLORER_TYPE_CASE

    template <typename T, std::size_t I, std::size_t N>
    using field_type_at_t = typename decltype(field_type_at<T, I, N>())::type;

#define CONTAINER_EXPLORER_ADDR_CASE(K) \
    else if constexpr (N == (K)) { \
        const auto& [CONTAINER_EXPLORER_REP_##K(CONTAINER_EXPLORER_BIND, CONTAINER_EXPLORER_COMMA)] = fake_object<T>(); \
        CONTAINER_EXPLORER_REP_##K(CONTAINER_EXPLORER_RET, CONTAINER_EXPLORER_EMPTY) \
    }

    template <typename T, std::size_t I, std::size_t N>
    consteval auto address_of_value_field() noexcept
    {
        if constexpr (N == 0) {
            return nullptr;
        }
        CONTAINER_EXPLORER_ADDR_CASE(1)  CONTAINER_EXPLORER_ADDR_CASE(2)  CONTAINER_EXPLORER_ADDR_CASE(3)  CONTAINER_EXPLORER_ADDR_CASE(4)
        CONTAINER_EXPLORER_ADDR_CASE(5)  CONTAINER_EXPLORER_ADDR_CASE(6)  CONTAINER_EXPLORER_ADDR_CASE(7)  CONTAINER_EXPLORER_ADDR_CASE(8)
        CONTAINER_EXPLORER_ADDR_CASE(9)  CONTAINER_EXPLORER_ADDR_CASE(10) CONTAINER_EXPLORER_ADDR_CASE(11) CONTAINER_EXPLORER_ADDR_CASE(12)
        CONTAINER_EXPLORER_ADDR_CASE(13) CONTAINER_EXPLORER_ADDR_CASE(14) CONTAINER_EXPLORER_ADDR_CASE(15) CONTAINER_EXPLORER_ADDR_CASE(16)
        CONTAINER_EXPLORER_ADDR_CASE(17) CONTAINER_EXPLORER_ADDR_CASE(18) CONTAINER_EXPLORER_ADDR_CASE(19) CONTAINER_EXPLORER_ADDR_CASE(20)
        CONTAINER_EXPLORER_ADDR_CASE(21) CONTAINER_EXPLORER_ADDR_CASE(22) CONTAINER_EXPLORER_ADDR_CASE(23) CONTAINER_EXPLORER_ADDR_CASE(24)
        CONTAINER_EXPLORER_ADDR_CASE(25) CONTAINER_EXPLORER_ADDR_CASE(26) CONTAINER_EXPLORER_ADDR_CASE(27) CONTAINER_EXPLORER_ADDR_CASE(28)
        CONTAINER_EXPLORER_ADDR_CASE(29) CONTAINER_EXPLORER_ADDR_CASE(30) CONTAINER_EXPLORER_ADDR_CASE(31) CONTAINER_EXPLORER_ADDR_CASE(32)
        else {
            return nullptr;
        }
    }

#undef CONTAINER_EXPLORER_ADDR_CASE

    template <typename T>
    struct field_ptr_wrapper
    {
        T ptr;
    };
    template <typename T>
    field_ptr_wrapper(T) -> field_ptr_wrapper<T>;

    constexpr bool is_ident_char(char c) noexcept
    {
        return (c >= 'a' && c <= 'z') ||
               (c >= 'A' && c <= 'Z') ||
               (c >= '0' && c <= '9') ||
               c == '_';
    }

    template <typename MsvcWorkaround, auto WrappedPtr>
    consteval auto name_of_wrapped_field() noexcept
    {
#if defined(__FUNCSIG__)
        constexpr std::string_view sv = __FUNCSIG__;
#elif defined(__PRETTY_FUNCTION__) || defined(__GNUC__) || defined(__clang__)
        constexpr std::string_view sv = __PRETTY_FUNCTION__;
#else
        constexpr std::string_view sv = "";
#endif
        constexpr std::string_view fn = [](std::string_view sig) consteval noexcept -> std::string_view {
            const std::size_t close_brace = sig.rfind('}');
            if (close_brace == std::string_view::npos)
                return {};
            std::size_t end = close_brace;
            while (end > 0 && (sig[end - 1] == ')' || sig[end - 1] == ' '))
                --end;
            std::size_t start = end;
            while (start > 0 && is_ident_char(sig[start - 1]))
                --start;
            if (start == end || start == 0)
                return {};
            const char sep = sig[start - 1];
            if (sep != '>' && sep != '.' && sep != ':')
                return {};
            return sig.substr(start, end - start);
        }(sv);

        std::array<char, fn.size() + 1> res{};
        if constexpr (!fn.empty())
        {
            for (std::size_t i = 0; i < fn.size(); ++i)
                res[i] = fn[i];
        }
        return res;
    }

    template <typename T, std::size_t I, std::size_t N>
    inline constexpr auto stored_field_name = name_of_wrapped_field<
        T,
        field_ptr_wrapper{address_of_value_field<T, I, N>()}
    >();

#if CONTAINER_EXPLORER_HAS_CLANG_DUMP_STRUCT
    struct clang_dummy_ref_provider
    {
        void* ptr;
        template <typename U>
        operator U&() const noexcept
        {
            return *static_cast<U*>(ptr);
        }
    };

    template <typename T, std::size_t... Is>
    T make_clang_dummy_aggregate(void* buf, std::index_sequence<Is...>)
    {
        return T{((void)Is, clang_dummy_ref_provider{buf})...};
    }

    template <typename T, std::size_t N>
    std::vector<std::string> clang_dump_struct_member_names()
    {
        alignas(64) std::byte buf[1024]{};
        T dummy = make_clang_dummy_aggregate<T>(buf, std::make_index_sequence<N>{});

        std::vector<std::string> names;
        auto cb = [&](const char*, const auto&... args)
        {
            if constexpr (sizeof...(args) >= 3)
            {
                auto tup = std::forward_as_tuple(args...);
                if constexpr (std::is_convertible_v<decltype(std::get<0>(tup)), std::string_view> &&
                              std::is_convertible_v<decltype(std::get<2>(tup)), std::string_view>)
                {
                    std::string_view indent(std::get<0>(tup));
                    std::string_view fname(std::get<2>(tup));
                    if (indent == "  " && !fname.empty())
                    {
                        names.emplace_back(fname);
                    }
                }
            }
        };
        __builtin_dump_struct(&dummy, cb);
        return names;
    }
#endif

    template <typename T>
    constexpr std::string_view extract_clean_udt_name() noexcept
    {
#if defined(__FUNCSIG__)
        std::string_view sig = __FUNCSIG__;
        const std::size_t open = sig.find("extract_clean_udt_name<");
        if (open == std::string_view::npos)
            return {};
        const std::size_t start = open + 23;
        const std::size_t close = sig.rfind('>');
        if (close == std::string_view::npos || close <= start)
            return {};
        std::string_view name = sig.substr(start, close - start);
#elif defined(__PRETTY_FUNCTION__)
        std::string_view sig = __PRETTY_FUNCTION__;
        const std::size_t eq = sig.find("T = ");
        if (eq == std::string_view::npos)
            return {};
        const std::size_t start = eq + 4;
        const std::size_t close = sig.find_first_of(";]", start);
        if (close == std::string_view::npos || close <= start)
            return {};
        std::string_view name = sig.substr(start, close - start);
#else
        std::string_view name = {};
#endif
        while (!name.empty() && name.front() == ' ')
            name.remove_prefix(1);
        while (!name.empty() && name.back() == ' ')
            name.remove_suffix(1);
        if (name.starts_with("struct "))
            name.remove_prefix(7);
        else if (name.starts_with("class "))
            name.remove_prefix(6);
        return name;
    }

    inline std::string strip_cpp_comments(std::string_view src)
    {
        std::string out;
        out.reserve(src.size());
        std::size_t i = 0;
        while (i < src.size())
        {
            if (i + 1 < src.size() && src[i] == '/' && src[i + 1] == '/')
            {
                i += 2;
                while (i < src.size() && src[i] != '\n')
                    ++i;
            }
            else if (i + 1 < src.size() && src[i] == '/' && src[i + 1] == '*')
            {
                i += 2;
                while (i + 1 < src.size() && !(src[i] == '*' && src[i + 1] == '/'))
                    ++i;
                if (i + 1 < src.size())
                    i += 2;
            }
            else
            {
                out.push_back(src[i]);
                ++i;
            }
        }
        return out;
    }

    inline std::string_view trim_sv(std::string_view s) noexcept
    {
        while (!s.empty() && (s.front() == ' ' || s.front() == '\t' || s.front() == '\r' || s.front() == '\n'))
            s.remove_prefix(1);
        while (!s.empty() && (s.back() == ' ' || s.back() == '\t' || s.back() == '\r' || s.back() == '\n'))
            s.remove_suffix(1);
        return s;
    }

    inline std::vector<std::string> parse_struct_body_members(std::string_view body)
    {
        std::vector<std::string> fields;
        std::size_t i = 0;
        while (i < body.size())
        {
            std::size_t stmt_start = i;
            int paren_depth = 0;
            int angle_depth = 0;
            int brace_depth = 0;
            bool saw_paren_before_eq = false;
            bool saw_eq_or_colon = false;
            bool skipped_block = false;

            while (i < body.size())
            {
                char c = body[i];
                if (c == '(')
                {
                    if (!saw_eq_or_colon && brace_depth == 0)
                        saw_paren_before_eq = true;
                    ++paren_depth;
                }
                else if (c == ')')
                {
                    if (paren_depth > 0) --paren_depth;
                }
                else if (c == '<')
                {
                    if (paren_depth == 0 && brace_depth == 0) ++angle_depth;
                }
                else if (c == '>')
                {
                    if (paren_depth == 0 && brace_depth == 0 && angle_depth > 0) --angle_depth;
                }
                else if ((c == '=' || c == ':') && paren_depth == 0 && brace_depth == 0 && angle_depth == 0)
                {
                    if (c == ':' && i + 1 < body.size() && body[i + 1] == ':')
                    {
                        ++i;
                    }
                    else
                    {
                        saw_eq_or_colon = true;
                    }
                }
                else if (c == '{')
                {
                    ++brace_depth;
                }
                else if (c == '}')
                {
                    if (brace_depth > 0)
                    {
                        --brace_depth;
                        if (brace_depth == 0 && saw_paren_before_eq)
                        {
                            ++i;
                            while (i < body.size() && (body[i] == ' ' || body[i] == '\t' || body[i] == '\r' || body[i] == '\n' || body[i] == ';'))
                                ++i;
                            skipped_block = true;
                            break;
                        }
                    }
                }
                else if (c == ';' && paren_depth == 0 && brace_depth == 0)
                {
                    break;
                }
                ++i;
            }

            if (skipped_block)
                continue;
            if (i >= body.size())
                break;

            std::string_view stmt = trim_sv(body.substr(stmt_start, i - stmt_start));
            ++i; // skip ';'

            if (stmt.empty())
                continue;
            if (stmt.starts_with("public:") || stmt.starts_with("private:") || stmt.starts_with("protected:"))
            {
                auto colon_pos = stmt.find(':');
                stmt = trim_sv(stmt.substr(colon_pos + 1));
                if (stmt.empty())
                    continue;
            }
            if (stmt.starts_with("static ") || stmt.starts_with("using ") || stmt.starts_with("typedef ") ||
                stmt.starts_with("friend ") || stmt.starts_with("constexpr ") || stmt.starts_with("template") ||
                stmt.starts_with("struct ") || stmt.starts_with("class ") || stmt.starts_with("enum "))
            {
                continue;
            }
            if (saw_paren_before_eq)
                continue;

            // Strip trailing initializer (= ... or {...}) or bit-field width (: ...)
            std::size_t cut = std::string_view::npos;
            int p_depth = 0;
            int a_depth = 0;
            for (std::size_t k = 0; k < stmt.size(); ++k)
            {
                char ch = stmt[k];
                if (ch == '(') ++p_depth;
                else if (ch == ')' && p_depth > 0) --p_depth;
                else if (ch == '<') ++a_depth;
                else if (ch == '>' && a_depth > 0) --a_depth;
                else if (p_depth == 0 && a_depth == 0)
                {
                    if (ch == '=' || ch == '{')
                    {
                        cut = k;
                        break;
                    }
                    if (ch == ':')
                    {
                        if (k + 1 < stmt.size() && stmt[k + 1] == ':')
                        {
                            ++k;
                        }
                        else
                        {
                            cut = k;
                            break;
                        }
                    }
                }
            }
            if (cut != std::string_view::npos)
                stmt = trim_sv(stmt.substr(0, cut));

            // Strip array suffix [N] if present
            while (!stmt.empty() && stmt.back() == ']')
            {
                auto open_br = stmt.rfind('[');
                if (open_br == std::string_view::npos)
                    break;
                stmt = trim_sv(stmt.substr(0, open_br));
            }

            // Extract trailing identifier
            std::size_t end_id = stmt.size();
            std::size_t start_id = end_id;
            while (start_id > 0 && is_ident_char(stmt[start_id - 1]))
                --start_id;
            if (start_id < end_id && start_id > 0)
            {
                fields.emplace_back(stmt.substr(start_id, end_id - start_id));
            }
        }
        return fields;
    }

    inline void index_structs_in_source(std::string_view raw_source, std::unordered_map<std::string, std::vector<std::string>>& out_map)
    {
        std::string clean = strip_cpp_comments(raw_source);
        std::string_view sv(clean);
        std::size_t pos = 0;
        while (pos < sv.size())
        {
            std::size_t kw = sv.find("struct", pos);
            if (kw == std::string_view::npos)
                break;
            if ((kw > 0 && is_ident_char(sv[kw - 1])) || (kw + 6 < sv.size() && is_ident_char(sv[kw + 6])))
            {
                pos = kw + 6;
                continue;
            }
            std::size_t cur = kw + 6;
            while (cur < sv.size())
            {
                while (cur < sv.size() && (sv[cur] == ' ' || sv[cur] == '\t' || sv[cur] == '\r' || sv[cur] == '\n'))
                    ++cur;
                if (sv.substr(cur).starts_with("alignas"))
                {
                    cur += 7;
                    while (cur < sv.size() && sv[cur] != '(') ++cur;
                    int pd = 0;
                    while (cur < sv.size())
                    {
                        if (sv[cur] == '(') ++pd;
                        else if (sv[cur] == ')') { --pd; if (pd == 0) { ++cur; break; } }
                        ++cur;
                    }
                }
                else
                {
                    break;
                }
            }
            std::size_t name_start = cur;
            while (cur < sv.size() && is_ident_char(sv[cur]))
                ++cur;
            std::string_view struct_name = sv.substr(name_start, cur - name_start);
            if (struct_name.empty())
            {
                pos = cur + 1;
                continue;
            }
            while (cur < sv.size() && sv[cur] != '{' && sv[cur] != ';')
                ++cur;
            if (cur >= sv.size() || sv[cur] != '{')
            {
                pos = cur + 1;
                continue;
            }
            std::size_t body_start = cur + 1;
            int bd = 1;
            std::size_t body_end = body_start;
            while (body_end < sv.size() && bd > 0)
            {
                if (sv[body_end] == '{') ++bd;
                else if (sv[body_end] == '}') --bd;
                if (bd > 0) ++body_end;
            }
            if (bd == 0)
            {
                auto parsed = parse_struct_body_members(sv.substr(body_start, body_end - body_start));
                if (!parsed.empty())
                {
                    out_map[std::string(struct_name)] = std::move(parsed);
                }
                pos = body_end + 1;
            }
            else
            {
                pos = body_start;
            }
        }
    }

    inline const std::unordered_map<std::string, std::vector<std::string>>& get_project_source_struct_map()
    {
        static const std::unordered_map<std::string, std::vector<std::string>> cache = []()
        {
            std::unordered_map<std::string, std::vector<std::string>> map;
            std::vector<std::filesystem::path> roots;
#ifdef CONTAINER_EXPLORER_PROJECT_SOURCE_DIR
            roots.emplace_back(CONTAINER_EXPLORER_PROJECT_SOURCE_DIR);
#endif
            std::error_code ec;
            auto cwd = std::filesystem::current_path(ec);
            if (!ec && !cwd.empty())
                roots.push_back(cwd);

            for (const auto& root : roots)
            {
                for (const char* sub : {"src", "include", "tests"})
                {
                    auto dir = root / sub;
                    if (!std::filesystem::exists(dir, ec) || !std::filesystem::is_directory(dir, ec))
                        continue;
                    for (const auto& entry : std::filesystem::recursive_directory_iterator(dir, ec))
                    {
                        if (!entry.is_regular_file(ec))
                            continue;
                        auto ext = entry.path().extension().string();
                        if (ext == ".hpp" || ext == ".h" || ext == ".cuh" || ext == ".cpp")
                        {
                            std::ifstream ifs(entry.path(), std::ios::binary);
                            if (ifs)
                            {
                                std::string content((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
                                index_structs_in_source(content, map);
                            }
                        }
                    }
                }
            }
            return map;
        }();
        return cache;
    }

    template <typename T>
    std::vector<std::string> source_header_get_nonstatic_member_names()
    {
        std::string_view full_name = extract_clean_udt_name<T>();
        if (full_name.empty())
            return {};
        auto last_colon = full_name.rfind("::");
        std::string short_name(last_colon == std::string_view::npos ? full_name : full_name.substr(last_colon + 2));
        const auto& map = get_project_source_struct_map();
        auto it = map.find(short_name);
        if (it != map.end())
            return it->second;
        return {};
    }

#if CONTAINER_EXPLORER_HAS_MSVC_PDB_REFLECTION
    template <typename T>
    constexpr std::string_view msvc_extract_udt_name() noexcept
    {
        return extract_clean_udt_name<T>();
    }

#define CONTAINER_EXPLORER_PDB_ANCHOR_CASE(K) \
    else if constexpr (N == (K)) { \
        const auto& [CONTAINER_EXPLORER_REP_##K(CONTAINER_EXPLORER_BIND, CONTAINER_EXPLORER_COMMA)] = *p; \
        (void)a0; \
    }

    template <typename T, std::size_t N>
    __declspec(noinline) const void* msvc_pdb_anchor_type(const T* p) noexcept
    {
        if (p != nullptr)
        {
            if constexpr (N == 0) {}
            CONTAINER_EXPLORER_PDB_ANCHOR_CASE(1)  CONTAINER_EXPLORER_PDB_ANCHOR_CASE(2)  CONTAINER_EXPLORER_PDB_ANCHOR_CASE(3)  CONTAINER_EXPLORER_PDB_ANCHOR_CASE(4)
            CONTAINER_EXPLORER_PDB_ANCHOR_CASE(5)  CONTAINER_EXPLORER_PDB_ANCHOR_CASE(6)  CONTAINER_EXPLORER_PDB_ANCHOR_CASE(7)  CONTAINER_EXPLORER_PDB_ANCHOR_CASE(8)
            CONTAINER_EXPLORER_PDB_ANCHOR_CASE(9)  CONTAINER_EXPLORER_PDB_ANCHOR_CASE(10) CONTAINER_EXPLORER_PDB_ANCHOR_CASE(11) CONTAINER_EXPLORER_PDB_ANCHOR_CASE(12)
            CONTAINER_EXPLORER_PDB_ANCHOR_CASE(13) CONTAINER_EXPLORER_PDB_ANCHOR_CASE(14) CONTAINER_EXPLORER_PDB_ANCHOR_CASE(15) CONTAINER_EXPLORER_PDB_ANCHOR_CASE(16)
            CONTAINER_EXPLORER_PDB_ANCHOR_CASE(17) CONTAINER_EXPLORER_PDB_ANCHOR_CASE(18) CONTAINER_EXPLORER_PDB_ANCHOR_CASE(19) CONTAINER_EXPLORER_PDB_ANCHOR_CASE(20)
            CONTAINER_EXPLORER_PDB_ANCHOR_CASE(21) CONTAINER_EXPLORER_PDB_ANCHOR_CASE(22) CONTAINER_EXPLORER_PDB_ANCHOR_CASE(23) CONTAINER_EXPLORER_PDB_ANCHOR_CASE(24)
            CONTAINER_EXPLORER_PDB_ANCHOR_CASE(25) CONTAINER_EXPLORER_PDB_ANCHOR_CASE(26) CONTAINER_EXPLORER_PDB_ANCHOR_CASE(27) CONTAINER_EXPLORER_PDB_ANCHOR_CASE(28)
            CONTAINER_EXPLORER_PDB_ANCHOR_CASE(29) CONTAINER_EXPLORER_PDB_ANCHOR_CASE(30) CONTAINER_EXPLORER_PDB_ANCHOR_CASE(31) CONTAINER_EXPLORER_PDB_ANCHOR_CASE(32)
        }
        return reinterpret_cast<const void*>(&msvc_pdb_anchor_type<T, N>);
    }

#undef CONTAINER_EXPLORER_PDB_ANCHOR_CASE

    struct pdb_enum_type_context
    {
        HANDLE proc;
        DWORD64 mod_base;
        ULONG best_type_index = 0;
        DWORD best_children_count = 0;
    };

    inline BOOL CALLBACK msvc_pdb_enum_types_cb(PSYMBOL_INFO pSymInfo, ULONG, PVOID UserContext)
    {
        auto* ctx = static_cast<pdb_enum_type_context*>(UserContext);
        if (pSymInfo != nullptr)
        {
            DWORD count = 0;
            if (SymGetTypeInfo(ctx->proc, ctx->mod_base, pSymInfo->TypeIndex, TI_GET_CHILDRENCOUNT, &count) && count > ctx->best_children_count)
            {
                ctx->best_type_index = pSymInfo->TypeIndex;
                ctx->best_children_count = count;
                return FALSE; // Stop enumeration once complete UDT definition is found
            }
        }
        return TRUE;
    }

    template <typename T, std::size_t N>
    std::vector<std::string> msvc_pdb_get_nonstatic_member_names()
    {
        const void* anchor_addr = msvc_pdb_anchor_type<T, N>(nullptr);
        static bool sym_ready = []()
        {
            HANDLE proc = GetCurrentProcess();
            SymSetOptions(SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS);
            if (!SymInitialize(proc, nullptr, TRUE))
            {
                SymRefreshModuleList(proc);
            }
            return true;
        }();
        (void)sym_ready;

        std::vector<std::string> result;
        std::string struct_name(msvc_extract_udt_name<T>());
        if (struct_name.empty())
            return result;

        HANDLE proc = GetCurrentProcess();
        DWORD64 mod_base = SymGetModuleBase64(proc, reinterpret_cast<DWORD64>(anchor_addr));
        if (mod_base == 0)
        {
            HMODULE hmod = nullptr;
            if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                                   reinterpret_cast<LPCWSTR>(anchor_addr), &hmod))
            {
                WCHAR mod_path[MAX_PATH]{};
                if (GetModuleFileNameW(hmod, mod_path, MAX_PATH) > 0)
                {
                    mod_base = SymLoadModuleExW(proc, nullptr, mod_path, nullptr, reinterpret_cast<DWORD64>(hmod), 0, nullptr, 0);
                }
            }
        }

        pdb_enum_type_context enum_ctx{proc, mod_base, 0, 0};
        SymEnumTypesByName(proc, mod_base, struct_name.c_str(), msvc_pdb_enum_types_cb, &enum_ctx);

        if (enum_ctx.best_children_count == 0)
        {
            SYMBOL_INFO_PACKAGE sip{};
            sip.si.SizeOfStruct = sizeof(SYMBOL_INFO);
            sip.si.MaxNameLen = MAX_SYM_NAME;
            if (SymGetTypeFromName(proc, mod_base, struct_name.c_str(), &sip.si))
            {
                DWORD count = 0;
                if (SymGetTypeInfo(proc, mod_base, sip.si.TypeIndex, TI_GET_CHILDRENCOUNT, &count) && count > 0)
                {
                    enum_ctx.best_type_index = sip.si.TypeIndex;
                    enum_ctx.best_children_count = count;
                }
            }
        }

        if (enum_ctx.best_children_count == 0)
            return result;

        ULONG type_index = enum_ctx.best_type_index;
        DWORD children_count = enum_ctx.best_children_count;

        std::vector<std::byte> buf(sizeof(TI_FINDCHILDREN_PARAMS) + children_count * sizeof(ULONG));
        auto* params = reinterpret_cast<TI_FINDCHILDREN_PARAMS*>(buf.data());
        params->Count = children_count;
        params->Start = 0;

        if (!SymGetTypeInfo(proc, mod_base, type_index, TI_FINDCHILDREN, params))
            return result;

        for (DWORD i = 0; i < children_count; ++i)
        {
            ULONG child_id = params->ChildId[i];
            DWORD sym_tag = 0;
            DWORD data_kind = 0;
            // SymTagData == 7, DataIsMember == 7 (non-static data member, including reference & bit-field members)
            if (SymGetTypeInfo(proc, mod_base, child_id, TI_GET_SYMTAG, &sym_tag) && sym_tag == 7 &&
                SymGetTypeInfo(proc, mod_base, child_id, TI_GET_DATAKIND, &data_kind) && data_kind == 7)
            {
                WCHAR* wname = nullptr;
                if (SymGetTypeInfo(proc, mod_base, child_id, TI_GET_SYMNAME, &wname) && wname != nullptr)
                {
                    int len = WideCharToMultiByte(CP_UTF8, 0, wname, -1, nullptr, 0, nullptr, nullptr);
                    if (len > 1)
                    {
                        std::string s(static_cast<std::size_t>(len - 1), '\0');
                        WideCharToMultiByte(CP_UTF8, 0, wname, -1, s.data(), len, nullptr, nullptr);
                        result.push_back(std::move(s));
                    }
                    LocalFree(wname);
                }
            }
        }
        return result;
    }
#endif

    template <typename T, std::size_t I>
    std::string get_struct_field_name()
    {
        using RawClean = std::remove_cvref_t<T>;
#if CONTAINER_EXPLORER_HAS_CPP26_REFLECTION
        constexpr auto ctx = std::meta::access_context::current();
        constexpr auto members = std::meta::nonstatic_data_members_of(^^RawClean, ctx);
        if constexpr (I < members.size())
        {
            constexpr std::string_view sv = std::meta::identifier_of(members[I]);
            if (!sv.empty())
                return std::string(sv);
        }
#else
        using Clean = resolve_schema_type_t<RawClean>;
        constexpr std::size_t N = aggregate_field_count<Clean>();
        if constexpr (N > 0 && N <= 32 && is_value_member_at<Clean, I, N>())
        {
            std::string_view sv(stored_field_name<Clean, I, N>.data());
            if (!sv.empty())
                return std::string(sv);
        }
#if CONTAINER_EXPLORER_HAS_CLANG_DUMP_STRUCT
        if constexpr (N > 0 && N <= 32)
        {
            static const std::vector<std::string> clang_names = clang_dump_struct_member_names<RawClean, N>();
            if (I < clang_names.size() && !clang_names[I].empty())
                return clang_names[I];
        }
#endif
#if CONTAINER_EXPLORER_HAS_MSVC_PDB_REFLECTION
        if constexpr (N > 0 && N <= 32)
        {
            static const std::vector<std::string> pdb_names = msvc_pdb_get_nonstatic_member_names<RawClean, N>();
            if (I < pdb_names.size() && !pdb_names[I].empty())
                return pdb_names[I];
        }
#endif
        if constexpr (N > 0 && N <= 32)
        {
            static const std::vector<std::string> src_names = source_header_get_nonstatic_member_names<RawClean>();
            if (src_names.size() == N && I < src_names.size() && !src_names[I].empty())
                return src_names[I];
        }
#endif
        return "field_" + std::to_string(I);
    }

    inline std::string join_path(std::string_view prefix, std::string_view suffix)
    {
        if (prefix.empty())
            return std::string(suffix);
        if (!suffix.empty() && suffix.front() == '[')
            return std::string(prefix) + std::string(suffix);
        return std::string(prefix) + "." + std::string(suffix);
    }

    // -------------------------------------------------------------------------
    // ImGui Scalar & Leaf Drawing
    // -------------------------------------------------------------------------
    template <typename T>
    constexpr ImGuiDataType imgui_data_type()
    {
        using Value = std::remove_cv_t<T>;
        if constexpr (std::is_same_v<Value, signed char> ||
                      (std::is_same_v<Value, char> && std::is_signed_v<char>))
            return ImGuiDataType_S8;
        else if constexpr (std::is_same_v<Value, unsigned char> ||
                           (std::is_same_v<Value, char> && !std::is_signed_v<char>) ||
                           std::is_same_v<Value, char8_t>)
            return ImGuiDataType_U8;
        else if constexpr (std::is_same_v<Value, short>)
            return ImGuiDataType_S16;
        else if constexpr (std::is_same_v<Value, unsigned short>)
            return ImGuiDataType_U16;
        else if constexpr (std::is_same_v<Value, int>)
            return ImGuiDataType_S32;
        else if constexpr (std::is_same_v<Value, unsigned int>)
            return ImGuiDataType_U32;
        else if constexpr (std::is_same_v<Value, long>)
            return sizeof(long) == 4 ? ImGuiDataType_S32 : ImGuiDataType_S64;
        else if constexpr (std::is_same_v<Value, unsigned long>)
            return sizeof(unsigned long) == 4 ? ImGuiDataType_U32 : ImGuiDataType_U64;
        else if constexpr (std::is_same_v<Value, long long>)
            return ImGuiDataType_S64;
        else if constexpr (std::is_same_v<Value, unsigned long long>)
            return ImGuiDataType_U64;
        else if constexpr (std::is_same_v<Value, wchar_t>)
            return sizeof(wchar_t) == 2 ? ImGuiDataType_U16 : ImGuiDataType_U32;
        else if constexpr (std::is_same_v<Value, char16_t>)
            return ImGuiDataType_U16;
        else if constexpr (std::is_same_v<Value, char32_t>)
            return ImGuiDataType_U32;
        else if constexpr (std::is_same_v<Value, float>)
            return ImGuiDataType_Float;
        else
            return ImGuiDataType_Double;
    }

    inline int string_resize_callback(ImGuiInputTextCallbackData* data)
    {
        if (data->EventFlag == ImGuiInputTextFlags_CallbackResize)
        {
            auto* str = static_cast<std::string*>(data->UserData);
            str->resize(static_cast<std::size_t>(data->BufTextLen));
            data->Buf = str->data();
        }
        return 0;
    }

    template <typename T>
    bool draw_basic_value(T& value)
    {
        using Value = std::remove_cv_t<T>;
        static_assert(std::is_fundamental_v<Value>);

        if constexpr (std::is_same_v<Value, bool>)
        {
            return ImGui::Checkbox("##value", &value);
        }
        else if constexpr (std::is_same_v<Value, long double>)
        {
            std::array<char, 128> text{};
            std::snprintf(text.data(), text.size(), "%Lg", value);
            if (ImGui::InputText("##value", text.data(), text.size(), ImGuiInputTextFlags_CharsScientific))
            {
                value = std::strtold(text.data(), nullptr);
                return true;
            }
            return false;
        }
        else
        {
            return ImGui::InputScalar("##value", imgui_data_type<Value>(), &value);
        }
    }
} // namespace container_explorer_detail

template <typename T>
bool draw_value(T&& value)
{
    using RawRef = T&&;
    using NoRef = std::remove_reference_t<RawRef>;
    using Value = std::remove_cv_t<NoRef>;
    constexpr bool is_readonly = std::is_const_v<NoRef> || !std::is_lvalue_reference_v<RawRef>;

    if constexpr (container_explorer_detail::BoolProxy<RawRef>)
    {
        bool b = static_cast<bool>(value);
        if constexpr (std::is_const_v<NoRef>)
        {
            ImGui::BeginDisabled(true);
            ImGui::Checkbox("##value", &b);
            ImGui::EndDisabled();
            return false;
        }
        else
        {
            if (ImGui::Checkbox("##value", &b))
            {
                value = b;
                return true;
            }
            return false;
        }
    }
    else if constexpr (std::is_fundamental_v<Value>)
    {
        if constexpr (is_readonly)
        {
            Value copy = value;
            ImGui::BeginDisabled(true);
            container_explorer_detail::draw_basic_value(copy);
            ImGui::EndDisabled();
            return false;
        }
        else
        {
            return container_explorer_detail::draw_basic_value(value);
        }
    }
    else if constexpr (std::is_enum_v<Value>)
    {
        using Underlying = std::underlying_type_t<Value>;
        Underlying raw = static_cast<Underlying>(value);
        if constexpr (is_readonly)
        {
            ImGui::BeginDisabled(true);
            ImGui::InputScalar("##value", container_explorer_detail::imgui_data_type<Underlying>(), &raw);
            ImGui::EndDisabled();
            return false;
        }
        else
        {
            if (ImGui::InputScalar("##value", container_explorer_detail::imgui_data_type<Underlying>(), &raw))
            {
                value = static_cast<Value>(raw);
                return true;
            }
            return false;
        }
    }
    else if constexpr (std::is_same_v<Value, std::string>)
    {
        if constexpr (is_readonly)
        {
            ImGui::TextUnformatted(value.c_str());
            return false;
        }
        else
        {
            if (ImGui::InputText("##value", value.data(), value.capacity() + 1,
                                 ImGuiInputTextFlags_CallbackResize,
                                 container_explorer_detail::string_resize_callback, &value))
            {
                value.resize(std::strlen(value.c_str()));
                return true;
            }
            return false;
        }
    }
    else if constexpr (std::is_same_v<Value, std::string_view>)
    {
        ImGui::Text("%.*s", static_cast<int>(value.size()), value.data());
        return false;
    }
    else if constexpr (std::is_pointer_v<Value>)
    {
        if (value == nullptr)
        {
            ImGui::TextUnformatted("<null>");
        }
        else if constexpr (std::is_same_v<std::remove_cv_t<std::remove_pointer_t<Value>>, char>)
        {
            ImGui::TextUnformatted(value);
        }
        else
        {
            ImGui::Text("%p", static_cast<const void*>(value));
        }
        return false;
    }
    else if constexpr (container_explorer_detail::glm_vector_info<Value>::is_vector)
    {
        using VectorInfo = container_explorer_detail::glm_vector_info<Value>;
        using Component = typename VectorInfo::value_type;

        if constexpr (is_readonly)
        {
            Value copy = value;
            ImGui::BeginDisabled(true);
            if constexpr (!std::is_same_v<Component, bool>)
            {
                ImGui::InputScalarN("##value", container_explorer_detail::imgui_data_type<Component>(),
                                    &copy[0], static_cast<int>(VectorInfo::length));
            }
            else
            {
                for (auto component = 0; component < VectorInfo::length; ++component)
                {
                    ImGui::PushID(static_cast<int>(component));
                    if (component != 0)
                        ImGui::SameLine();
                    ImGui::Checkbox("##value", &copy[component]);
                    ImGui::PopID();
                }
            }
            ImGui::EndDisabled();
            return false;
        }
        else
        {
            bool modified = false;
            if constexpr (!std::is_same_v<Component, bool>)
            {
                if (ImGui::InputScalarN("##value", container_explorer_detail::imgui_data_type<Component>(),
                                        &value[0], static_cast<int>(VectorInfo::length)))
                {
                    modified = true;
                }
            }
            else
            {
                for (auto component = 0; component < VectorInfo::length; ++component)
                {
                    ImGui::PushID(static_cast<int>(component));
                    if (component != 0)
                        ImGui::SameLine();
                    if (ImGui::Checkbox("##value", &value[component]))
                    {
                        modified = true;
                    }
                    ImGui::PopID();
                }
            }
            return modified;
        }
    }
    else
    {
        ImGui::TextUnformatted("<unsupported>");
        return false;
    }
}

namespace container_explorer_detail
{
    // -------------------------------------------------------------------------
    // Recursive Type Decomposition (Compile-Time Schema & Runtime Value Visitor)
    // -------------------------------------------------------------------------

    template <typename T>
    void collect_leaf_names_impl(std::vector<std::string>& names, std::string_view prefix);

    template <typename T, std::size_t... Is>
    void collect_tuple_names(std::vector<std::string>& names, std::string_view prefix, std::index_sequence<Is...>)
    {
        (collect_leaf_names_impl<std::tuple_element_t<Is, T>>(
             names, join_path(prefix, "[" + std::to_string(Is) + "]")),
         ...);
    }

    template <typename T, std::size_t... Is>
    void collect_aggregate_names(std::vector<std::string>& names, std::string_view prefix, std::index_sequence<Is...>)
    {
#if CONTAINER_EXPLORER_HAS_CPP26_REFLECTION
        using Clean = std::remove_cvref_t<T>;
        constexpr auto ctx = std::meta::access_context::current();
        (collect_leaf_names_impl<
             std::remove_reference_t<typename [:std::meta::type_of(std::meta::nonstatic_data_members_of(^^Clean, ctx)[Is]):]>>(
             names, join_path(prefix, get_struct_field_name<Clean, Is>())),
         ...);
#else
        using Schema = resolve_schema_type_t<std::remove_cvref_t<T>>;
        constexpr std::size_t N = sizeof...(Is);

        (collect_leaf_names_impl<field_type_at_t<Schema, Is, N>>(
             names, join_path(prefix, get_struct_field_name<Schema, Is>())),
         ...);
#endif
    }

    template <typename T>
    void collect_leaf_names_impl(std::vector<std::string>& names, std::string_view prefix)
    {
        using Clean = std::remove_cvref_t<T>;

        if constexpr (is_leaf_type_v<T>)
        {
            names.push_back(prefix.empty() ? "Value" : std::string(prefix));
        }
        else if constexpr (inherited_wrapper_traits<Clean>::is_wrapper)
        {
            using Underlying = typename inherited_wrapper_traits<Clean>::underlying_type;
            collect_leaf_names_impl<Underlying>(names, prefix);
        }
        else if constexpr (is_reference_wrapper_v<Clean>)
        {
            using Underlying = typename Clean::type;
            collect_leaf_names_impl<Underlying>(names, prefix);
        }
        else if constexpr (glm_matrix_info<Clean>::is_matrix)
        {
            using MatInfo = glm_matrix_info<Clean>;
            for (auto col = 0; col < MatInfo::cols; ++col)
            {
                collect_leaf_names_impl<typename MatInfo::col_type>(
                    names, join_path(prefix, "[" + std::to_string(col) + "]"));
            }
        }
        else if constexpr (std::is_bounded_array_v<Clean>)
        {
            using Elem = std::remove_extent_t<Clean>;
            constexpr std::size_t N = std::extent_v<Clean>;
            for (std::size_t i = 0; i < N; ++i)
            {
                collect_leaf_names_impl<Elem>(names, join_path(prefix, "[" + std::to_string(i) + "]"));
            }
        }
        else if constexpr (is_std_array_v<Clean>)
        {
            using Elem = typename Clean::value_type;
            constexpr std::size_t N = std::tuple_size_v<Clean>;
            for (std::size_t i = 0; i < N; ++i)
            {
                collect_leaf_names_impl<Elem>(names, join_path(prefix, "[" + std::to_string(i) + "]"));
            }
        }
        else if constexpr (is_std_pair_v<Clean>)
        {
            collect_leaf_names_impl<typename Clean::first_type>(names, join_path(prefix, "first"));
            collect_leaf_names_impl<typename Clean::second_type>(names, join_path(prefix, "second"));
        }
        else if constexpr (is_std_tuple_v<Clean>)
        {
            collect_tuple_names<Clean>(names, prefix, std::make_index_sequence<std::tuple_size_v<Clean>>{});
        }
        else if constexpr (std::is_aggregate_v<Clean>)
        {
            constexpr std::size_t N = aggregate_field_count<Clean>();
            if constexpr (N == 0)
            {
                names.push_back(prefix.empty() ? "<empty>" : std::string(prefix));
            }
            else
            {
                collect_aggregate_names<Clean>(names, prefix, std::make_index_sequence<N>{});
            }
        }
        else
        {
            names.push_back(prefix.empty() ? "Value" : std::string(prefix));
        }
    }

    template <typename T>
    std::vector<std::string> get_leaf_column_names()
    {
        std::vector<std::string> names;
        collect_leaf_names_impl<T>(names, "");
        return names;
    }

    // -------------------------------------------------------------------------
    // Runtime Leaf Visitor
    // -------------------------------------------------------------------------
    template <bool ForceConst, typename T, typename Callback>
    void visit_leaves_impl(T&& obj, Callback& cb);

    template <bool ForceConst, typename Tuple, typename Callback, std::size_t... Is>
    void visit_tuple_leaves(Tuple&& tup, Callback& cb, std::index_sequence<Is...>)
    {
        (visit_leaves_impl<ForceConst>(std::get<Is>(tup), cb), ...);
    }

#define CONTAINER_EXPLORER_VISIT_FIELD(i) \
    if constexpr (requires { std::addressof(a##i); }) { \
        visit_leaves_impl<ForceConst>(a##i, cb); \
    } else if constexpr (ForceConst || IsConstObj) { \
        const auto bf_copy_##i = a##i; \
        visit_leaves_impl<true>(bf_copy_##i, cb); \
    } else { \
        auto bf_copy_##i = a##i; \
        visit_leaves_impl<false>(bf_copy_##i, cb); \
        a##i = bf_copy_##i; \
    }

#define CONTAINER_EXPLORER_VISIT_CASE(K) \
    else if constexpr (N == (K)) { \
        auto& [CONTAINER_EXPLORER_REP_##K(CONTAINER_EXPLORER_BIND, CONTAINER_EXPLORER_COMMA)] = target_ref; \
        CONTAINER_EXPLORER_REP_##K(CONTAINER_EXPLORER_VISIT_FIELD, CONTAINER_EXPLORER_EMPTY) \
    }

    template <bool ForceConst, bool IsConstObj, std::size_t N, typename RefT, typename Callback>
    void visit_aggregate_structured(RefT& target_ref, Callback& cb)
    {
        if constexpr (N == 0) {}
        CONTAINER_EXPLORER_VISIT_CASE(1)  CONTAINER_EXPLORER_VISIT_CASE(2)  CONTAINER_EXPLORER_VISIT_CASE(3)  CONTAINER_EXPLORER_VISIT_CASE(4)
        CONTAINER_EXPLORER_VISIT_CASE(5)  CONTAINER_EXPLORER_VISIT_CASE(6)  CONTAINER_EXPLORER_VISIT_CASE(7)  CONTAINER_EXPLORER_VISIT_CASE(8)
        CONTAINER_EXPLORER_VISIT_CASE(9)  CONTAINER_EXPLORER_VISIT_CASE(10) CONTAINER_EXPLORER_VISIT_CASE(11) CONTAINER_EXPLORER_VISIT_CASE(12)
        CONTAINER_EXPLORER_VISIT_CASE(13) CONTAINER_EXPLORER_VISIT_CASE(14) CONTAINER_EXPLORER_VISIT_CASE(15) CONTAINER_EXPLORER_VISIT_CASE(16)
        CONTAINER_EXPLORER_VISIT_CASE(17) CONTAINER_EXPLORER_VISIT_CASE(18) CONTAINER_EXPLORER_VISIT_CASE(19) CONTAINER_EXPLORER_VISIT_CASE(20)
        CONTAINER_EXPLORER_VISIT_CASE(21) CONTAINER_EXPLORER_VISIT_CASE(22) CONTAINER_EXPLORER_VISIT_CASE(23) CONTAINER_EXPLORER_VISIT_CASE(24)
        CONTAINER_EXPLORER_VISIT_CASE(25) CONTAINER_EXPLORER_VISIT_CASE(26) CONTAINER_EXPLORER_VISIT_CASE(27) CONTAINER_EXPLORER_VISIT_CASE(28)
        CONTAINER_EXPLORER_VISIT_CASE(29) CONTAINER_EXPLORER_VISIT_CASE(30) CONTAINER_EXPLORER_VISIT_CASE(31) CONTAINER_EXPLORER_VISIT_CASE(32)
    }

#undef CONTAINER_EXPLORER_VISIT_CASE
#undef CONTAINER_EXPLORER_VISIT_FIELD
#undef CONTAINER_EXPLORER_REP_32
#undef CONTAINER_EXPLORER_REP_31
#undef CONTAINER_EXPLORER_REP_30
#undef CONTAINER_EXPLORER_REP_29
#undef CONTAINER_EXPLORER_REP_28
#undef CONTAINER_EXPLORER_REP_27
#undef CONTAINER_EXPLORER_REP_26
#undef CONTAINER_EXPLORER_REP_25
#undef CONTAINER_EXPLORER_REP_24
#undef CONTAINER_EXPLORER_REP_23
#undef CONTAINER_EXPLORER_REP_22
#undef CONTAINER_EXPLORER_REP_21
#undef CONTAINER_EXPLORER_REP_20
#undef CONTAINER_EXPLORER_REP_19
#undef CONTAINER_EXPLORER_REP_18
#undef CONTAINER_EXPLORER_REP_17
#undef CONTAINER_EXPLORER_REP_16
#undef CONTAINER_EXPLORER_REP_15
#undef CONTAINER_EXPLORER_REP_14
#undef CONTAINER_EXPLORER_REP_13
#undef CONTAINER_EXPLORER_REP_12
#undef CONTAINER_EXPLORER_REP_11
#undef CONTAINER_EXPLORER_REP_10
#undef CONTAINER_EXPLORER_REP_9
#undef CONTAINER_EXPLORER_REP_8
#undef CONTAINER_EXPLORER_REP_7
#undef CONTAINER_EXPLORER_REP_6
#undef CONTAINER_EXPLORER_REP_5
#undef CONTAINER_EXPLORER_REP_4
#undef CONTAINER_EXPLORER_REP_3
#undef CONTAINER_EXPLORER_REP_2
#undef CONTAINER_EXPLORER_REP_1
#undef CONTAINER_EXPLORER_IS_VAL_RET
#undef CONTAINER_EXPLORER_TYPE_RET
#undef CONTAINER_EXPLORER_RET
#undef CONTAINER_EXPLORER_BIND
#undef CONTAINER_EXPLORER_EMPTY
#undef CONTAINER_EXPLORER_COMMA

    template <bool ForceConst, typename T, typename Callback, std::size_t... Is>
    void visit_aggregate_leaves(T&& obj, Callback& cb, std::index_sequence<Is...>)
    {
        using Clean = std::remove_cvref_t<T>;
#if CONTAINER_EXPLORER_HAS_CPP26_REFLECTION
        constexpr auto ctx = std::meta::access_context::current();
        if constexpr (!std::is_lvalue_reference_v<T&&> || std::is_const_v<std::remove_reference_t<T>> || ForceConst)
        {
            const Clean& const_ref = obj;
            (visit_leaves_impl<ForceConst>(const_ref.[:std::meta::nonstatic_data_members_of(^^Clean, ctx)[Is]:], cb), ...);
        }
        else
        {
            Clean& mut_ref = obj;
            (visit_leaves_impl<ForceConst>(mut_ref.[:std::meta::nonstatic_data_members_of(^^Clean, ctx)[Is]:], cb), ...);
        }
#else
        constexpr std::size_t N = sizeof...(Is);
        if constexpr (!std::is_lvalue_reference_v<T&&> || std::is_const_v<std::remove_reference_t<T>> || ForceConst)
        {
            const Clean& const_ref = obj;
            visit_aggregate_structured<ForceConst, true, N>(const_ref, cb);
        }
        else
        {
            Clean& mut_ref = obj;
            visit_aggregate_structured<ForceConst, false, N>(mut_ref, cb);
        }
#endif
    }

    template <bool ForceConst, typename T, typename Callback>
    void visit_leaves_impl(T&& obj, Callback& cb)
    {
        using Clean = std::remove_cvref_t<T>;

        if constexpr (is_leaf_type_v<T>)
        {
            if constexpr (ForceConst && std::is_lvalue_reference_v<T&&>)
            {
                cb(std::as_const(obj));
            }
            else
            {
                cb(std::forward<T>(obj));
            }
        }
        else if constexpr (inherited_wrapper_traits<Clean>::is_wrapper)
        {
            using Underlying = typename inherited_wrapper_traits<Clean>::underlying_type;
            if constexpr (std::is_const_v<std::remove_reference_t<T>> || ForceConst)
            {
                const Underlying& base_ref = static_cast<const Underlying&>(obj);
                visit_leaves_impl<true>(base_ref, cb);
            }
            else
            {
                // Proxy wrappers like soa::RefWrapper<TReference> may be returned by value
                // from operator[], while holding lvalue references inside TReference.
                Underlying& base_ref = static_cast<Underlying&>(obj);
                visit_leaves_impl<ForceConst>(base_ref, cb);
            }
        }
        else if constexpr (is_reference_wrapper_v<Clean>)
        {
            visit_leaves_impl<ForceConst>(obj.get(), cb);
        }
        else if constexpr (glm_matrix_info<Clean>::is_matrix)
        {
            using MatInfo = glm_matrix_info<Clean>;
            for (auto col = 0; col < MatInfo::cols; ++col)
            {
                visit_leaves_impl<ForceConst>(obj[col], cb);
            }
        }
        else if constexpr (std::is_bounded_array_v<Clean>)
        {
            constexpr std::size_t N = std::extent_v<Clean>;
            for (std::size_t i = 0; i < N; ++i)
            {
                visit_leaves_impl<ForceConst>(obj[i], cb);
            }
        }
        else if constexpr (is_std_array_v<Clean>)
        {
            constexpr std::size_t N = std::tuple_size_v<Clean>;
            for (std::size_t i = 0; i < N; ++i)
            {
                visit_leaves_impl<ForceConst>(obj[i], cb);
            }
        }
        else if constexpr (is_std_pair_v<Clean>)
        {
            visit_leaves_impl<ForceConst>(obj.first, cb);
            visit_leaves_impl<ForceConst>(obj.second, cb);
        }
        else if constexpr (is_std_tuple_v<Clean>)
        {
            visit_tuple_leaves<ForceConst>(obj, cb, std::make_index_sequence<std::tuple_size_v<Clean>>{});
        }
        else if constexpr (std::is_aggregate_v<Clean>)
        {
            constexpr std::size_t N = aggregate_field_count<Clean>();
            if constexpr (N == 0)
            {
                cb(std::string_view("<empty>"));
            }
            else
            {
                visit_aggregate_leaves<ForceConst>(std::forward<T>(obj), cb, std::make_index_sequence<N>{});
            }
        }
        else
        {
            if constexpr (ForceConst && std::is_lvalue_reference_v<T&&>)
                cb(std::as_const(obj));
            else
                cb(std::forward<T>(obj));
        }
    }

    // -------------------------------------------------------------------------
    // Container Capability Concepts
    // -------------------------------------------------------------------------
    template <typename C>
    concept HasSizeMethod = requires(C& c) {
        { c.size() } -> std::convertible_to<std::size_t>;
    };

    template <typename C>
    concept HasUsedSizeMethod = requires(C& c) {
        { c.used_size() } -> std::convertible_to<std::size_t>;
    };

    template <typename C>
    concept HasCapacityMethod = requires(C& c) {
        { c.capacity() } -> std::convertible_to<std::size_t>;
    };

    template <typename C>
    concept HasSubscript = requires(C& c, std::size_t i) {
        c[i];
    };

    template <typename C>
    concept HasResizeMethod = !std::is_const_v<C> && requires(C& c, std::size_t n) {
        c.resize(n);
    };

    template <typename C>
    concept HasClearMethod = !std::is_const_v<C> && !HasUsedSizeMethod<C> && requires(C& c) {
        c.clear();
    };

    template <typename C>
    concept HasIsAllocatedMethod = requires(C& c, std::size_t i) {
        { c.is_allocated(i) } -> std::convertible_to<bool>;
    };

    template <typename C>
    std::size_t get_container_size(C& container)
    {
        if constexpr (HasSizeMethod<C>)
        {
            return static_cast<std::size_t>(container.size());
        }
        else if constexpr (std::ranges::sized_range<C>)
        {
            return static_cast<std::size_t>(std::ranges::size(container));
        }
        else
        {
            return static_cast<std::size_t>(std::distance(std::begin(container), std::end(container)));
        }
    }

    template <typename C, bool Subscript = HasSubscript<C>>
    struct container_element_traits
    {
        using type = decltype(std::declval<C&>()[std::size_t{0}]);
    };

    template <typename C>
    struct container_element_traits<C, false>
    {
        using type = decltype(*std::begin(std::declval<C&>()));
    };

    template <typename C>
    using container_element_t = typename container_element_traits<C>::type;

    // -------------------------------------------------------------------------
    // Type-Erased Container View Implementation
    // -------------------------------------------------------------------------
    class container_view_interface
    {
    public:
        virtual void draw(bool show_data) = 0;
        virtual void set_on_modified(std::function<void(std::size_t)> on_modified) = 0;
        virtual const std::function<void(std::size_t)>& get_on_modified() const = 0;
        virtual ImGuiID get_last_widget_id() const = 0;
        virtual ~container_view_interface() = default;
    };

    template <typename Container>
    class container_ui_view final : public container_view_interface
    {
    public:
        using ElementRef = container_element_t<Container>;
        static constexpr bool is_const_container = std::is_const_v<Container>;

        container_ui_view(Container& container, std::string name,
                          std::function<void(std::size_t)> on_modified = nullptr)
            : m_container(container),
              m_name(std::move(name)),
              m_on_modified(std::move(on_modified)),
              m_column_names(get_leaf_column_names<ElementRef>())
        {
        }

        void set_on_modified(std::function<void(std::size_t)> on_modified) override
        {
            m_on_modified = std::move(on_modified);
        }

        const std::function<void(std::size_t)>& get_on_modified() const override
        {
            return m_on_modified;
        }

        [[nodiscard]] ImGuiID get_last_widget_id() const override
        {
            return m_last_rendered_item_id;
        }

        void draw(bool show_data) override
        {
            const std::size_t element_count = show_data ? get_container_size(m_container) : 0;
            const std::size_t field_count = m_column_names.size();
            const std::size_t rows_per_page = m_rows_per_page > 0 ? m_rows_per_page : 1000;
            const std::size_t calculated_page_count = (element_count + rows_per_page - 1) / rows_per_page;
            const std::size_t page_count = calculated_page_count == 0 ? 1 : calculated_page_count;
            if (m_page >= page_count)
                m_page = page_count - 1;

            ImGui::PushID(this);

            // Header bar: Name, row & field count, optional pool/capacity metadata
            ImGui::Text("%s (%zu rows, %zu fields)", m_name.c_str(), element_count, field_count);

            if constexpr (HasUsedSizeMethod<Container>)
            {
                if (show_data)
                {
                    ImGui::SameLine();
                    ImGui::Text("| Used: %zu", static_cast<std::size_t>(m_container.used_size()));
                }
            }

            if constexpr (HasCapacityMethod<Container>)
            {
                if (show_data)
                {
                    ImGui::SameLine();
                    ImGui::Text("| Capacity: %zu", static_cast<std::size_t>(m_container.capacity()));
                }
            }

            // Pagination controls
            ImGui::SameLine();
            if (ImGui::Button("<") && m_page > 0)
                --m_page;
            ImGui::SameLine();
            if (ImGui::Button(">") && m_page + 1 < page_count)
                ++m_page;
            ImGui::SameLine();

            std::uint64_t requested_page = static_cast<std::uint64_t>(m_page + 1);
            ImGui::SetNextItemWidth(90.0f);
            if (ImGui::InputScalar("Page", ImGuiDataType_U64, &requested_page))
            {
                if (requested_page < 1)
                    requested_page = 1;
                else if (requested_page > page_count)
                    requested_page = static_cast<std::uint64_t>(page_count);
                m_page = static_cast<std::size_t>(requested_page - 1);
            }
            ImGui::SameLine();
            ImGui::Text("of %zu", page_count);

            // Optional container modification controls (Resize / Clear)
            if constexpr (HasResizeMethod<Container> || HasClearMethod<Container>)
            {
                if (show_data)
                {
                    if constexpr (HasResizeMethod<Container>)
                    {
                        ImGui::SameLine();
                        if (!m_resize_initialized)
                        {
                            m_PendingResize = static_cast<std::uint64_t>(element_count);
                            m_resize_initialized = true;
                        }
                        ImGui::SetNextItemWidth(90.0f);
                        ImGui::InputScalar("##resize_count", ImGuiDataType_U64, &m_PendingResize);
                        ImGui::SameLine();
                        if (ImGui::Button("Resize"))
                        {
                            m_container.resize(static_cast<std::size_t>(m_PendingResize));
                        }
                    }

                    if constexpr (HasClearMethod<Container>)
                    {
                        ImGui::SameLine();
                        if (ImGui::Button("Clear"))
                        {
                            m_container.clear();
                            m_PendingResize = 0;
                        }
                    }
                }
            }

            const std::size_t current_element_count = show_data ? get_container_size(m_container) : 0;
            const std::size_t first_row = m_page * rows_per_page;
            const std::size_t page_end = first_row + rows_per_page;
            const std::size_t last_row = page_end < current_element_count ? page_end : current_element_count;

            const ImGuiTableFlags table_flags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                ImGuiTableFlags_Resizable | ImGuiTableFlags_Reorderable | ImGuiTableFlags_Hideable |
                ImGuiTableFlags_ScrollX | ImGuiTableFlags_ScrollY | ImGuiTableFlags_SizingFixedFit;

            if (ImGui::BeginTable("container_table", static_cast<int>(field_count + 1), table_flags, ImVec2(0.0f, 0.0f)))
            {
                ImGui::TableSetupColumn("Index", ImGuiTableColumnFlags_NoHide);
                for (const std::string& col_name : m_column_names)
                    ImGui::TableSetupColumn(col_name.c_str());
                ImGui::TableHeadersRow();

                if (show_data && first_row < last_row)
                {
                    if constexpr (HasSubscript<Container>)
                    {
                        for (std::size_t row = first_row; row < last_row; ++row)
                        {
                            draw_row(row, m_container[row]);
                        }
                    }
                    else
                    {
                        auto it = std::begin(m_container);
                        std::advance(it, static_cast<std::ptrdiff_t>(first_row));
                        for (std::size_t row = first_row; row < last_row; ++row, ++it)
                        {
                            draw_row(row, *it);
                        }
                    }
                }

                ImGui::EndTable();
            }

            ImGui::PopID();
        }

    private:
        template <typename ElemT>
        void draw_row(std::size_t row, ElemT&& element)
        {
            ImGui::PushID(static_cast<int>(row));
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::Text("%zu", row);

            bool row_modified = false;
            int column = 1;
            auto draw_leaf = [&](auto&& leaf_val)
            {
                ImGui::TableSetColumnIndex(column++);
                ImGui::PushID(column);
                ImGui::SetNextItemWidth(-FLT_MIN);
                if (draw_value(std::forward<decltype(leaf_val)>(leaf_val)))
                {
                    row_modified = true;
                }
                m_last_rendered_item_id = ImGui::GetItemID();
                ImGui::PopID();
            };

            visit_leaves_impl<is_const_container>(std::forward<ElemT>(element), draw_leaf);
            if (row_modified && m_on_modified)
            {
                m_on_modified(row);
            }
            ImGui::PopID();
        }

        Container& m_container;
        std::string m_name;
        std::function<void(std::size_t)> m_on_modified;
        ImGuiID m_last_rendered_item_id{0};
        std::vector<std::string> m_column_names;
        std::size_t m_page{};
        std::size_t m_rows_per_page{1000};
        std::uint64_t m_PendingResize{};
        bool m_resize_initialized{false};
    };
} // namespace container_explorer_detail

class ContainerExplorer
{
public:
    using ElementModifiedCallback = std::function<void(std::size_t)>;

    ContainerExplorer() = default;
    ~ContainerExplorer() = default;

    ContainerExplorer(const ContainerExplorer&) = delete;
    ContainerExplorer& operator=(const ContainerExplorer&) = delete;
    ContainerExplorer(ContainerExplorer&&) noexcept = default;
    ContainerExplorer& operator=(ContainerExplorer&&) noexcept = default;

    template <typename Container>
    void register_container(std::string name, Container& container,
                            ElementModifiedCallback on_modified = nullptr)
    {
        auto view = std::make_unique<container_explorer_detail::container_ui_view<Container>>(
            container, name, std::move(on_modified));
        auto it = std::find_if(m_entries.begin(), m_entries.end(),
                               [&name](const Entry& e) { return e.name == name; });
        if (it != m_entries.end())
        {
            it->view = std::move(view);
        }
        else
        {
            m_entries.push_back(Entry{name, std::move(view)});
        }

        if (m_active_container.empty())
        {
            m_active_container = std::move(name);
        }
    }

    template <typename Container>
    void add_container(std::string name, Container& container,
                       ElementModifiedCallback on_modified = nullptr)
    {
        register_container(std::move(name), container, std::move(on_modified));
    }

    bool set_element_modified_callback(std::string_view name, ElementModifiedCallback on_modified)
    {
        auto it = std::find_if(m_entries.begin(), m_entries.end(),
                               [name](const Entry& e) { return e.name == name; });
        if (it == m_entries.end() || it->view == nullptr)
            return false;
        it->view->set_on_modified(std::move(on_modified));
        return true;
    }

    [[nodiscard]] const ElementModifiedCallback* get_element_modified_callback(std::string_view name) const
    {
        auto it = std::find_if(m_entries.begin(), m_entries.end(),
                               [name](const Entry& e) { return e.name == name; });
        if (it == m_entries.end() || it->view == nullptr)
            return nullptr;
        return &it->view->get_on_modified();
    }

    [[nodiscard]] ImGuiID get_last_widget_id(std::string_view name) const
    {
        auto it = std::find_if(m_entries.begin(), m_entries.end(),
                               [name](const Entry& e) { return e.name == name; });
        if (it == m_entries.end() || it->view == nullptr)
            return 0;
        return it->view->get_last_widget_id();
    }

    bool remove_container(std::string_view name)
    {
        auto it = std::find_if(m_entries.begin(), m_entries.end(),
                               [name](const Entry& e) { return e.name == name; });
        if (it == m_entries.end())
            return false;

        const bool was_active = (m_active_container == name);
        m_entries.erase(it);

        if (was_active)
        {
            m_active_container = m_entries.empty() ? std::string{} : m_entries.front().name;
        }
        return true;
    }

    [[nodiscard]] bool has_container(std::string_view name) const noexcept
    {
        return std::any_of(m_entries.begin(), m_entries.end(),
                           [name](const Entry& e) { return e.name == name; });
    }

    [[nodiscard]] std::size_t container_count() const noexcept
    {
        return m_entries.size();
    }

    void clear_containers() noexcept
    {
        m_entries.clear();
        m_active_container.clear();
    }

    void set_active_container(std::string_view name)
    {
        if (has_container(name))
            m_active_container = std::string(name);
    }

    [[nodiscard]] const std::string& active_container() const noexcept
    {
        return m_active_container;
    }

    void draw_contents(bool show_data = true)
    {
        if (m_entries.empty())
        {
            ImGui::TextUnformatted("No containers registered.");
            return;
        }

        if (ImGui::BeginCombo("Select container", m_active_container.c_str()))
        {
            for (const Entry& entry : m_entries)
            {
                const bool is_selected = (entry.name == m_active_container);
                if (ImGui::Selectable(entry.name.c_str(), is_selected))
                {
                    m_active_container = entry.name;
                }
                if (is_selected)
                {
                    ImGui::SetItemDefaultFocus();
                }
            }
            ImGui::EndCombo();
        }

        auto it = std::find_if(m_entries.begin(), m_entries.end(),
                               [this](const Entry& e) { return e.name == m_active_container; });
        if (it != m_entries.end() && it->view != nullptr)
        {
            it->view->draw(show_data);
        }
    }

    bool draw(const char* win_name, bool show_data = true)
    {
        bool open = true;
        if (ImGui::Begin(win_name, &open))
        {
            draw_contents(show_data);
        }
        ImGui::End();
        return open;
    }

    bool draw(const std::string& win_name, bool show_data = true)
    {
        return draw(win_name.c_str(), show_data);
    }

private:
    struct Entry
    {
        std::string name;
        std::unique_ptr<container_explorer_detail::container_view_interface> view;
    };

    std::vector<Entry> m_entries;
    std::string m_active_container;
};
