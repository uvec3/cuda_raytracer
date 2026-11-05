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
#include <functional>
#include <iterator>
#include <memory>
#include <ranges>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <boost/pfr.hpp>
#include "imgui.h"

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

    template <typename T, std::size_t N = 0, std::size_t Max = 64>
    constexpr std::size_t find_top_aggregate_arity()
    {
        if constexpr (test_top_init<T>(std::make_index_sequence<N>{}))
        {
            return N;
        }
        else if constexpr (N < Max)
        {
            return find_top_aggregate_arity<T, N + 1, Max>();
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
        if constexpr (requires { sizeof(Clean{}); })
        {
            return boost::pfr::tuple_size_v<Clean>;
        }
        else
        {
            return find_top_aggregate_arity<Clean>();
        }
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
    // Boost.PFR Field Name Extraction (with safe fallback for reference members)
    // -------------------------------------------------------------------------
    template <typename T, std::size_t I, std::size_t N>
    constexpr bool is_value_member_at()
    {
        using Clean = std::remove_cvref_t<T>;
        // In Boost.PFR's tie_as_tuple(const Clean&), value members become const Field&,
        // whereas non-const reference members (e.g. float& in SOA reference structs) remain non-const Field&.
        using TiedConstTuple = decltype(boost::pfr::detail::tie_as_tuple(
            std::declval<const Clean&>(), boost::pfr::detail::size_t_<N>{}));
        using ElemRef = decltype(boost::pfr::detail::sequence_tuple::get<I>(std::declval<TiedConstTuple>()));
        return std::is_const_v<std::remove_reference_t<ElemRef>>;
    }

#define CONTAINER_EXPLORER_COMMA() ,
#define CONTAINER_EXPLORER_EMPTY()
#define CONTAINER_EXPLORER_BIND(i) a##i
#define CONTAINER_EXPLORER_RET(i) if constexpr (I == (i)) return std::addressof(a##i);

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

#define CONTAINER_EXPLORER_ADDR_CASE(K) \
    else if constexpr (N == (K)) { \
        const auto& [CONTAINER_EXPLORER_REP_##K(CONTAINER_EXPLORER_BIND, CONTAINER_EXPLORER_COMMA)] = obj; \
        CONTAINER_EXPLORER_REP_##K(CONTAINER_EXPLORER_RET, CONTAINER_EXPLORER_EMPTY) \
    }

    template <typename T, std::size_t I, std::size_t N>
    consteval auto address_of_value_field() noexcept
    {
        const T& obj = boost::pfr::detail::fake_object<T>();
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
            return std::addressof(
                boost::pfr::detail::sequence_tuple::get<I>(
                    boost::pfr::detail::tie_as_tuple(obj, boost::pfr::detail::size_t_<N>{})));
        }
    }

#undef CONTAINER_EXPLORER_ADDR_CASE
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
#undef CONTAINER_EXPLORER_RET
#undef CONTAINER_EXPLORER_BIND
#undef CONTAINER_EXPLORER_EMPTY
#undef CONTAINER_EXPLORER_COMMA

    template <typename T, std::size_t N, std::size_t... Is>
    constexpr bool all_value_members_impl(std::index_sequence<Is...>)
    {
        return (is_value_member_at<T, Is, N>() && ...);
    }

    template <typename T, std::size_t I>
    std::string get_struct_field_name()
    {
        using Clean = resolve_schema_type_t<std::remove_cvref_t<T>>;
        constexpr std::size_t N = aggregate_field_count<Clean>();
#if defined(BOOST_PFR_CORE_NAME_ENABLED) && BOOST_PFR_CORE_NAME_ENABLED
        if constexpr (N > 0 && all_value_members_impl<Clean, N>(std::make_index_sequence<N>{}) && requires { sizeof(Clean{}); })
        {
            constexpr std::string_view sv = boost::pfr::get_name<I, Clean>();
            if (!sv.empty())
                return std::string(sv);
        }
        else if constexpr (N > 0 && is_value_member_at<Clean, I, N>() &&
                           (N <= 32 || all_value_members_impl<Clean, N>(std::make_index_sequence<N>{})))
        {
            constexpr auto stored = boost::pfr::detail::name_of_field<
                Clean,
                boost::pfr::detail::make_clang_wrapper(address_of_value_field<Clean, I, N>())>();
            std::string_view sv(stored.data());
            if (!sv.empty())
                return std::string(sv);
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
        using Schema = resolve_schema_type_t<std::remove_cvref_t<T>>;
        constexpr std::size_t N = sizeof...(Is);
        using TiedTuple = decltype(boost::pfr::detail::tie_as_tuple(
            std::declval<Schema&>(), boost::pfr::detail::size_t_<N>{}));

        (collect_leaf_names_impl<
             std::remove_reference_t<decltype(boost::pfr::detail::sequence_tuple::get<Is>(std::declval<TiedTuple>()))>>(
             names, join_path(prefix, get_struct_field_name<Schema, Is>())),
         ...);
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

    template <bool ForceConst, typename T, typename Callback, std::size_t... Is>
    void visit_aggregate_leaves(T&& obj, Callback& cb, std::index_sequence<Is...>)
    {
        using Clean = std::remove_cvref_t<T>;
        constexpr std::size_t N = sizeof...(Is);
        if constexpr (!std::is_lvalue_reference_v<T&&> || std::is_const_v<std::remove_reference_t<T>> || ForceConst)
        {
            const Clean& const_ref = obj;
            auto tied = boost::pfr::detail::tie_as_tuple(const_ref, boost::pfr::detail::size_t_<N>{});
            (visit_leaves_impl<ForceConst>(boost::pfr::detail::sequence_tuple::get<Is>(tied), cb), ...);
        }
        else
        {
            Clean& mut_ref = obj;
            auto tied = boost::pfr::detail::tie_as_tuple(mut_ref, boost::pfr::detail::size_t_<N>{});
            (visit_leaves_impl<ForceConst>(boost::pfr::detail::sequence_tuple::get<Is>(tied), cb), ...);
        }
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
