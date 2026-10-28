#pragma once
#include <array>
#include <cfloat>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <glm/glm.hpp>
#include "imgui.h"
// #include "../../external/soa_vector/src/soa_vector.hpp"
#include "../pool/soa_pool.hpp"

namespace soa_ui_detail
{
    template<typename T>
    struct glm_vector_info
    {
        static constexpr bool is_vector = false;
    };

    template<glm::length_t Length, typename T, glm::qualifier Qualifier>
    struct glm_vector_info<glm::vec<Length, T, Qualifier>>
    {
        static constexpr bool is_vector = true;
        static constexpr glm::length_t length = Length;
        using value_type = T;
    };

    template<typename T>
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

    template<typename T>
    void draw_basic_value(T& value)
    {
        using Value = std::remove_cv_t<T>;
        static_assert(std::is_fundamental_v<Value>);

        if constexpr (std::is_same_v<Value, bool>)
        {
            ImGui::Checkbox("##value", &value);
        }
        else if constexpr (std::is_same_v<Value, long double>)
        {
            // ImGui has no long-double scalar widget. Edit its decimal form to
            // preserve the stored type and avoid narrowing through double.
            std::array<char, 128> text{};
            std::snprintf(text.data(), text.size(), "%Lg", value);
            if (ImGui::InputText("##value", text.data(), text.size(), ImGuiInputTextFlags_CharsScientific))
                value = std::strtold(text.data(), nullptr);
        }
        else
        {
            ImGui::InputScalar("##value", imgui_data_type<Value>(), &value);
        }
    }
}

template<typename T>
void draw_value(T& value)
{
    using Value = std::remove_cvref_t<T>;
    if constexpr (std::is_fundamental_v<Value>)
    {
        soa_ui_detail::draw_basic_value(value);
    }
    else if constexpr (soa_ui_detail::glm_vector_info<Value>::is_vector)
    {
        using VectorInfo = soa_ui_detail::glm_vector_info<Value>;
        using Component = typename VectorInfo::value_type;

        if constexpr (!std::is_same_v<Component, bool>)
        {
            ImGui::InputScalarN("##value", soa_ui_detail::imgui_data_type<Component>(),
                                &value[0], static_cast<int>(VectorInfo::length));
        }
        else
        {
            // ImGui has no boolean InputScalarN variant.
            for (glm::length_t component = 0; component < VectorInfo::length; ++component)
            {
                ImGui::PushID(static_cast<int>(component));
                if (component != 0)
                    ImGui::SameLine();
                ImGui::Checkbox("##value", &value[component]);
                ImGui::PopID();
            }
        }
    }
    else
    {
        ImGui::TextUnformatted("<unsupported>");
    }
}

class soa_ui_table_interface
{
public:
    virtual void draw(bool show_data)=0;
    virtual void download()=0;
    virtual void upload()=0;
    virtual ~soa_ui_table_interface()= default;
};

// 1. The underlying C++14 Type Trait
template < template < typename... > class Base, typename Derived >
struct is_derived_from_template {
private:
    template < typename... Ts >
    static constexpr std::true_type test( const Base< Ts... >* );

    static constexpr std::false_type test( ... );

public:
    static constexpr bool value = decltype( test( std::declval< Derived* >() ) )::value;
};

template < template < typename... > class Base, typename Derived >
inline constexpr bool is_derived_from_template_v = is_derived_from_template< Base, Derived >::value;

template<typename SoaType>
class soa_ui_view:public soa_ui_table_interface
{
public:
    soa_ui_view(SoaType& soa,std::string name="View"):m_soa{soa},m_name{std::move(name)}
    {
    }

    void draw(bool show_data) override
    {
        constexpr size_t rows_per_page = 1000;
        const size_t element_count = m_soa.size();
        const size_t calculated_page_count = (element_count + rows_per_page - 1) / rows_per_page;
        const size_t page_count = calculated_page_count == 0 ? 1 : calculated_page_count;
        if (m_page >= page_count)
            m_page = page_count - 1;

        // Each leaf pointer is a column in the table.
        size_t field_count = std::size(m_soa.view().m_data);

        ImGui::PushID(this);
        ImGui::Text("%s (%zu rows, %zu fields)", m_name.c_str(), element_count, field_count);
        ImGui::SameLine();
        ImGui::Text("Used size %zu",m_soa.used_size());
        ImGui::SameLine();
        if (ImGui::Button("<") && m_page > 0)
            --m_page;
        ImGui::SameLine();
        if (ImGui::Button(">") && m_page + 1 < page_count)
            ++m_page;
        ImGui::SameLine();

        uint64_t requested_page = static_cast<uint64_t>(m_page + 1);
        ImGui::SetNextItemWidth(100.0f);
        if (ImGui::InputScalar("Page", ImGuiDataType_U64, &requested_page))
        {
            if (requested_page < 1)
                requested_page = 1;
            else if (requested_page > page_count)
                requested_page = static_cast<uint64_t>(page_count);
            m_page = static_cast<size_t>(requested_page - 1);
        }
        ImGui::SameLine();
        ImGui::Text("of %zu", page_count);

        const size_t first_row = m_page * rows_per_page;
        const size_t page_end = first_row + rows_per_page;
        const size_t last_row = page_end < element_count ? page_end : element_count;
        const ImGuiTableFlags table_flags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
            ImGuiTableFlags_Resizable | ImGuiTableFlags_Reorderable | ImGuiTableFlags_Hideable |
            ImGuiTableFlags_ScrollX | ImGuiTableFlags_ScrollY | ImGuiTableFlags_SizingFixedFit;

        if (ImGui::BeginTable("soa_table", static_cast<int>(field_count + 1), table_flags, ImVec2(0.0f, 0.0f)))
        {
            ImGui::TableSetupColumn("Index", ImGuiTableColumnFlags_NoHide);
            std::vector<std::string> field_names;
            field_names.reserve(field_count);
#ifdef BOOST_PFR_CORE_NAME_ENABLED
            for_each_ptr_in_soa_recursive_with_name(m_soa.data(), [&field_names](auto&, const std::string& name)
            {
                field_names.push_back(name);
            }, m_name);
#else
            for (size_t field = 0; field < field_count; ++field)
                field_names.emplace_back("Field " + std::to_string(field));
#endif
            for (const std::string& field_name : field_names)
                ImGui::TableSetupColumn(field_name.c_str());
            ImGui::TableHeadersRow();

            if (show_data)
            {
                for (size_t row = first_row; row < last_row; ++row)
                {
                    ImGui::PushID(static_cast<int>(row));
                    ImGui::TableNextRow();
                    ImGui::TableSetColumnIndex(0);
                    ImGui::Text("%zu", row);
                    int column=1;

                    auto callback = [&]< std::size_t I, typename T >(T& ref)
                    {
                        T* pointer=&ref;
                        ImGui::TableSetColumnIndex(static_cast<int>(column++));
                        ImGui::PushID(static_cast<int>(column));
                        if (pointer != nullptr)
                        {
                            // A negative width consumes all remaining space in the
                            // current table cell. InputScalarN splits it equally.
                            ImGui::SetNextItemWidth(-FLT_MIN);
                            draw_value(pointer[row]);
                        }
                        else
                            ImGui::TextUnformatted("<null>");
                        ImGui::PopID();
                    };
                    auto ref=m_soa.view()[0];
                    soa::iterate_struct_recursive(callback,ref);
                    ImGui::PopID();
                }
            }
            ImGui::EndTable();
        }
        ImGui::PopID();
    }


    void download() override{};
    void upload() override{};
    ~soa_ui_view() override= default;

private:
    SoaType& m_soa;
    std::string m_name;
    size_t m_page{};

};
