#include <gtest/gtest.h>
#include <container_explorer.hpp>
#include <imgui_internal.h>

#include <array>
#include <deque>
#include <list>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <vector>

#if defined(_MSC_VER) && !defined(__CUDACC__) && !defined(__forceinline__)
#define __forceinline__ __forceinline
#endif

#include "soa_vector.hpp"
#include "aosoa_vector.hpp"

namespace test_types
{
    template <typename T>
    class TestPool
    {
    public:
        TestPool(std::size_t size = 0) : m_storage(size), m_used_size(0) {}
        std::size_t size() const { return m_storage.size(); }
        std::size_t capacity() const { return m_storage.capacity(); }
        std::size_t used_size() const { return m_used_size; }
        T& operator[](std::size_t i) { return m_storage[i]; }
        const T& operator[](std::size_t i) const { return m_storage[i]; }
        std::size_t alloc_element()
        {
            std::size_t idx = m_storage.size();
            m_storage.emplace_back();
            ++m_used_size;
            return idx;
        }
    private:
        std::vector<T> m_storage;
        std::size_t m_used_size{0};
    };
    enum class Status : int
    {
        Idle = 0,
        Active = 1,
        Done = 2
    };

    struct Vec2Simple
    {
        float x{};
        float y{};
    };

    struct TransformStruct
    {
        Vec2Simple pos{};
        float scale{1.0f};
    };

    struct ComplexEntity
    {
        int id{};
        Status status{Status::Idle};
        TransformStruct transform{};
        std::array<int, 2> flags{};
        std::string name;
#if CONTAINER_EXPLORER_HAS_GLM
        glm::vec3 velocity{0.0f};
#endif
    };

    // SOA reference types (matching soa_vector tests)
    struct SoaNested
    {
        float& n1;
        double& n2;
    };

    struct SoaTypeRef
    {
        float& a;
        int& b;
        SoaNested nested1;
        SoaNested nested2;
    };

    // SOA reference types paired with value structs for full reference name reflection
    struct SoaNestedValue
    {
        float n1;
        double n2;
    };

    struct SoaTypeValue
    {
        float a;
        int b;
        SoaNestedValue nested1;
        SoaNestedValue nested2;
    };

    struct SoaNamedTypeRef
    {
        using value_type = SoaTypeValue;
        float& a;
        int& b;
        SoaNested nested1;
        SoaNested nested2;
    };

    // Mixed reference + value struct with > 8 fields (tests macro-generated address_of_value_field up to 32)
    struct SoaWideMixedRef
    {
        float& r0;
        int& r1;
        float& r2;
        int& r3;
        float& r4;
        int& r5;
        float& r6;
        int& r7;
        SoaNested ninth_sub;
        SoaNested tenth_sub;
    };
} // namespace test_types

using namespace test_types;

namespace
{
    // Scoped headless ImGui context for testing draw()
    class HeadlessImGuiContext
    {
    public:
        HeadlessImGuiContext()
        {
            IMGUI_CHECKVERSION();
            m_ctx = ImGui::CreateContext();
            ImGuiIO& io = ImGui::GetIO();
            io.DisplaySize = ImVec2(1280.0f, 720.0f);
            io.DeltaTime = 1.0f / 60.0f;

            unsigned char* pixels = nullptr;
            int width = 0;
            int height = 0;
            io.Fonts->GetTexDataAsRGBA32(&pixels, &width, &height);
            io.Fonts->SetTexID(static_cast<ImTextureID>(1));
        }

        ~HeadlessImGuiContext()
        {
            if (m_ctx != nullptr)
                ImGui::DestroyContext(m_ctx);
        }

        void begin_frame()
        {
            ImGui::NewFrame();
        }

        void end_frame()
        {
            ImGui::Render();
        }

    private:
        ImGuiContext* m_ctx{nullptr};
    };
} // namespace

TEST(ContainerExplorerReflection, BoostPfrColumnNamesForNestedStructs)
{
    auto cols = container_explorer_detail::get_leaf_column_names<ComplexEntity>();

#if defined(BOOST_PFR_CORE_NAME_ENABLED) && BOOST_PFR_CORE_NAME_ENABLED
    ASSERT_GE(cols.size(), 7u);
    EXPECT_EQ(cols[0], "id");
    EXPECT_EQ(cols[1], "status");
    EXPECT_EQ(cols[2], "transform.pos.x");
    EXPECT_EQ(cols[3], "transform.pos.y");
    EXPECT_EQ(cols[4], "transform.scale");
    EXPECT_EQ(cols[5], "flags[0]");
    EXPECT_EQ(cols[6], "flags[1]");
    EXPECT_EQ(cols[7], "name");
#if CONTAINER_EXPLORER_HAS_GLM
    ASSERT_EQ(cols.size(), 9u);
    EXPECT_EQ(cols[8], "velocity");
#endif
#endif
}

TEST(ContainerExplorerReflection, DecomposesSoaRefWrapperAndReferenceMembers)
{
    using SoaVec = soa::soa_vector<SoaTypeRef>;
    using ElemRef = container_explorer_detail::container_element_t<SoaVec>;

    auto cols = container_explorer_detail::get_leaf_column_names<ElemRef>();
    ASSERT_EQ(cols.size(), 6u);

#if defined(BOOST_PFR_CORE_NAME_ENABLED) && BOOST_PFR_CORE_NAME_ENABLED
    EXPECT_EQ(cols[0], "field_0");
    EXPECT_EQ(cols[1], "field_1");
    EXPECT_EQ(cols[2], "nested1.field_0");
    EXPECT_EQ(cols[3], "nested1.field_1");
    EXPECT_EQ(cols[4], "nested2.field_0");
    EXPECT_EQ(cols[5], "nested2.field_1");
#endif
}

TEST(ContainerExplorerReflection, ReflectsSoaReferenceLeafNamesViaValueTypeAndBeyondEightFields)
{
    using NamedSoaVec = soa::soa_vector<SoaNamedTypeRef>;
    using NamedElemRef = container_explorer_detail::container_element_t<NamedSoaVec>;

    auto named_cols = container_explorer_detail::get_leaf_column_names<NamedElemRef>();
    ASSERT_EQ(named_cols.size(), 6u);

#if defined(BOOST_PFR_CORE_NAME_ENABLED) && BOOST_PFR_CORE_NAME_ENABLED
    EXPECT_EQ(named_cols[0], "a");
    EXPECT_EQ(named_cols[1], "b");
    EXPECT_EQ(named_cols[2], "nested1.n1");
    EXPECT_EQ(named_cols[3], "nested1.n2");
    EXPECT_EQ(named_cols[4], "nested2.n1");
    EXPECT_EQ(named_cols[5], "nested2.n2");
#endif

    auto wide_cols = container_explorer_detail::get_leaf_column_names<SoaWideMixedRef>();
    ASSERT_EQ(wide_cols.size(), 12u);

#if defined(BOOST_PFR_CORE_NAME_ENABLED) && BOOST_PFR_CORE_NAME_ENABLED
    EXPECT_EQ(wide_cols[8], "ninth_sub.field_0");
    EXPECT_EQ(wide_cols[9], "ninth_sub.field_1");
    EXPECT_EQ(wide_cols[10], "tenth_sub.field_0");
    EXPECT_EQ(wide_cols[11], "tenth_sub.field_1");
#endif
}

TEST(ContainerExplorerReflection, MutatesLeavesThroughVisitorOnStdAndSoaContainers)
{
    // 1. std::vector<ComplexEntity>
    std::vector<ComplexEntity> entities(2);
    entities[0].id = 10;
    entities[0].transform.pos.x = 1.5f;
    entities[0].transform.pos.y = 2.5f;
    entities[0].transform.scale = 3.0f;

    auto increment_floats = [](auto&& leaf)
    {
        using LeafT = std::remove_cvref_t<decltype(leaf)>;
        if constexpr (std::is_same_v<LeafT, float> && !std::is_const_v<std::remove_reference_t<decltype(leaf)>>)
        {
            leaf += 10.0f;
        }
    };

    container_explorer_detail::visit_leaves_impl<false>(entities[0], increment_floats);
    EXPECT_FLOAT_EQ(entities[0].transform.pos.x, 11.5f);
    EXPECT_FLOAT_EQ(entities[0].transform.pos.y, 12.5f);
    EXPECT_FLOAT_EQ(entities[0].transform.scale, 13.0f);

    // 2. soa::soa_vector<SoaTypeRef>
    soa::soa_vector<SoaTypeRef> soa_vec(4);
    soa_vec[1].a = 2.0f;
    soa_vec[1].b = 7;
    soa_vec[1].nested1.n1 = 4.0f;
    soa_vec[1].nested1.n2 = 8.0;
    soa_vec[1].nested2.n1 = 6.0f;
    soa_vec[1].nested2.n2 = 12.0;

    container_explorer_detail::visit_leaves_impl<false>(soa_vec[1], increment_floats);
    EXPECT_FLOAT_EQ(soa_vec[1].a, 12.0f);
    EXPECT_FLOAT_EQ(soa_vec[1].nested1.n1, 14.0f);
    EXPECT_FLOAT_EQ(soa_vec[1].nested2.n1, 16.0f);

    // 3. soa::aosoa_vector<SoaTypeRef> (operator[] returns SoaTypeRef by value holding references)
    soa::aosoa_vector<SoaTypeRef> aosoa_vec(4);
    aosoa_vec[2].a = 5.0f;
    aosoa_vec[2].nested1.n1 = 9.0f;
    aosoa_vec[2].nested2.n1 = 11.0f;

    container_explorer_detail::visit_leaves_impl<false>(aosoa_vec[2], increment_floats);
    EXPECT_FLOAT_EQ(aosoa_vec[2].a, 15.0f);
    EXPECT_FLOAT_EQ(aosoa_vec[2].nested1.n1, 19.0f);
    EXPECT_FLOAT_EQ(aosoa_vec[2].nested2.n1, 21.0f);
}

TEST(ContainerExplorerRegistration, RegisterAndRemoveContainers)
{
    ContainerExplorer explorer;
    EXPECT_EQ(explorer.container_count(), 0u);

    std::vector<int> vec_a{1, 2, 3};
    std::list<double> list_b{4.0, 5.0};
    soa::soa_vector<SoaTypeRef> soa_c(10);

    explorer.register_container("VecA", vec_a);
    explorer.register_container("ListB", list_b);
    explorer.register_container("SoaC", soa_c);

    EXPECT_EQ(explorer.container_count(), 3u);
    EXPECT_TRUE(explorer.has_container("VecA"));
    EXPECT_TRUE(explorer.has_container("ListB"));
    EXPECT_TRUE(explorer.has_container("SoaC"));
    EXPECT_EQ(explorer.active_container(), "VecA");

    // Removing the active container selects the next remaining container
    EXPECT_TRUE(explorer.remove_container("VecA"));
    EXPECT_FALSE(explorer.has_container("VecA"));
    EXPECT_EQ(explorer.container_count(), 2u);
    EXPECT_EQ(explorer.active_container(), "ListB");

    // Removing a non-existent container returns false
    EXPECT_FALSE(explorer.remove_container("NonExistent"));

    EXPECT_TRUE(explorer.remove_container("ListB"));
    EXPECT_EQ(explorer.active_container(), "SoaC");

    EXPECT_TRUE(explorer.remove_container("SoaC"));
    EXPECT_EQ(explorer.container_count(), 0u);
    EXPECT_TRUE(explorer.active_container().empty());
}

TEST(ContainerExplorerUI, DrawsAllContainerTypesInHeadlessImGui)
{
    HeadlessImGuiContext imgui;
    ContainerExplorer explorer;

    std::vector<ComplexEntity> entities(5);
    entities[0].id = 42;
    entities[0].name = "Hero";

    std::list< std::pair<int, std::string> > pairs{{1, "one"}, {2, "two"}};
    std::map<std::string, Vec2Simple> map_data{{"origin", {0.0f, 0.0f}}, {"target", {10.0f, 20.0f}}};
    std::vector<bool> bitset{true, false, true, true};
    const std::array<float, 4> const_arr{1.0f, 2.0f, 3.0f, 4.0f};

    soa::soa_vector<SoaTypeRef> soa_vec(15);
    for (std::size_t i = 0; i < soa_vec.size(); ++i)
    {
        soa_vec[i].a = static_cast<float>(i);
        soa_vec[i].b = static_cast<int>(i * 2);
        soa_vec[i].nested1.n1 = static_cast<float>(i) * 0.5f;
        soa_vec[i].nested1.n2 = static_cast<double>(i) * 1.5;
        soa_vec[i].nested2.n1 = static_cast<float>(i) * 2.5f;
        soa_vec[i].nested2.n2 = static_cast<double>(i) * 3.5;
    }

    soa::aosoa_vector<SoaTypeRef> aosoa_vec(8);
    TestPool<ComplexEntity> pool(16);
    auto idx0 = pool.alloc_element();
    pool[idx0].id = 99;

    explorer.register_container("Entities (std::vector)", entities);
    explorer.register_container("Pairs (std::list)", pairs);
    explorer.register_container("Lookup (std::map)", map_data);
    explorer.register_container("Flags (std::vector<bool>)", bitset);
    explorer.register_container("ConstArray (const std::array)", const_arr);
    explorer.register_container("SOA Vector", soa_vec);
    explorer.register_container("AoSoA Vector", aosoa_vec);
    explorer.register_container("Pool Container", pool);

    const std::vector<std::string> names = {
        "Entities (std::vector)",
        "Pairs (std::list)",
        "Lookup (std::map)",
        "Flags (std::vector<bool>)",
        "ConstArray (const std::array)",
        "SOA Vector",
        "AoSoA Vector",
        "Pool Container"
    };

    for (const auto& name : names)
    {
        explorer.set_active_container(name);
        ASSERT_EQ(explorer.active_container(), name);

        imgui.begin_frame();
        EXPECT_TRUE(explorer.draw("Container Explorer", true));
        imgui.end_frame();

        imgui.begin_frame();
        EXPECT_TRUE(explorer.draw("Container Explorer", false));
        imgui.end_frame();
    }
}

TEST(ContainerExplorerCallback, DrawValueReturnsModifiedState)
{
    HeadlessImGuiContext imgui;

    // Const value: should never be modified
    {
        const int const_val = 123;
        imgui.begin_frame();
        EXPECT_FALSE(draw_value(const_val));
        imgui.end_frame();
    }

    // Mutable bool: test activation
    {
        bool flag = false;

        // Frame 1: draw item, query its ID
        imgui.begin_frame();
        ImGui::PushID("test_flag");
        ImGuiID flag_id = ImGui::GetID("##value");
        EXPECT_FALSE(draw_value(flag));
        ImGui::PopID();
        imgui.end_frame();
        EXPECT_FALSE(flag);

        // Queue activation
        ImGui::ActivateItemByID(flag_id);

        // Frame 2: activation processed, should return true and mutate flag
        imgui.begin_frame();
        ImGui::PushID("test_flag");
        EXPECT_TRUE(draw_value(flag));
        ImGui::PopID();
        imgui.end_frame();
        EXPECT_TRUE(flag);
    }
}

TEST(ContainerExplorerCallback, ElementModifiedCallbackSignalledOnStdVector)
{
    HeadlessImGuiContext imgui;
    ContainerExplorer explorer;
    std::vector<bool> flags{false, false, false};
    std::size_t modified_element = 999;
    explorer.register_container("Flags", flags, [&](std::size_t id) {
        modified_element = id;
    });

    // Verify getter
    const auto* cb = explorer.get_element_modified_callback("Flags");
    ASSERT_NE(cb, nullptr);
    EXPECT_TRUE(*cb != nullptr);

    // Initial frame: no modifications
    imgui.begin_frame();
    explorer.draw("TestWin");
    ImGuiID last_item_id = explorer.get_last_widget_id("Flags");
    imgui.end_frame();

    ASSERT_NE(last_item_id, 0u);
    EXPECT_FALSE(flags[2]);
    EXPECT_EQ(modified_element, 999u);

    // Frame 2: Activate the last item ID (element 2)
    ImGui::ActivateItemByID(last_item_id);

    imgui.begin_frame();
    explorer.draw("TestWin");
    imgui.end_frame();

    // Verify element 2 was toggled and callback received id 2
    EXPECT_TRUE(flags[2]);
    EXPECT_EQ(modified_element, 2u);

    // Frame 3: Next frame without activation: callback is not called again
    modified_element = 999;
    imgui.begin_frame();
    explorer.draw("TestWin");
    imgui.end_frame();
    EXPECT_EQ(modified_element, 999u);
}

TEST(ContainerExplorerCallback, AddContainerAliasAndDynamicCallbackUpdate)
{
    HeadlessImGuiContext imgui;
    ContainerExplorer explorer;
    std::vector<bool> flags{false, false};
    std::size_t first_cb_id = 999;
    std::size_t second_cb_id = 999;

    // Use add_container alias
    explorer.add_container("Flags", flags, [&](std::size_t id) {
        first_cb_id = id;
    });

    // Update callback dynamically via set_element_modified_callback
    EXPECT_TRUE(explorer.set_element_modified_callback("Flags", [&](std::size_t id) {
        second_cb_id = id;
    }));
    EXPECT_FALSE(explorer.set_element_modified_callback("NonExistent", nullptr));

    imgui.begin_frame();
    explorer.draw("TestWin");
    ImGuiID last_item_id = explorer.get_last_widget_id("Flags");
    imgui.end_frame();

    ASSERT_NE(last_item_id, 0u);
    ImGui::ActivateItemByID(last_item_id);

    imgui.begin_frame();
    explorer.draw("TestWin");
    imgui.end_frame();

    EXPECT_TRUE(flags[1]);
    EXPECT_EQ(first_cb_id, 999u); // Old callback was replaced
    EXPECT_EQ(second_cb_id, 1u);   // New callback was invoked with element id 1
}

TEST(ContainerExplorerCallback, OptionalCallbackOmittedDoesNotCrashOnEdit)
{
    HeadlessImGuiContext imgui;
    ContainerExplorer explorer;
    std::vector<bool> flags{false};

    // Register without callback
    explorer.register_container("Flags", flags);
    const auto* cb = explorer.get_element_modified_callback("Flags");
    ASSERT_NE(cb, nullptr);
    EXPECT_FALSE(*cb != nullptr);

    imgui.begin_frame();
    explorer.draw("TestWin");
    ImGuiID item_id = explorer.get_last_widget_id("Flags");
    imgui.end_frame();

    ASSERT_NE(item_id, 0u);
    ImGui::ActivateItemByID(item_id);

    imgui.begin_frame();
    explorer.draw("TestWin");
    imgui.end_frame();

    EXPECT_TRUE(flags[0]); // Mutated cleanly without callback
}

TEST(ContainerExplorerCallback, ConstContainerNeverTriggersCallback)
{
    HeadlessImGuiContext imgui;
    ContainerExplorer explorer;
    const std::vector<int> const_ints{1, 2, 3};
    bool called = false;

    explorer.register_container("ConstInts", const_ints, [&](std::size_t) {
        called = true;
    });

    imgui.begin_frame();
    explorer.draw("TestWin");
    ImGuiID item_id = explorer.get_last_widget_id("ConstInts");
    imgui.end_frame();

    if (item_id != 0u)
    {
        ImGui::ActivateItemByID(item_id);
    }

    imgui.begin_frame();
    explorer.draw("TestWin");
    imgui.end_frame();

    EXPECT_FALSE(called);
}
