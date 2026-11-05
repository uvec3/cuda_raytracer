# imgui_container_explorer

A C++20 header-only library for viewing and editing generic C++ containers (`std::vector`, `std::list`, `std::map`, `std::array`, `soa::soa_vector`, `soa::aosoa_vector`, `soa_pool`, etc.) via Dear ImGui and Boost.PFR reflection.

## Features

- **Header-Only**: Single header [`include/container_explorer.hpp`](include/container_explorer.hpp).
- **Generic Container Support**: Works with random-access indexable containers (`operator[]` + `size()`) for $O(1)$ paginated slicing as well as general iterable ranges (`begin()` / `end()`, such as `std::list`, `std::set`, `std::map`).
- **Recursive Type & Reference Decomposition**:
  - Automatically flattens nested aggregate structs, reference-member structs (like `soa::soa_vector` / `soa::aosoa_vector` proxy elements), `soa::RefWrapper<T>`, `std::reference_wrapper<T>`, `std::pair`, `std::tuple`, fixed-size arrays, and `glm::mat` into table columns.
  - Uses **Boost.PFR** for compile-time field name reflection (`transform.pos.x`, `flags[0]`, etc.) with automatic fallback for C++20 reference members.
- **In-Place Editing & Modification Callbacks**:
  - Editable widgets for fundamentals, enums, `std::string`, `std::vector<bool>` proxies, and `glm::vec<L, T, Q>`.
  - Automatically disables editing (`ImGui::BeginDisabled`) for `const` containers or `const` fields (such as `std::map` keys or `std::set` elements).
  - Optional modification callback `on_modified(std::size_t element_id)` signaled when an element is manually edited in the UI.
- **Container Metadata & Controls**:
  - Automatically detects `.capacity()`, `.used_size()`, `.resize(n)`, and `.clear()` via C++20 concepts.

## Dependencies (Fetched via Git)

When building tests or examples, the following dependencies are fetched automatically from GitHub via CMake `FetchContent`:
- **Boost.PFR**: `https://github.com/boostorg/pfr.git` (lightweight, zero other Boost library dependencies)
- **Dear ImGui**: `https://github.com/ocornut/imgui.git` (docking branch)
- **soa_vector**: `https://github.com/uvec3/soa_vector.git`
- **GoogleTest**: `https://github.com/google/googletest.git`

## Quick Start

```cpp
#include <container_explorer.hpp>

ContainerExplorer explorer;

// Register any container by reference (with optional element modification callback)
explorer.register_container("Entities", my_entities_vector, [](std::size_t element_id) {
    std::cout << "Entity #" << element_id << " was modified manually!\n";
});
// add_container is also supported as an alias
explorer.add_container("Vertices SOA", my_soa_vertices);

// Inside your ImGui frame loop:
explorer.draw("Container Explorer", /*show_data=*/true);

// Remove a container when no longer needed:
explorer.remove_container("Entities");
```
