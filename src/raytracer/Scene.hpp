#pragma once

#include <string_view>
#include <string>
#include <vector>

#include "../types.hpp"
#include "../types.cuh"
#include "../pool/soa_pool_cuda.hpp"
#include "../acceleration_structures/octree/OctreeAS.cuh"
#include "../acceleration_structures/uniform_grid/UniformGridAS.cuh"
#include "../acceleration_structures/optix/OptixAS.cuh"
#include "glm/detail/type_quat.hpp"
#include "glm/gtc/quaternion.hpp"

struct tg3_model;

class Node
{
	glm::vec3 m_translation = glm::vec3(0, 0, 0);
	glm::vec3 m_scale = glm::vec3(1, 1, 1);
	glm::quat m_rotation = glm::quat(1, 0, 0, 0);
	glm::mat4 m_transform = glm::mat4(1);
	bool m_dirty = true;

public:
	std::string name;
	ASRef as_ref;
	uint32_t meshId = 0;
	std::vector<Node> children;

	glm::vec3 getTranslation() const { return m_translation; }
	void setTranslation(const glm::vec3 &translation)
	{
		this->m_translation = translation;
		m_dirty = true;
	}

	glm::quat getRotation() const { return m_rotation; }
	void setRotation(const glm::quat &rotation)
	{
		this->m_rotation = rotation;
		m_dirty = true;
	}

	glm::vec3 getScale() const { return m_scale; }
	void setScale(const glm::vec3 &scale)
	{
		this->m_scale = scale;
		m_dirty = true;
	}

	glm::mat4 getTransform();
	void setTransform(glm::mat4 &transform);

	void get_mesh_transform_relative(glm::mat4 current_transform,
	                                 std::vector<std::pair<glm::mat4, Mesh&>>& result,
	                                 std::vector<Mesh>& meshes);

	size_t count_subtree_meshes() const;
	size_t count_subtree_triangles(const std::vector<Mesh>& meshes) const;
	size_t count_subtree_vertices(const std::vector<Mesh>& meshes) const;
};

class Scene
{
public:
	Scene(std::string_view name);
	~Scene();

	Scene(const Scene&) = delete;
	Scene& operator=(const Scene&) = delete;
	Scene(Scene&& other) noexcept;
	Scene& operator=(Scene&& other) noexcept;

	void load(std::string_view data);
	void sync_materials_to_device();
	void destroy_textures();

	bool is_loaded() const;
	const std::string& get_name() const;
	void set_active_space(uint32_t space_id);
	int get_root_box() const;
	uint32_t get_active_space() const;

	uint32_t create_space();
	void free_space(uint32_t space_id);
	void normalize_space(uint32_t space_id, bool stretch);
	void sort_space(uint32_t space_id, soa::cuda_device_vector<Triangle>& d_temp_triangles);
	void pack_primitives(uint32_t space_id, const soa::cuda_device_vector<Triangle>& d_temp_triangles, uint32_t triangles_count, PrimitiveType triangle_type);

	template<range_of<std::pair<glm::mat4, Mesh&>> Range>
	uint32_t build_as(Range&& meshes, ASType type);

	std::vector<std::pair<glm::mat4, Mesh&>> get_mesh_transform(const Node* node = nullptr);

	std::vector<Mesh> m_meshes;
	std::vector<Texture> m_textures;
	soa::soa_vector<Material> m_materials;
	soa::cuda_device_vector<Material> d_materials;
	Node root_node;
	uint32_t default_scene = 0;

	uint32_t root_space = 0;

	soa_pool<Space> spaces;
	GeometryContext geometry_context;

	OctreeAS as_octree;
	UniformGridAS as_uniform_grid;
	OptixAS as_optix;

	bool m_is_loaded = false;
	std::string m_name;

private:
	uint32_t m_active_space = 0;
	int m_root_box = -1;

	std::vector<uint32_t> loadTextures(const tg3_model& model);
	std::vector<uint32_t> loadMeshes(const tg3_model& model, const std::vector<uint32_t>& materialIds);
	std::vector<uint32_t> loadMaterials(const tg3_model& model, const std::vector<uint32_t>& textureIds);
	Node importNode(const tg3_model& model, uint32_t nodeIndex, const std::vector<uint32_t>& meshIds);
	void load_nodes(const tg3_model& model, const std::vector<uint32_t>& meshIds);
};

template<range_of<std::pair<glm::mat4, Mesh&>> Range>
inline uint32_t Scene::build_as(Range&& meshes, ASType type)
{
	if (type == ASType::Octree)
	{
		size_t triangles_size = 0;
		size_t vertices_size = 0;

		for (auto& mesh : meshes)
		{
			triangles_size += mesh.second.triangles.size();
			vertices_size += mesh.second.vertices.size();
		}

		auto space_id = create_space();
		auto vertices_start = static_cast<uint32_t>(geometry_context.vertices.alloc(vertices_size));

		spaces[space_id].vertices_start = vertices_start;
		spaces[space_id].vertices_size = static_cast<uint32_t>(vertices_size);
		spaces[space_id].as_type = type;

		soa::soa_vector<Triangle> temp_triangles(triangles_size);
		soa::soa_vector<Vertex> temp_vertices(vertices_size);
		size_t tri_offset = 0;
		size_t vert_offset = 0;

		for (auto& [transform, mesh] : meshes)
		{
			for (size_t i = 0; i < mesh.triangles.size(); ++i)
			{
				const auto& src_triangle = mesh.triangles[i];
				auto&& dst_triangle = temp_triangles[tri_offset + i];

				dst_triangle.type = PrimitiveType::Triangle;
				dst_triangle.material_id = src_triangle.material_id;
				dst_triangle.v0 = static_cast<uint32_t>(src_triangle.v0 + vert_offset + vertices_start);
				dst_triangle.v1 = static_cast<uint32_t>(src_triangle.v1 + vert_offset + vertices_start);
				dst_triangle.v2 = static_cast<uint32_t>(src_triangle.v2 + vert_offset + vertices_start);
			}

			for (size_t i = 0; i < mesh.vertices.size(); ++i)
			{
				const auto& src_vertex = mesh.vertices[i];
				auto&& dst_vertex = temp_vertices[vert_offset + i];

				dst_vertex = src_vertex;

				dst_vertex.position = transform * glm::vec4(dst_vertex.position, 1);
				dst_vertex.normal = glm::normalize(transform * glm::vec4(dst_vertex.normal, 0));
				if (as_octree.config.swapYZ)
				{
					std::swap(dst_vertex.position.y, dst_vertex.position.z);
					dst_vertex.position.y = -dst_vertex.position.y;
					std::swap(dst_vertex.normal.y, dst_vertex.normal.z);
					dst_vertex.normal.y = -dst_vertex.normal.y;
				}
			}

			tri_offset += mesh.triangles.size();
			vert_offset += mesh.vertices.size();
		}

		geometry_context.vertices.range(vertices_start, vertices_size) = temp_vertices;
		soa::cuda_device_vector<Triangle> d_temp_triangles = temp_triangles;

		normalize_space(space_id, as_octree.config.stretch);
		if (as_octree.config.sort_space)
		{
			sort_space(space_id, d_temp_triangles);
		}

		pack_primitives(space_id, d_temp_triangles, static_cast<uint32_t>(triangles_size), as_octree.config.triangle_type);

		as_octree.build(space_id, spaces[space_id]);
		geometry_context.d_spaces = spaces;
		return space_id;
	}
	else if (type == ASType::Optix)
	{
		size_t triangles_size = 0;
		size_t vertices_size = 0;

		for (auto& mesh : meshes)
		{
			triangles_size += mesh.second.triangles.size();
			vertices_size += mesh.second.vertices.size();
		}

		auto space_id = create_space();
		auto vertices_start = static_cast<uint32_t>(geometry_context.vertices.alloc(vertices_size));

		spaces[space_id].vertices_start = vertices_start;
		spaces[space_id].vertices_size = static_cast<uint32_t>(vertices_size);
		spaces[space_id].as_type = type;

		soa::soa_vector<Triangle> temp_triangles(triangles_size);
		soa::soa_vector<Vertex> temp_vertices(vertices_size);
		size_t tri_offset = 0;
		size_t vert_offset = 0;

		for (auto& [transform, mesh] : meshes)
		{
			for (size_t i = 0; i < mesh.triangles.size(); ++i)
			{
				const auto& src_triangle = mesh.triangles[i];
				auto&& dst_triangle = temp_triangles[tri_offset + i];

				dst_triangle.type = PrimitiveType::Triangle;
				dst_triangle.material_id = src_triangle.material_id;
				dst_triangle.v0 = static_cast<uint32_t>(src_triangle.v0 + vert_offset + vertices_start);
				dst_triangle.v1 = static_cast<uint32_t>(src_triangle.v1 + vert_offset + vertices_start);
				dst_triangle.v2 = static_cast<uint32_t>(src_triangle.v2 + vert_offset + vertices_start);
			}

			for (size_t i = 0; i < mesh.vertices.size(); ++i)
			{
				const auto& src_vertex = mesh.vertices[i];
				auto&& dst_vertex = temp_vertices[vert_offset + i];

				dst_vertex = src_vertex;

				dst_vertex.position = transform * glm::vec4(dst_vertex.position, 1);
				dst_vertex.normal = glm::normalize(transform * glm::vec4(dst_vertex.normal, 0));
				if (as_octree.config.swapYZ)
				{
					std::swap(dst_vertex.position.y, dst_vertex.position.z);
					dst_vertex.position.y = -dst_vertex.position.y;
					std::swap(dst_vertex.normal.y, dst_vertex.normal.z);
					dst_vertex.normal.y = -dst_vertex.normal.y;
				}
			}

			tri_offset += mesh.triangles.size();
			vert_offset += mesh.vertices.size();
		}

		geometry_context.vertices.range(vertices_start, vertices_size) = temp_vertices;
		soa::cuda_device_vector<Triangle> d_temp_triangles = temp_triangles;

		normalize_space(space_id, as_octree.config.stretch);

		pack_primitives(space_id, d_temp_triangles, static_cast<uint32_t>(triangles_size), PrimitiveType::Triangle);

		as_optix.build(space_id, spaces[space_id]);
		geometry_context.d_spaces = spaces;
		return space_id;
	}
	else if (type == ASType::UniformGrid)
	{
		return 0;
	}
	return 0;
}
