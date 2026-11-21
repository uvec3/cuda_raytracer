#pragma once

#include <chrono>
#include <limits>
#include <string_view>
#include <string>
#include <vector>

#include "../types.hpp"
#include "../pool/soa_pool_cuda.hpp"
#include "../acceleration_structures/octree/OctreeAS.cuh"
#include "../acceleration_structures/uniform_grid/UniformGridAS.cuh"
#include "../acceleration_structures/kdtree/KDTreeAS.cuh"
#include "../acceleration_structures/bvh/BVHAS.cuh"
#include "../acceleration_structures/heterogeneous/HeterogeneousAS.cuh"
#include "../acceleration_structures/optix/OptixAS.cuh"
#include "glm/detail/type_quat.hpp"
#include "glm/gtc/quaternion.hpp"

struct tg3_model;

struct Node
{
	glm::vec3 m_translation = glm::vec3(0, 0, 0);
	glm::vec3 m_scale = glm::vec3(1, 1, 1);
	glm::quat m_rotation = glm::quat(1, 0, 0, 0);
	mutable glm::mat4 m_transform = glm::mat4(1);
	mutable bool m_dirty = true;


	std::string name;
	ASRef as_ref;
	uint32_t meshId = 0;
	std::vector<uint32_t> children;

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

	glm::mat4 getTransform() const;
	void setTransform(glm::mat4 &transform);

	void get_mesh_transform_relative(glm::mat4 current_transform,
	                                 std::vector<std::pair<glm::mat4, Mesh&>>& result,
	                                 const std::vector<Node>& nodes,
	                                 std::vector<Mesh>& meshes) const;

	size_t count_subtree_meshes(const std::vector<Node>& nodes) const;
	size_t count_subtree_triangles(const std::vector<Node>& nodes, const std::vector<Mesh>& meshes) const;
	size_t count_subtree_vertices(const std::vector<Node>& nodes, const std::vector<Mesh>& meshes) const;
};

struct MeshInstanceDesc
{
	glm::mat4 transform{1.0f};
	Mesh* mesh = nullptr;
	uint32_t instance_node_id = UINT32_MAX;
	uint32_t mesh_node_id = UINT32_MAX;
	std::string instance_name;
};

enum class AnimationInterpolation : uint8_t
{
	Linear,
	Step,
	CubicSpline
};

enum class AnimationPath : uint8_t
{
	Translation,
	Rotation,
	Scale
};

struct AnimationSampler
{
	std::vector<float> inputs;
	std::vector<glm::vec4> outputs;
	AnimationInterpolation interpolation = AnimationInterpolation::Linear;
};

struct AnimationChannel
{
	uint32_t sampler_index = 0;
	uint32_t node_id = UINT32_MAX;
	AnimationPath path = AnimationPath::Translation;
};

struct Animation
{
	std::string name;
	float start_time = 0.0f;
	float end_time = 0.0f;
	std::vector<AnimationSampler> samplers;
	std::vector<AnimationChannel> channels;

	float duration() const { return (end_time >= start_time) ? (end_time - start_time) : 0.0f; }
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

	void load(std::string_view data, bool load_textures = true);
	void sync_materials_to_device();
	void destroy_textures();

	bool is_loaded() const;
	const std::string& get_name() const;
	void set_active_space(uint32_t space_id);
	int get_root_box() const;
	uint32_t get_active_space() const;

	void free_space(uint32_t space_id);

	uint32_t build_node_as(uint32_t node_id, ASType type);
	void free_node_as(uint32_t node_id);
	bool apply_animation(uint32_t animation_index, float time_sec, bool loop = false);
	bool apply_animations(const std::vector<uint32_t>& animation_indices, float time_sec, bool loop = false);
	bool rebuild_active_as();

	bool is_mesh_node_shared(uint32_t node_id) const;
	std::vector<std::pair<glm::mat4, Mesh&>> get_mesh_transform(const Node* node = nullptr);

	Node& get_root_node() { return m_nodes[root_node_id]; }
	const Node& get_root_node() const { return m_nodes[root_node_id]; }

	struct Config
	{
		bool swapYZ = true;
		bool sort_space = false;
		PrimitiveType triangle_type = PrimitiveType::Triangle;
	};

	Config config;
	void config_ui();

	std::vector<Mesh> m_meshes;
	soa::cuda_device_vector<Vertex> d_source_vertices;
	soa::cuda_device_vector<Triangle> d_source_triangles;
	std::vector<Texture> m_textures;
	soa::soa_vector<Material> m_materials;
	soa::cuda_device_vector<Material> d_materials;
	std::vector<Node> m_nodes;
	std::vector<Animation> m_animations;
	uint32_t root_node_id = 0;
	std::vector<uint32_t> m_mesh_node_ids;
	uint32_t default_scene = 0;

	uint32_t root_space = 0;

	soa_pool<Space> spaces;
	GeometryContext geometry_context;

	OctreeAS as_octree;
	UniformGridAS as_uniform_grid;
	KDTreeAS as_kdtree;
	BVHAS as_bvh;
	HeterogeneousAS as_heterogeneous;
	OptixAS as_optix;

	bool m_is_loaded = false;
	std::string m_name;

private:
	uint32_t m_active_space = 0;
	int m_root_box = -1;
	int32_t* m_d_aabb_scratch = nullptr;

	uint32_t create_space();
	void release_space_resources(uint32_t space_id);
	void normalize_space(uint32_t space_id, glm::vec3 aabb_min, glm::vec3 aabb_max, bool stretch, bool margin = false);
	void sort_space(uint32_t space_id, soa::cuda_device_vector<Triangle>& d_temp_triangles);
	void pack_primitives(uint32_t space_id, const soa::cuda_device_vector<Triangle>& d_temp_triangles, uint32_t triangles_count, PrimitiveType triangle_type);

	template<range_of<std::pair<glm::mat4, Mesh&>> Range>
	void prepare_and_pack_space(uint32_t space_id, Range&& meshes, ASType type, bool swap_yz, bool stretch, bool do_sort, PrimitiveType triangle_type);

	template<range_of<std::pair<glm::mat4, Mesh&>> Range>
	uint32_t build_as_into_space(uint32_t space_id, Range&& meshes, ASType type);

	uint32_t build_heterogeneous_instances(const std::vector<MeshInstanceDesc>& instances);
	void collect_subtree_instances(uint32_t node_id, glm::mat4 current_transform, std::vector<MeshInstanceDesc>& out) const;
	void free_tlas_space_only(uint32_t space_id);

	std::vector<uint32_t> loadTextures(const tg3_model& model);
	std::vector<uint32_t> loadMeshes(const tg3_model& model, const std::vector<uint32_t>& materialIds);
	std::vector<uint32_t> loadMaterials(const tg3_model& model, const std::vector<uint32_t>& textureIds);
	uint32_t importNode(const tg3_model& model, uint32_t nodeIndex, const std::vector<uint32_t>& meshIds, std::vector<uint32_t>& importedGltfNodes);
	std::vector<uint32_t> load_nodes(const tg3_model& model, const std::vector<uint32_t>& meshIds);
	void load_animations(const tg3_model& model, const std::vector<uint32_t>& importedGltfNodes);
};
