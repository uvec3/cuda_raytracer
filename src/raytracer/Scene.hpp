#pragma once

#include <string_view>

#include "../types.hpp"
#include "../types.cuh"
#include "../pool/soa_pool_cuda.hpp"
#include "../acceleration_structures/octree/OctreeAS.cuh"
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
	uint32_t meshId = 0;
	uint32_t parentId = 0;
	uint32_t child_start = 0;
	uint32_t child_size = 0;


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
};


struct  RootNode
{
	uint32_t root_node=0;
	std::string name;
};


class Scene
{
public:
    Scene(std::string_view name);
    void load(std::string_view data);
	bool is_loaded();
    const std::string& get_name();
	void set_active_space(uint32_t space_id);
	int get_root_box() const;
	uint32_t get_active_space() const;

    std::vector<uint32_t> loadMeshes(const tg3_model& model, const std::vector<uint32_t>& materialIds);
	std::vector<uint32_t> loadMaterials(const tg3_model &model, const std::vector<uint32_t> &textureIds);
    uint32_t importNode(const tg3_model& model, uint32_t nodeIndex, uint32_t parentId, std::vector<uint32_t>& meshIds);
    void load_nodes(const tg3_model& model, std::vector<uint32_t>& meshIds);

    std::vector<std::pair<glm::mat4, Mesh&>> get_mesh_transform();


	void build_triangles();

	void transformMeshVertices(Mesh& mesh,const glm::mat3& mat, glm::vec3 transpose);

    std::vector<Mesh> m_meshes;
	std::vector<Material> m_materials;

	uint32_t root_space=0;

	OctreeAS as;
	bool m_is_loaded=false;
	std::string m_name;

private:
	uint32_t m_active_space;
	int m_root_box;

	std::vector<Node> nodes;
	std::vector<RootNode> root_nodes;


    void mesh_transform_recursive(uint32_t node_id, glm::mat4 transform, std::vector<std::pair<glm::mat4, Mesh&>>& result);
};
