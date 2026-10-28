#include "glm/glm.hpp"
#include "Scene.hpp"
#include "../../external/tiny_gltf_v3.h"
#include <algorithm>



void loadMeshFromGLB(const char* filename) {
    tg3_parse_options opts;
    tg3_error_stack errors;
    tg3_model model;

    // 1. Initialize v3 tracking structures
    tg3_parse_options_init(&opts);
    tg3_error_stack_init(&errors);

    // 2. Parse the file directly (handles both .gltf and .glb automatically)
    // The "10" is the maximum parse depth allowed.
    tg3_error_code err = tg3_parse_file(&model, &errors, filename, 10, &opts);

    if (err != TG3_OK) {
        for (uint32_t i = 0; i < errors.count; i++) {
            fprintf(stderr, "Err [%d]: %s\n",
                    (int)errors.entries[i].severity,
                    errors.entries[i].message ? errors.entries[i].message : "Unknown");
        }
        tg3_error_stack_free(&errors);
        return;
    }

    // 3. Iterate through all meshes using C-style counts
    for (uint32_t m = 0; m < model.meshes_count; ++m) {
        const tg3_mesh *mesh = &model.meshes[m];

        // Strings in v3 are tg3_str (pointer + length), not null-terminated!
        printf("Mesh: %.*s\n", mesh->name.len, mesh->name.data);

        for (uint32_t p = 0; p < mesh->primitives_count; ++p) {
            const tg3_primitive *primitive = &mesh->primitives[p];

            // 4. Find the POSITION attribute (no more std::map lookups)
            int32_t position_accessor_idx = -1;
            for (uint32_t a = 0; a < primitive->attributes_count; ++a) {
                const tg3_str_int_pair *attr = &primitive->attributes[a];

                // Safe string comparison using the tg3_str length
                if (attr->key.len == 8 && strncmp(attr->key.data, "POSITION", 8) == 0) {
                    position_accessor_idx = attr->value;
                    break;
                }
            }

            if (position_accessor_idx >= 0) {

                // 5. Follow the reference chain using integer indices
                const tg3_accessor *accessor = &model.accessors[position_accessor_idx];
                const tg3_buffer_view *bufferView = &model.buffer_views[accessor->buffer_view];
                const tg3_buffer *buffer = &model.buffers[bufferView->buffer];

                // Pointer math using the data span from the arena
                const unsigned char* dataPtr = buffer->data.data + bufferView->byte_offset + accessor->byte_offset;

                // Handle stride. In v3, a byte_stride of 0 means tightly packed.
                // 3 floats (vec3) * 4 bytes = 12 bytes.
                uint32_t stride = bufferView->byte_stride == 0 ? 12 : bufferView->byte_stride;

                printf("  Vertices: %llu\n", (unsigned long long)accessor->count);

                // 6. Read the data
                uint64_t limit = accessor->count < 3 ? accessor->count : 3;
                for (uint64_t i = 0; i < limit; ++i) {
                    const float* position = (const float*)(dataPtr + i * stride);
                    printf("    v[%llu]: (%f, %f, %f)\n",
                           (unsigned long long)i, position[0], position[1], position[2]);
                }
            }
        }
    }

    // 7. Free the entire arena at once
    tg3_model_free(&model);
    tg3_error_stack_free(&errors);
}



Scene::Scene(std::string_view name):m_name(name)
{
    m_active_space=root_space;
}

void Scene::load(std::string_view data)
{
    tg3_parse_options opts;
    tg3_error_stack errors;
    tg3_model model;

    // 1. Initialize tracking structures
    tg3_parse_options_init(&opts);
    tg3_error_stack_init(&errors);

    // 2. Parse directly from the memory buffer
    // Automatically handles parsing the GLB header, JSON chunk, and binary chunk.
    tg3_error_code err = tg3_parse_glb(&model,&errors,reinterpret_cast<const uint8_t*>(data.data()),data.size(),nullptr,10,&opts);

    if (err != TG3_OK) {
        for (uint32_t i = 0; i < errors.count; i++) {
            fprintf(stderr, "Err [%d]: %s\n",
                    (int)errors.entries[i].severity,
                    errors.entries[i].message ? errors.entries[i].message : "Unknown");
        }
        tg3_error_stack_free(&errors);
        return;
    }

    std::vector<uint32_t> materialIds = loadMaterials(model,{});
    std::vector<uint32_t> meshIds = loadMeshes(model,materialIds);

    load_nodes(model, meshIds);

    // 4. Clean up allocated arena memory
    tg3_model_free(&model);
    tg3_error_stack_free(&errors);

    m_is_loaded=true;

    get_mesh_transform();
}

bool Scene::is_loaded()
{
    return m_is_loaded;
}

const std::string& Scene::get_name()
{
    return m_name;
}

void Scene::set_active_space(uint32_t space_id)
{
    TEST_ASSERT(as.spaces.is_allocated(space_id));
    m_active_space=space_id;
    m_root_box=as.spaces[m_active_space].root_box;
}

int Scene::get_root_box() const
{
    return m_root_box;
}

uint32_t Scene::get_active_space() const
{
    return m_active_space;
}

struct SubMesh
{
    size_t vertexStart = 0;
    size_t vertexCount = 0;
    size_t indexStart = 0;
    size_t indexCount = 0;

    const tg3_str_int_pair* attribute_position=nullptr;
    const tg3_str_int_pair* attribute_normal=nullptr;
    const tg3_str_int_pair* attribute_color=nullptr;
    const tg3_str_int_pair* attribute_texcoord=nullptr;

    uint32_t materialId = 0;
};

std::vector<uint32_t> Scene::loadMeshes(const tg3_model& model, const std::vector<uint32_t>& materialIds)
{
    std::vector<uint32_t> meshIds(model.meshes_count);
    // load all gltf mesh and primitive data
    for (int i = 0; i < model.meshes_count; ++i)
    {
        Mesh mesh;
        const tg3_mesh* tg3mesh = &model.meshes[i];
        mesh.name = tg3mesh->name.data != nullptr ? tg3mesh->name.data : "No Name";

        // lambda to perform attribute data copy
        auto writeAttribute = [this, &model,&mesh]<typename T>(T* dst, const tg3_str_int_pair* attr,uint32_t start)
        {
            const tg3_accessor* accessor = &model.accessors[attr->value];
            const tg3_buffer_view* bufferView = &model.buffer_views[accessor->buffer_view];
            const tg3_buffer* buffer = &model.buffers[bufferView->buffer];
            const size_t bufferOffset = bufferView->byte_offset + accessor->byte_offset;
            const size_t stride = bufferView->byte_stride != 0 ? bufferView->byte_stride : sizeof(T);

            for (uint64_t idx = 0; idx < accessor->count; ++idx)
            {
                const size_t elementOffset = bufferOffset + idx * stride;
                const float* data = reinterpret_cast<const float*>(buffer->data.data + elementOffset);
                if constexpr (std::is_same<T, glm::vec3>())
                {
                    dst[start + idx] = glm::vec3(data[0], data[1], data[2]);
                }
                else if constexpr (std::is_same<T, glm::vec2>())
                {
                    dst[start + idx] = glm::vec2(data[0], data[1]);
                }
            }
        };


        //extract submeshes
        std::vector<SubMesh>  subMeshes;
        std::vector<uint32_t> indices;
        subMeshes.resize(tg3mesh->primitives_count);
        for (int s = 0; s < tg3mesh->primitives_count; ++s)
        {
            const tg3_primitive* primitive = &tg3mesh->primitives[s];
            subMeshes[s].materialId = materialIds[primitive->material];


            for (int a = 0; a < primitive->attributes_count; ++a)
            {
                const tg3_str_int_pair* attr = &primitive->attributes[a];
                if (strcmp(attr->key.data, "POSITION") == 0)
                {
                    const tg3_accessor* accessor = &model.accessors[attr->value];
                    assert(accessor->type == TG3_TYPE_VEC3 && accessor->component_type == TG3_COMPONENT_TYPE_FLOAT);
                    subMeshes[s].vertexCount = accessor->count;
                    subMeshes[s].attribute_position=attr;
                }
                else if (strcmp(attr->key.data, "NORMAL") == 0)
                {
                    const tg3_accessor* accessor = &model.accessors[attr->value];
                    assert(accessor->type == TG3_TYPE_VEC3 && accessor->component_type == TG3_COMPONENT_TYPE_FLOAT);
                    subMeshes[s].attribute_normal=attr;
                }
                else if (strcmp(attr->key.data, "COLOR_0") == 0)
                {
                    const tg3_accessor* accessor = &model.accessors[attr->value];
                    assert(accessor->type == TG3_TYPE_VEC3 || accessor->type == TG3_TYPE_VEC4);
                    assert(accessor->component_type == TG3_COMPONENT_TYPE_FLOAT);
                    subMeshes[s].attribute_color=attr;

                }
                else if (strcmp(attr->key.data, "TEXCOORD_0") == 0)
                {
                    const tg3_accessor* accessor = &model.accessors[attr->value];
                    assert(accessor->type == TG3_TYPE_VEC2 && accessor->component_type == TG3_COMPONENT_TYPE_FLOAT);
                    subMeshes[s].attribute_texcoord=attr;
                }
            }


            // copy index data
            if (primitive->indices != -1)
            {
                const tg3_accessor* accessor = &model.accessors[primitive->indices];
                const tg3_buffer_view* bufferView = &model.buffer_views[accessor->buffer_view];
                const tg3_buffer* buffer = &model.buffers[bufferView->buffer];
                auto start=indices.size();
                indices.resize(start+accessor->count);

                subMeshes[s].indexStart = start;
                subMeshes[s].indexCount = accessor->count;

                if (accessor->component_type == TG3_COMPONENT_TYPE_UNSIGNED_INT)
                {
                    const uint32_t* buffData = reinterpret_cast<const uint32_t*>(buffer->data.data + bufferView->
                        byte_offset + accessor->byte_offset);
                    memcpy(&indices[start], buffData, accessor->count * sizeof(uint32_t));
                }
                else if (accessor->component_type == TG3_COMPONENT_TYPE_UNSIGNED_SHORT)
                {
                    const uint16_t* buffData = reinterpret_cast<const uint16_t*>(buffer->data.data + bufferView->
                        byte_offset + accessor->byte_offset);
                    for (uint64_t idx = 0; idx < accessor->count; ++idx)
                    {
                        indices[start + idx] = static_cast<uint32_t>(buffData[idx]);
                    }
                }
            }
        }

        uint32_t total_number_of_vertices=0;
        for (auto& m:subMeshes)
        {
            total_number_of_vertices+=m.vertexCount;
        }
        mesh.vertices.reserve(total_number_of_vertices);


        for (int s = 0; s < tg3mesh->primitives_count; ++s)
        {
            subMeshes[s].vertexStart = mesh.vertices.allocate(subMeshes[s].vertexCount);

            if (subMeshes[s].attribute_position)
            {
                writeAttribute(&mesh.vertices[0].position, subMeshes[s].attribute_position,subMeshes[s].vertexStart);
            }

            if (subMeshes[s].attribute_normal)
            {
                writeAttribute(&mesh.vertices[0].normal, subMeshes[s].attribute_normal,subMeshes[s].vertexStart);
            }

            if (subMeshes[s].attribute_color)
            {
                writeAttribute(&mesh.vertices[0].color, subMeshes[s].attribute_color,subMeshes[s].vertexStart);
            }

            if (subMeshes[s].attribute_texcoord)
            {
                writeAttribute(&mesh.vertices[0].uv, subMeshes[s].attribute_texcoord,subMeshes[s].vertexStart);
            }
        }

        mesh.triangles.reserve(indices.size()/3);
        for (auto& submesh:subMeshes)
        {
            auto number_of_triangles=submesh.indexCount/3;
            auto start=mesh.triangles.allocate(number_of_triangles);
            for (int i=0;i<number_of_triangles;i++)
            {
                mesh.triangles[start+i].v1= submesh.vertexStart + indices[submesh.indexStart+i*3];
                mesh.triangles[start+i].v2= submesh.vertexStart+ indices[submesh.indexStart+i*3+1];
                mesh.triangles[start+i].v3= submesh.vertexStart+ indices[submesh.indexStart+i*3+2];
                mesh.triangles[start+i].material_id=submesh.materialId;
            }
        }

        m_meshes.push_back(std::move(mesh));
        meshIds[i] = m_meshes.size();
    }
    return meshIds;
}

std::vector<uint32_t> Scene::loadMaterials(const tg3_model& model, const std::vector<uint32_t>& textureIds)
{
    std::vector<uint32_t> materialIds(model.materials_count);
    for (int i = 0; i < model.materials_count; ++i)
    {
        const tg3_material *tg3mat = &model.materials[i];
        m_materials.push_back(Material
            {
                .baseColor = glm::vec4(
                    tg3mat->pbr_metallic_roughness.base_color_factor[0],
                    tg3mat->pbr_metallic_roughness.base_color_factor[1],
                    tg3mat->pbr_metallic_roughness.base_color_factor[2],
                    tg3mat->pbr_metallic_roughness.base_color_factor[3]),
                .textureIndex = 0
                // .textureIndex = tg3mat->pbr_metallic_roughness.base_color_texture.index != -1
                // 	? textureIds[tg3mat->pbr_metallic_roughness.base_color_texture.index] - 1
                // 	: 0
            });
        materialIds[i] = m_materials.size();
    }
    return materialIds;
}


uint32_t Scene::importNode(const tg3_model &model,uint32_t nodeIndex, uint32_t parentId,std::vector<uint32_t>& meshIds)
{
    const tg3_node &tg3Node = model.nodes[nodeIndex];
    Node node;

    if (tg3Node.has_matrix)
    {
        glm::mat4 transform(1);
        float *transformPtr = &transform[0][0];
        for (int i = 0; i < 16; ++i)
        {
            transformPtr[i] = static_cast<float>(tg3Node.matrix[i]);
        }
        node.setTransform(transform);
    }
    else
    {
        glm::vec3 translation(tg3Node.translation[0], tg3Node.translation[1], tg3Node.translation[2]);
        glm::quat rotation(tg3Node.rotation[3], tg3Node.rotation[0], tg3Node.rotation[1], tg3Node.rotation[2]);
        glm::vec3 scale(tg3Node.scale[0], tg3Node.scale[1], tg3Node.scale[2]);

        node.setTranslation(translation);
        node.setRotation(rotation);
        node.setScale(scale);
    }

    if (tg3Node.mesh != -1)
    {
        node.meshId = meshIds[tg3Node.mesh];
    }


    node.child_start=nodes.size();
    node.child_size=tg3Node.children_count;
    // iterate through child nodes
    for (int i = 0; i < tg3Node.children_count; ++i)
    {
        int32_t childIndex = tg3Node.children[i];
        importNode(model, childIndex, nodeIndex, meshIds);
    }

    nodes.push_back(node);
    return nodes.size()-1;
}

void Scene::load_nodes(const tg3_model& model, std::vector<uint32_t>& meshIds)
{
    for (auto scene_id=0;scene_id<model.scenes_count;++scene_id)
    {
        // import scene nodes
        const tg3_scene *scene = &model.scenes[model.default_scene != -1 ? model.default_scene : 0];

        // iterate over the scene's roots nodes, attach to existing root node (if present)
        for (int i = 0; i < scene->nodes_count; ++i)
        {
            uint32_t nodeId = importNode( model, scene->nodes[i], 0, meshIds);
            std::string name=std::string(scene->name.data,scene->name.data+scene->name.len);
            root_nodes.push_back(RootNode{.root_node = nodeId,.name = name});
        }
    }
}

std::vector<std::pair<glm::mat4, Mesh&>> Scene::get_mesh_transform()
{
    std::vector<std::pair<glm::mat4, Mesh&>> resut;

    for (const auto& root: root_nodes)
    {
        mesh_transform_recursive(root.root_node,glm::mat4(1),resut);
    }
    return resut;
}


void Scene::mesh_transform_recursive(uint32_t node_id,glm::mat4 transform, std::vector<std::pair<glm::mat4, Mesh&>>& result)
{
    auto&& node=nodes[node_id];
    auto current_transform=node.getTransform()*transform;
    if (node.meshId>0)
    {
        result.push_back({current_transform,m_meshes[node.meshId-1]});
    }

    for (auto i=0;i<node.child_size;++i)
    {
        mesh_transform_recursive(node.child_start+i,current_transform,result);
    }
}

