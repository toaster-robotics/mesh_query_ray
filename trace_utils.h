#include <cuda_runtime.h>
#include <vector>

struct mesh_query_ray_t
{
    bool result;
    float u, v, t;
    int face;

    float3 hit_point;
    float ray_distance;
    float3 normal;
};

struct Mesh
{
    float3 *points;
    uint3 *indices;
    int num_tris;
};

struct MeshPair
{
    Mesh h_mesh;
    Mesh *d_mesh;
    std::vector<float3> h_points;
    std::vector<uint3> h_indices;
};

MeshPair get_mesh();