#include <cuda_runtime.h>

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
    int *indices;
    int num_tris;
};

struct MeshPair
{
    Mesh h_mesh;
    Mesh *d_mesh;
    float3 *h_points;
    int *h_indices;
};

MeshPair get_mesh();