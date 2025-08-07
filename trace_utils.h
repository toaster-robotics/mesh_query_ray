#include <cuda_runtime.h>
#include <vector>
#include "bvh.h"

struct mesh_query_ray_t
{
    bool result;
    float u, v, t;
    int face;

    float3 hit_point;
    float3 normal;
};

struct Mesh
{
    float3 *vertices;
    uint3 *indices;
    int num_tris;
};

struct MeshContainer
{
    Mesh host;
    Mesh *device;
    std::vector<float3> vertices;
    std::vector<uint3> indices;

    float3 center;
    float radius;
};

MeshContainer get_mesh();

struct BVHContainer
{
    BVH *device_bvh;
    BVHPackedNode *device_nodes;
};

BVHContainer get_bvh(const MeshContainer &mesh);

struct MinMaxResult
{
    float2 min_point;
    float2 max_point;
};

MinMaxResult computeMinMaxPoints(
    const std::vector<float3> &mesh_points,
    const float3 &r_hat,
    const float3 &phi_hat,
    const float3 &theta_hat);