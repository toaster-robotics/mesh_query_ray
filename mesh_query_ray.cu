// mesh_query_ray.cu (minimal CUDA-only version)
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cmath>
#include "sah_bvh_builder.h"
// #include "vector_math.h"

#define BVH_STACK_SIZE 64

//-------------------------------------------------
// Mesh structs

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

//-------------------------------------------------
// BVH structs

struct BVH
{
    BVHPackedNode *nodes;
    int num_nodes;
};

__device__ inline bool intersect_ray_aabb(const float3 &origin, const float3 &inv_dir,
                                          const float3 &lower, const float3 &upper,
                                          float &tmin, float &tmax)
{
    float t0 = 0.0f, t1 = tmax;

    for (int i = 0; i < 3; ++i)
    {
        float invD = (&inv_dir.x)[i];
        float o = (&origin.x)[i];
        float tNear = ((&lower.x)[i] - o) * invD;
        float tFar = ((&upper.x)[i] - o) * invD;

        if (tNear > tFar)
        {
            float tmp = tNear;
            tNear = tFar;
            tFar = tmp;
        }

        t0 = fmaxf(t0, tNear);
        t1 = fminf(t1, tFar);

        if (t0 > t1)
            return false;
    }

    tmin = t0;
    tmax = t1;
    return true;
}

//-------------------------------------------------
// Ray-triangle intersection (Möller-Trumbore)

__device__ mesh_query_ray_t intersect_ray_triangle(
    const float3 &orig, const float3 &dir,
    const float3 &v0, const float3 &v1, const float3 &v2,
    float max_t, int face_index)
{
    const float EPSILON = 1e-6f;
    mesh_query_ray_t out = {};

    float3 edge1 = v1 - v0;
    float3 edge2 = v2 - v0;
    float3 h = cross(dir, edge2);
    float a = dot(edge1, h);

    if (fabs(a) < EPSILON)
        return out;

    float f = 1.0f / a;
    float3 s = orig - v0;
    float u = f * dot(s, h);
    if (u < 0.0f || u > 1.0f)
        return out;

    float3 q = cross(s, edge1);
    float v = f * dot(dir, q);
    if (v < 0.0f || u + v > 1.0f)
        return out;

    float t = f * dot(edge2, q);
    if (t > EPSILON && t < max_t)
    {
        out.result = true;
        out.u = u;
        out.v = v;
        out.t = t;
        out.face = face_index;

        out.hit_point = orig + dir * t;
        float3 n = cross(v1 - v0, v2 - v0);
        out.normal = normalize(n);
    }
    return out;
}

//-------------------------------------------------
// mesh_query_ray: brute-force (no BVH)

__device__ mesh_query_ray_t mesh_query_ray(Mesh *mesh, BVH *bvh, const float3 &origin, const float3 &dir, float max_t)
{
    mesh_query_ray_t closest_hit = {};
    closest_hit.t = max_t;

    float3 inv_dir = make_float3(1.0f / dir.x, 1.0f / dir.y, 1.0f / dir.z);

    int stack[BVH_STACK_SIZE];
    int stack_size = 0;

    stack[stack_size++] = 0; // root node index

    while (stack_size > 0)
    {
        int node_idx = stack[--stack_size];
        const BVHPackedNode &node = bvh->nodes[node_idx];

        float tmin_left = 0.0f, tmax_left = closest_hit.t;
        float tmin_right = 0.0f, tmax_right = closest_hit.t;

        bool hit_left = intersect_ray_aabb(origin, inv_dir, node.left.lower, node.left.upper, tmin_left, tmax_left);
        bool hit_right = intersect_ray_aabb(origin, inv_dir, node.right.lower, node.right.upper, tmin_right, tmax_right);

        if (node.flags == 1) // leaf
        {
            int start = node.prim_index;
            int count = node.prim_count;

            for (int i = 0; i < count; ++i)
            {
                int tri_index = start + i;
                int i0 = mesh->indices[tri_index * 3 + 0];
                int i1 = mesh->indices[tri_index * 3 + 1];
                int i2 = mesh->indices[tri_index * 3 + 2];

                float3 v0 = mesh->points[i0];
                float3 v1 = mesh->points[i1];
                float3 v2 = mesh->points[i2];

                mesh_query_ray_t hit = intersect_ray_triangle(origin, dir, v0, v1, v2, closest_hit.t, tri_index);
                if (hit.result && hit.t < closest_hit.t)
                    closest_hit = hit;
            }
        }
        else
        {
            if (hit_right)
                stack[stack_size++] = node.right_child;

            if (hit_left)
                stack[stack_size++] = node.child;
        }
    }

    return closest_hit;
}

//-------------------------------------------------
// Kernel and main

__global__ void ray_kernel(Mesh *mesh, BVH *bvh)
{
    float3 orig1 = make_float3(5.0f, 0.5f, 0.5f);
    float3 orig2 = make_float3(5.0f, 1.5f, 1.5f);
    float3 dir = make_float3(-1.0f, 0.0f, 0.0f);

    mesh_query_ray_t q1 = mesh_query_ray(mesh, bvh, orig1, dir, 1.0e6f);
    mesh_query_ray_t q2 = mesh_query_ray(mesh, bvh, orig2, dir, 1.0e6f);

    printf("Hit1: %d t=%.4f face=%d\n", q1.result, q1.t, q1.face);
    printf("Hit2: %d t=%.4f face=%d\n", q2.result, q2.t, q2.face);
}

int main()
{
    // Simple square mesh: 2 triangles
    float3 h_points[] = {
        make_float3(0, 0, 0), make_float3(0, 0, 1), make_float3(0, 1, 1), make_float3(0, 1, 0)};
    int h_indices[] = {0, 2, 1, 0, 3, 2};

    Mesh h_mesh;
    h_mesh.num_tris = 2;

    // Allocate on device
    Mesh *d_mesh;
    float3 *d_points;
    int *d_indices;

    cudaMalloc(&d_mesh, sizeof(Mesh));
    cudaMalloc(&d_points, sizeof(float3) * 4);
    cudaMalloc(&d_indices, sizeof(int) * 6);

    cudaMemcpy(d_points, h_points, sizeof(float3) * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_indices, h_indices, sizeof(int) * 6, cudaMemcpyHostToDevice);

    h_mesh.points = d_points;
    h_mesh.indices = d_indices;
    cudaMemcpy(d_mesh, &h_mesh, sizeof(Mesh), cudaMemcpyHostToDevice);

    // BVH -----------------------------------------------------
    std::vector<BVHPackedNode> bvh_nodes;
    build_bvh_sah(h_points, h_indices, h_mesh.num_tris, bvh_nodes);

    // Upload nodes to GPU
    BVHPackedNode *d_bvh_nodes;
    cudaMalloc(&d_bvh_nodes, sizeof(BVHPackedNode) * bvh_nodes.size());
    cudaMemcpy(d_bvh_nodes, bvh_nodes.data(), sizeof(BVHPackedNode) * bvh_nodes.size(), cudaMemcpyHostToDevice);

    // Wrap in BVH struct
    BVH h_bvh;
    h_bvh.nodes = d_bvh_nodes;
    h_bvh.num_nodes = static_cast<int>(bvh_nodes.size());

    BVH *d_bvh_struct;
    cudaMalloc(&d_bvh_struct, sizeof(BVH));
    cudaMemcpy(d_bvh_struct, &h_bvh, sizeof(BVH), cudaMemcpyHostToDevice);

    // BVH -----------------------------------------------------

    // ray_kernel<<<num_blocks, threads_per_block>>>(...);
    ray_kernel<<<1, 1>>>(d_mesh, d_bvh_struct);
    cudaDeviceSynchronize();

    // Cleanup
    cudaFree(d_mesh);
    cudaFree(d_points);
    cudaFree(d_indices);

    cudaFree(d_bvh_nodes);
    cudaFree(d_bvh_struct);

    return 0;
}
