// mesh_query_ray.cu (minimal CUDA-only version)
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cmath>
#include "bvh.h"
#include "trace_utils.h"

__device__ __forceinline__ float3 f3(const float4 &v)
{
    return make_float3(v.x, v.y, v.z);
}

__device__ inline bool intersect_ray_aabb_safe(const float3 &o, const float3 &d, const float3 &lower, const float3 &upper, float &tmin_out, float &tmax_out)
{
    float tmin = 0.0f;
    float tmax = tmax_out; // caller sets current far

    // X/Y/Z axes
    for (int ax = 0; ax < 3; ++ax)
    {
        float ro = (&o.x)[ax];
        float rd = (&d.x)[ax];
        float lo = (&lower.x)[ax];
        float hi = (&upper.x)[ax];

        if (fabsf(rd) < 1e-20f)
        {
            // Ray parallel to slab: must be inside the slab
            if (ro < lo || ro > hi)
                return false;
            // else no constraint from this axis
        }
        else
        {
            float invd = 1.0f / rd;
            float t0 = (lo - ro) * invd;
            float t1 = (hi - ro) * invd;
            if (t0 > t1)
            {
                float tmp = t0;
                t0 = t1;
                t1 = tmp;
            }
            tmin = fmaxf(tmin, t0);
            tmax = fminf(tmax, t1);
            if (tmin > tmax)
                return false;
        }
    }

    tmin_out = tmin;
    tmax_out = tmax;
    return true;
}

__device__ inline bool intersect_ray_aabb(const float3 &origin, const float3 &inv_dir, const float3 &lower, const float3 &upper, float &tmin, float &tmax)
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
__device__ mesh_query_ray_t intersect_ray_triangle(const float3 &orig, const float3 &dir, const float3 &v0, const float3 &v1, const float3 &v2, float max_t, int face_index)
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
// mesh_query_ray: BVH
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

        // Leaf first: don't touch child AABBs at all
        if (node.flags == 1)
        {
            int start = node.prim_index;
            int count = node.prim_count;
            for (int i = 0; i < count; ++i)
            {
                int tri_index = start + i;
                uint3 tri = mesh->indices[tri_index];
                float3 v0 = mesh->vertices[tri.x];
                float3 v1 = mesh->vertices[tri.y];
                float3 v2 = mesh->vertices[tri.z];

                mesh_query_ray_t hit = intersect_ray_triangle(origin, dir, v0, v1, v2, closest_hit.t, tri_index);
                if (hit.result && hit.t < closest_hit.t)
                    closest_hit = hit;
            }
            continue;
        }

        float tmin_left = 0.0f, tmax_left = closest_hit.t;
        float tmin_right = 0.0f, tmax_right = closest_hit.t;

        // Convert float4 bounds stored in the BVH to float3 for the AABB test
        float3 left_lower = f3(node.left.lower);
        float3 left_upper = f3(node.left.upper);
        float3 right_lower = f3(node.right.lower);
        float3 right_upper = f3(node.right.upper);

        bool hit_left = intersect_ray_aabb(origin, inv_dir, left_lower, left_upper, tmin_left, tmax_left);
        bool hit_right = intersect_ray_aabb(origin, inv_dir, right_lower, right_upper, tmin_right, tmax_right);

        // (optional) push nearer first
        if (hit_left && hit_right)
        {
            // choose order by entry t
            if (tmin_right < tmin_left)
            {
                stack[stack_size++] = node.child;
                stack[stack_size++] = node.right_child;
            }
            else
            {
                stack[stack_size++] = node.right_child;
                stack[stack_size++] = node.child;
            }
        }
        else
        {
            if (hit_left)
                stack[stack_size++] = node.child;
            if (hit_right)
                stack[stack_size++] = node.right_child;
        }
    }

    return closest_hit;
}

//-------------------------------------------------
// mesh_query_ray: brute force
__device__ mesh_query_ray_t brute_mesh_query_ray(Mesh *mesh, const float3 &origin, const float3 &dir, float max_t)
{
    mesh_query_ray_t brute = {};
    brute.t = 1.0e6f;

    for (int tri_index = 0; tri_index < mesh->num_tris; ++tri_index)
    {
        uint3 tri = mesh->indices[tri_index];
        float3 v0 = mesh->vertices[tri.x];
        float3 v1 = mesh->vertices[tri.y];
        float3 v2 = mesh->vertices[tri.z];
        mesh_query_ray_t hit = intersect_ray_triangle(origin, dir, v0, v1, v2, brute.t, tri_index);
        if (hit.result && hit.t < brute.t)
            brute = hit;
    }
    return brute;
}

//-------------------------------------------------
// Kernel and main

__global__ void ray_kernel(Mesh *mesh, BVH *bvh)
{
    float d = 1.0f;
    d -= 0.000001f;
    // float3 orig1 = make_float3(5.0f, 0.5f, 0.5f);
    // float3 orig2 = make_float3(5.0f, d, d);
    // float3 orig1 = make_float3(5.0f, 0.5f, 0.5f);
    // float3 orig1 = make_float3(2.232051, 0.031250, 0.531250);
    float3 orig1 = make_float3(2.232051, 0.0, d);

    float3 dir = make_float3(-1.0f, 0.0f, 0.0f);

    mesh_query_ray_t brute = brute_mesh_query_ray(mesh, orig1, dir, 1.0e6f);
    mesh_query_ray_t q1 = mesh_query_ray(mesh, bvh, orig1, dir, 1.0e6f);
    // mesh_query_ray_t q2 = mesh_query_ray(mesh, bvh, orig2, dir, 1.0e6f);

    printf("BVH   Hit1: %d t=%.4f face=%d\n", q1.result, q1.t, q1.face);
    printf("BRUTE Hit1: %d t=%.4f face=%d\n", brute.result, brute.t, brute.face);
    // printf("Hit2: %d t=%.4f face=%d\n", q2.result, q2.t, q2.face);
}

int main()
{
    MeshContainer mesh = get_mesh();
    BVHContainer bvh = get_bvh(mesh);

    // printf("%f %f %f %f\n", mesh.center.x, mesh.center.y, mesh.center.z, mesh.radius);

    // BVH -----------------------------------------------------

    // ray_kernel<<<num_blocks, threads_per_block>>>(...);
    ray_kernel<<<1, 1>>>(mesh.device, bvh.device_bvh);
    cudaDeviceSynchronize();

    // Cleanup
    cudaFree(mesh.device);
    cudaFree(mesh.host.vertices);
    cudaFree(mesh.host.indices);

    cudaFree(bvh.device_nodes);
    cudaFree(bvh.device_bvh);

    return 0;
}
