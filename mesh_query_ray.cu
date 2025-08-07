// mesh_query_ray.cu (minimal CUDA-only version)

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cmath>

#define BVH_STACK_SIZE 64

__host__ __device__ inline float3 operator+(const float3 &a, const float3 &b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ inline float3 operator-(const float3 &a, const float3 &b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__ inline float3 operator*(const float3 &a, float s)
{
    return make_float3(a.x * s, a.y * s, a.z * s);
}

__host__ __device__ inline float3 cross(const float3 &a, const float3 &b)
{
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x);
}

__host__ __device__ inline float dot(const float3 &a, const float3 &b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

//-------------------------------------------------
// Mesh structs

struct mesh_query_ray_t
{
    bool result;
    float u, v, t;
    int face;
};

struct Mesh
{
    float3 *points;
    int *indices;
    int num_tris;
};

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
    }
    return out;
}

//-------------------------------------------------
// mesh_query_ray: brute-force (no BVH)

__device__ mesh_query_ray_t mesh_query_ray(Mesh *mesh, const float3 &orig, const float3 &dir, float max_t)
{
    mesh_query_ray_t closest_hit = {};
    closest_hit.t = max_t;

    for (int i = 0; i < mesh->num_tris; ++i)
    {
        int i0 = mesh->indices[i * 3 + 0];
        int i1 = mesh->indices[i * 3 + 1];
        int i2 = mesh->indices[i * 3 + 2];

        float3 v0 = mesh->points[i0];
        float3 v1 = mesh->points[i1];
        float3 v2 = mesh->points[i2];

        mesh_query_ray_t hit = intersect_ray_triangle(orig, dir, v0, v1, v2, closest_hit.t, i);

        if (hit.result && hit.t < closest_hit.t)
            closest_hit = hit;
    }

    return closest_hit;
}

//-------------------------------------------------
// Kernel and main

__global__ void ray_kernel(Mesh *mesh)
{
    float3 orig1 = make_float3(5.0f, 0.5f, 0.5f);
    float3 orig2 = make_float3(5.0f, 1.5f, 1.5f);
    float3 dir = make_float3(-1.0f, 0.0f, 0.0f);

    mesh_query_ray_t q1 = mesh_query_ray(mesh, orig1, dir, 1.0e6f);
    mesh_query_ray_t q2 = mesh_query_ray(mesh, orig2, dir, 1.0e6f);

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

    ray_kernel<<<1, 1>>>(d_mesh);
    cudaDeviceSynchronize();

    // Cleanup
    cudaFree(d_mesh);
    cudaFree(d_points);
    cudaFree(d_indices);

    return 0;
}
