#pragma once
#include <cuda_runtime.h>
#include <cmath>

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

__host__ __device__ inline float3 normalize(const float3 &v)
{
    float len_sq = v.x * v.x + v.y * v.y + v.z * v.z;

#ifdef __CUDA_ARCH__
    // Device: use fast hardware intrinsic
    float inv_len = rsqrtf(len_sq + 1e-8f);
#else
    // Host: fall back to standard sqrtf
    float inv_len = 1.0f / sqrtf(len_sq + 1e-8f);
#endif
    return make_float3(v.x * inv_len, v.y * inv_len, v.z * inv_len);
}