#include "trace_utils.h"

MeshPair get_mesh()
{
    auto *h_points = new float3[4]{
        make_float3(0, 0, 0),
        make_float3(0, 0, 1),
        make_float3(0, 1, 1),
        make_float3(0, 1, 0),
    };

    auto *h_indices = new int[6]{0, 2, 1, 0, 3, 2};

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

    MeshPair mesh;
    mesh.d_mesh = d_mesh;
    mesh.h_mesh = h_mesh;
    mesh.h_points = h_points;
    mesh.h_indices = h_indices;

    return mesh;
}