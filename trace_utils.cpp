#include "trace_utils.h"

MeshPair get_mesh()
{
    std::vector<float3> h_points = {
        make_float3(0, 0, 0),
        make_float3(0, 0, 1),
        make_float3(0, 1, 1),
        make_float3(0, 1, 0),
    };

    std::vector<uint3> h_indices = {
        make_uint3(0, 2, 1),
        make_uint3(0, 3, 2),
    };

    Mesh h_mesh;
    h_mesh.num_tris = static_cast<int>(h_indices.size());

    // Allocate on device
    Mesh *d_mesh;
    float3 *d_points;
    uint3 *d_indices;

    cudaMalloc(&d_mesh, sizeof(Mesh));
    cudaMalloc(&d_points, sizeof(float3) * h_points.size());
    cudaMalloc(&d_indices, sizeof(uint3) * h_indices.size());

    cudaMemcpy(d_points, h_points.data(), sizeof(float3) * h_points.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_indices, h_indices.data(), sizeof(uint3) * h_indices.size(), cudaMemcpyHostToDevice);

    h_mesh.points = d_points;
    h_mesh.indices = d_indices;
    cudaMemcpy(d_mesh, &h_mesh, sizeof(Mesh), cudaMemcpyHostToDevice);

    MeshPair mesh;
    mesh.d_mesh = d_mesh;
    mesh.h_mesh = h_mesh;
    mesh.h_points = std::move(h_points);
    mesh.h_indices = std::move(h_indices);

    return mesh;
}

BVHPair bvh_stuff(const MeshPair &mesh)
{

    // BVH -----------------------------------------------------
    std::vector<BVHPackedNode> bvh_nodes;
    build_bvh_sah(mesh.h_points.data(), mesh.h_indices.data(), mesh.h_mesh.num_tris, bvh_nodes);

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

    return {d_bvh_struct, d_bvh_nodes};
}