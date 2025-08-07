#include "trace_utils.h"

MeshContainer get_mesh()
{
    // std::vector<float3> host_vertices = {
    //     make_float3(0, 0, 0),
    //     make_float3(0, 0, 1),
    //     make_float3(0, 1, 1),
    //     make_float3(0, 1, 0),
    // };

    // std::vector<uint3> host_indices = {
    //     make_uint3(0, 2, 1),
    //     make_uint3(0, 3, 2),
    // };

    float ds = 1.0f;
    std::vector<float3> host_vertices = {
        // facet 1
        make_float3(0.0f, 0.0f, 0.0f),
        make_float3(ds, 0.0f, ds),
        make_float3(0.0f, 0.0f, ds),
        // facet 2
        make_float3(0.0f, 0.0f, 0.0f),
        make_float3(ds, 0.0f, 0.0f),
        make_float3(ds, 0.0f, ds),
        // facet 3
        make_float3(0.0f, 0.0f, 0.0f),
        make_float3(0.0f, ds, ds),
        make_float3(0.0f, 0.0f, ds),
        // facet 4
        make_float3(0.0f, 0.0f, 0.0f),
        make_float3(0.0f, ds, 0.0f),
        make_float3(0.0f, ds, ds),
    };
    std::vector<uint3> host_indices = {
        make_uint3(0, 1, 2),
        make_uint3(3, 4, 5),
        make_uint3(6, 7, 8),
        make_uint3(9, 10, 11),
    };

    Mesh h_mesh;
    h_mesh.num_tris = static_cast<int>(host_indices.size());

    // Allocate on device
    Mesh *d_mesh;
    float3 *d_points;
    uint3 *d_indices;

    cudaMalloc(&d_mesh, sizeof(Mesh));
    cudaMalloc(&d_points, sizeof(float3) * host_vertices.size());
    cudaMalloc(&d_indices, sizeof(uint3) * host_indices.size());

    cudaMemcpy(d_points, host_vertices.data(), sizeof(float3) * host_vertices.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_indices, host_indices.data(), sizeof(uint3) * host_indices.size(), cudaMemcpyHostToDevice);

    h_mesh.vertices = d_points;
    h_mesh.indices = d_indices;
    cudaMemcpy(d_mesh, &h_mesh, sizeof(Mesh), cudaMemcpyHostToDevice);

    float3 minCorner = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
    float3 maxCorner = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    for (const auto &v : host_vertices)
    {
        minCorner.x = fminf(minCorner.x, v.x);
        minCorner.y = fminf(minCorner.y, v.y);
        minCorner.z = fminf(minCorner.z, v.z);

        maxCorner.x = fmaxf(maxCorner.x, v.x);
        maxCorner.y = fmaxf(maxCorner.y, v.y);
        maxCorner.z = fmaxf(maxCorner.z, v.z);
    }

    MeshContainer mesh;
    mesh.device = d_mesh;
    mesh.host = h_mesh;
    mesh.vertices = std::move(host_vertices);
    mesh.indices = std::move(host_indices);

    mesh.center = (minCorner + maxCorner) * 0.5f;
    mesh.radius = length(maxCorner - minCorner) * 0.5;

    return mesh;
}

BVHContainer get_bvh(const MeshContainer &mesh)
{

    // BVH -----------------------------------------------------
    std::vector<BVHPackedNode> bvh_nodes;
    build_sah_bvh(mesh.vertices.data(), mesh.indices.data(), mesh.host.num_tris, bvh_nodes);

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

MinMaxResult computeMinMaxPoints(const std::vector<float3> &mesh_points, const float3 &r_hat, const float3 &phi_hat, const float3 &theta_hat)
{
    float2 min_point = make_float2(std::numeric_limits<float>::max(),
                                   std::numeric_limits<float>::max());
    float2 max_point = make_float2(std::numeric_limits<float>::lowest(),
                                   std::numeric_limits<float>::lowest());

    float3 u = phi_hat;
    float3 w = -theta_hat;

    for (const auto &pt : mesh_points)
    {
        float px = dot(pt, u); // projection on phi_hat
        float py = dot(pt, w); // projection on -theta_hat

        // update min
        min_point.x = fminf(min_point.x, px);
        min_point.y = fminf(min_point.y, py);

        // update max
        max_point.x = fmaxf(max_point.x, px);
        max_point.y = fmaxf(max_point.y, py);
    }

    return {min_point, max_point};
}