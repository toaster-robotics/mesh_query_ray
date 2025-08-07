import warp as wp


@wp.kernel
def ray_trace(mesh: wp.uint64):
    ray_origin1 = wp.vec3(5.0, 0.5, 0.5)
    ray_origin2 = wp.vec3(5.0, 1.5, 1.5)
    ray_direction = wp.vec3(-1.0, 0.0, 0.0)

    query1 = wp.mesh_query_ray(mesh, ray_origin1, ray_direction, 1.0e6)
    query2 = wp.mesh_query_ray(mesh, ray_origin2, ray_direction, 1.0e6)

    print(query1.result)
    print(query2.result)


points = [[0.0, 0.0, 0.0,],
          [0.0, 0.0, 1.0,],
          [0.0, 1.0, 1.0,],
          [0.0, 1.0, 0.0,],]
indices = [0, 2, 1, 0, 3, 2]


points = wp.array(points, dtype=wp.vec3)
indices = wp.array(indices, dtype=int)
mesh = wp.Mesh(points=points, indices=indices, velocities=None)

with wp.ScopedDevice('cuda'):
    wp.launch(
        kernel=ray_trace,
        dim=(1,),
        inputs=[mesh.id],

    )
