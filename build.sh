#!/bin/bash
rm raytrace 
nvcc -std=c++14 -o raytrace mesh_query_ray.cu
./raytrace