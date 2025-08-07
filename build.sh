#!/bin/bash

set -e

rm -f raytrace 

nvcc -std=c++17 -O2 -o raytrace mesh_query_ray.cu sah_bvh_builder.cpp
./raytrace