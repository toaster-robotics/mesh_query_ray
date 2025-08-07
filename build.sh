#!/bin/bash
clear
set -e

rm -f raytrace 

nvcc -std=c++17 -O2 -o raytrace mesh_query_ray.cu bvh.cpp trace_utils.cpp
./raytrace