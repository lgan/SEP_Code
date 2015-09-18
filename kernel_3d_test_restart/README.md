# 3d-elastic-kernel

This is the code I've done in SEP, and is the 3D elastic RTM kernel.

Unlike the 2D one that is the whole program from forward and backward, the 3D one only contains the backward with random 
input and output. 

The purpose of this code is to provide a 3D GPU interfaces for late usage. 


l1_kernel_multi_GPU_910_best:
Better than 910
The latest version before computation on-the-fly


l1_kernel_multi_GPU_910:

Good version that has been sorted and pushed to Bob






l1_kernel_multi_GPU_909:

Finish GPU version of add_source and imaging



l1_kernel_multi_GPU_908:

add_source and mode are both right, which means version 905 is validated. 
I also can draw the circle now using make image




l1_kernel_multi_GPU_905:

have added a source_add function and a model function for constant variables, without knowing the validation of them both. 




l1_kernel_multi_GPU_830:
3D code using multiple GPUs



Performance 

Simply using L1, 200*200*200, ts = 1
CPU compiler: +openmp, +O3

<<<<<<<<<<<<<<<<PERFORMANCE PROFILING>>>>>>>>>>>>>>>>
CPU time for 1 steps:  10.03823566
Computing   Speedup: 89.90
Application Speedup: 11.78
<<<<<<<<<<<<<<<<<PERFORMANCE PROFILING>>>>>>>>>>>>>>>>

