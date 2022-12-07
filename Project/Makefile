all: part1cpu part1gpu part2gpu part3gpu part4gpu

part1cpu:
	g++ matrixMul_cpu.cpp -o matrixMul_cpu.exe
part1gpu:
	nvcc matrixMul_gpu.cu -o matrixMul_gpu.exe
part2gpu:
	nvcc matrixMul_gpu_part2.cu -o matrixMul_gpu_part2.exe
part3gpu:
	nvcc matrixMul_gpu_part3.cu -o matrixMul_gpu_part3.exe
part4gpu:
	nvcc matrixMul_gpu_part4.cu -o matrixMul_gpu_part4.exe



