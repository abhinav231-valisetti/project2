all: part1 part2 part3 part4

part1:
	g++ matrixMul_cpu.cpp -o matrixMul_cpu.exe
	nvcc matrixMul_gpu.cu -o matrixMul_gpu.exe
part2:
	nvcc matrixMul_gpu_part2.cu -o matrixMul_gpu_part2.exe
part3:
	nvcc matrixMul_gpu_part3.cu -o matrixMul_gpu_part3.exe
part4:
	nvcc matrixMul_gpu_part4.cu -o matrixMul_gpu_part4.exe



