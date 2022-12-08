all: part1 part2 part3 part4

part1:
	g++ matrixMul_cpu.cpp -o matrixMul_cpu.exe
	nvcc matrixMul_gpu.cu -o matrixMul_gpu.exe
part2:
	nvcc part2_matrixMul_gpu.cu -o part2_matrixMul_gpu.exe
part3:
	nvcc part3_matrixMul_gpu.cu -o part3_matrixMul_gpu.exe
part4:
	nvcc part4_matrixMul_gpu.cu -o part4_matrixMul_gpu.exe



