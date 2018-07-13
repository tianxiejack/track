################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/BackTrk.cpp \
../src/CKalmanTrk.cpp \
../src/ConnectRegion.cpp \
../src/DFT.cpp \
../src/DFTInit.cpp \
../src/DFTMat.cpp \
../src/DFTTrack.cpp \
../src/Enhanc.cpp \
../src/GaussianFilter.cpp \
../src/HistTrack.cpp \
../src/HogFeat.cpp \
../src/Kalman.cpp \
../src/MOG3.cpp \
../src/MVDetector.cpp \
../src/MatMem.cpp \
../src/OptDFTSize.cpp \
../src/RectMat.cpp \
../src/SaliencyProcRS.cpp \
../src/SalientSR.cpp \
../src/SceneMV.cpp \
../src/TemplateMatch.cpp \
../src/TemplateMatch2.cpp \
../src/Trajectory.cpp \
../src/UtcTrack.cpp \
../src/malloc_align.cpp 

OBJS += \
./src/BackTrk.o \
./src/CKalmanTrk.o \
./src/ConnectRegion.o \
./src/DFT.o \
./src/DFTInit.o \
./src/DFTMat.o \
./src/DFTTrack.o \
./src/Enhanc.o \
./src/GaussianFilter.o \
./src/HistTrack.o \
./src/HogFeat.o \
./src/Kalman.o \
./src/MOG3.o \
./src/MVDetector.o \
./src/MatMem.o \
./src/OptDFTSize.o \
./src/RectMat.o \
./src/SaliencyProcRS.o \
./src/SalientSR.o \
./src/SceneMV.o \
./src/TemplateMatch.o \
./src/TemplateMatch2.o \
./src/Trajectory.o \
./src/UtcTrack.o \
./src/malloc_align.o 

CPP_DEPS += \
./src/BackTrk.d \
./src/CKalmanTrk.d \
./src/ConnectRegion.d \
./src/DFT.d \
./src/DFTInit.d \
./src/DFTMat.d \
./src/DFTTrack.d \
./src/Enhanc.d \
./src/GaussianFilter.d \
./src/HistTrack.d \
./src/HogFeat.d \
./src/Kalman.d \
./src/MOG3.d \
./src/MVDetector.d \
./src/MatMem.d \
./src/OptDFTSize.d \
./src/RectMat.d \
./src/SaliencyProcRS.d \
./src/SalientSR.d \
./src/SceneMV.d \
./src/TemplateMatch.d \
./src/TemplateMatch2.d \
./src/Trajectory.d \
./src/UtcTrack.d \
./src/malloc_align.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-8.0/bin/nvcc -I/home/nvidia/TrackerPC/include -I/home/nvidia/TrackerPC/OSA_CAP/inc -I/usr/include/opencv -O3 -Xcompiler -fPIC -Xcompiler -fopenmp -ccbin aarch64-linux-gnu-g++ -gencode arch=compute_53,code=sm_53 -m64 -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-8.0/bin/nvcc -I/home/nvidia/TrackerPC/include -I/home/nvidia/TrackerPC/OSA_CAP/inc -I/usr/include/opencv -O3 -Xcompiler -fPIC -Xcompiler -fopenmp --compile -m64 -ccbin aarch64-linux-gnu-g++  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


