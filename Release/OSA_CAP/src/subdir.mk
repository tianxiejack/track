################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../OSA_CAP/src/osa.cpp \
../OSA_CAP/src/osa_buf.cpp \
../OSA_CAP/src/osa_eth_client.cpp \
../OSA_CAP/src/osa_eth_server.cpp \
../OSA_CAP/src/osa_event.cpp \
../OSA_CAP/src/osa_file.cpp \
../OSA_CAP/src/osa_i2c.cpp \
../OSA_CAP/src/osa_mbx.cpp \
../OSA_CAP/src/osa_msgq.cpp \
../OSA_CAP/src/osa_mutex.cpp \
../OSA_CAP/src/osa_pipe.cpp \
../OSA_CAP/src/osa_prf.cpp \
../OSA_CAP/src/osa_que.cpp \
../OSA_CAP/src/osa_rng.cpp \
../OSA_CAP/src/osa_sem.cpp \
../OSA_CAP/src/osa_thr.cpp \
../OSA_CAP/src/osa_tsk.cpp 

OBJS += \
./OSA_CAP/src/osa.o \
./OSA_CAP/src/osa_buf.o \
./OSA_CAP/src/osa_eth_client.o \
./OSA_CAP/src/osa_eth_server.o \
./OSA_CAP/src/osa_event.o \
./OSA_CAP/src/osa_file.o \
./OSA_CAP/src/osa_i2c.o \
./OSA_CAP/src/osa_mbx.o \
./OSA_CAP/src/osa_msgq.o \
./OSA_CAP/src/osa_mutex.o \
./OSA_CAP/src/osa_pipe.o \
./OSA_CAP/src/osa_prf.o \
./OSA_CAP/src/osa_que.o \
./OSA_CAP/src/osa_rng.o \
./OSA_CAP/src/osa_sem.o \
./OSA_CAP/src/osa_thr.o \
./OSA_CAP/src/osa_tsk.o 

CPP_DEPS += \
./OSA_CAP/src/osa.d \
./OSA_CAP/src/osa_buf.d \
./OSA_CAP/src/osa_eth_client.d \
./OSA_CAP/src/osa_eth_server.d \
./OSA_CAP/src/osa_event.d \
./OSA_CAP/src/osa_file.d \
./OSA_CAP/src/osa_i2c.d \
./OSA_CAP/src/osa_mbx.d \
./OSA_CAP/src/osa_msgq.d \
./OSA_CAP/src/osa_mutex.d \
./OSA_CAP/src/osa_pipe.d \
./OSA_CAP/src/osa_prf.d \
./OSA_CAP/src/osa_que.d \
./OSA_CAP/src/osa_rng.d \
./OSA_CAP/src/osa_sem.d \
./OSA_CAP/src/osa_thr.d \
./OSA_CAP/src/osa_tsk.d 


# Each subdirectory must supply rules for building sources it contributes
OSA_CAP/src/%.o: ../OSA_CAP/src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-8.0/bin/nvcc -I/home/nvidia/TrackerPC/include -I/home/nvidia/TrackerPC/OSA_CAP/inc -I/usr/include/opencv -O3 -Xcompiler -fPIC -Xcompiler -fopenmp -ccbin aarch64-linux-gnu-g++ -gencode arch=compute_53,code=sm_53 -m64 -odir "OSA_CAP/src" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-8.0/bin/nvcc -I/home/nvidia/TrackerPC/include -I/home/nvidia/TrackerPC/OSA_CAP/inc -I/usr/include/opencv -O3 -Xcompiler -fPIC -Xcompiler -fopenmp --compile -m64 -ccbin aarch64-linux-gnu-g++  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


