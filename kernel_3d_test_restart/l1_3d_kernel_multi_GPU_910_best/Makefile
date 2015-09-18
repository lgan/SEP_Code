CUDIR = /usr/local/cuda-7.0
CULIB = ${CUDIR}/lib64 -lcudart
CUINC = ${CUDIR}/include
CUFLAG = -gencode arch=compute_35,code=sm_35 -Xptxas=-v -fmad=false -use_fast_math #-maxrregcount=107
CFLAG = -fopenmp -O3

SEP=/data/bob/SEP
SEPINC=${SEP}/include
SEPLIB=${SEP}/lib #-lsep3d  -lsep
SEPFLAG=-I${SEPINC} -L${SEPLIB} -L$(SEPLIB) -lsepauxf90 -lsepgeef90 -lsepf90 -lsep2df90 -lsep3df90 -lsep3d -lsepf90 -lsep



CC = g++
CU = nvcc
FLIBS= -L/usr/local/intel_14/composer_xe_2013_sp1.2.144/compiler/lib/intel64 -lifcore

all: a.out

a.out:	rtmlib-modC.o rtmlib-modC-gpu-multi.o
	${CC} ${CFLAG} rtmlib-modC.o rtmlib-modC-gpu-multi.o -L${CULIB} -L${SEPLIB} -o a.out ${SEPFLAG} ${FLIBS}



rtmlib-modC.o:	rtmlib-modC.c
	${CC} ${CFLAG} -o rtmlib-modC.o -c rtmlib-modC.c -I${SEPINC}


rtmlib-modC-gpu-multi.o: rtmlib-modC-gpu-multi.cu
	${CU} -c rtmlib-modC-gpu-multi.o ${CUFLAG} rtmlib-modC-gpu-multi.cu -I${CUINC} -I${SEPINC}


run:
	@echo make run-cpu: run the CPU code only
	@echo make run-gpu: run the GPU code only
	@echo make run-all: run both codes
	

run-cpu:	
	./a.out 0 data_source=Dat/data_source.H data_Vx0=Dat/data_Vx0.H data_Vy0=Dat/data_Vy0.H data_Vz0=Dat/data_Vz0.H  

run-gpu:	
	./a.out 1 g_data_Vx0=Dat/g_data_Vx0.H g_data_Vy0=Dat/g_data_Vy0.H g_data_Vz0=Dat/g_data_Vz0.H  

run-all:	
	./a.out 2 g_data_Vx0=Dat/g_data_Vx0.H g_data_Vy0=Dat/g_data_Vy0.H g_data_Vz0=Dat/g_data_Vz0.H data_Vx0=Dat/data_Vx0.H data_Vy0=Dat/data_Vy0.H data_Vz0=Dat/data_Vz0.H  


add-cpu:
	echo "n1=210 n2=210 n3=210 n4=100" >> Dat/data_Vx0.H
	echo "n1=210 n2=210 n3=210 n4=100" >> Dat/data_Vy0.H
	echo "n1=210 n2=210 n3=210 n4=100" >> Dat/data_Vz0.H


add add-gpu:
	echo "n1=210 n2=210 n3=210 n4=10" >> Dat/g_data_Vx0.H
	echo "n1=210 n2=210 n3=210 n4=10" >> Dat/g_data_Vy0.H
	echo "n1=210 n2=210 n3=210 n4=10" >> Dat/g_data_Vz0.H


image-cpu:
	make add-cpu
	Window < Dat/data_Vx0.H f3=105 n3=1 | Grey gainpanel=a | Tube
	Window < Dat/data_Vy0.H f3=105 n3=1 | Grey gainpanel=a | Tube
	Window < Dat/data_Vz0.H f3=105 n3=1 | Grey gainpanel=a | Tube

image image-gpu:
	make add-gpu
	Window < Dat/g_data_Vx0.H f3=105 n3=1 | Grey gainpanel=a | Tube
	Window < Dat/g_data_Vy0.H f3=105 n3=1 | Grey gainpanel=a | Tube
	Window < Dat/g_data_Vz0.H f3=105 n3=1 | Grey gainpanel=a | Tube

check:
	make add
	Add scale=1,-1 Dat/g_data_Vx0.H Dat/g_data_Vx0_check.H > result-Vx0
	Add scale=1,-1 Dat/g_data_Vy0.H Dat/g_data_Vy0_check.H > result-Vy0
	Add scale=1,-1 Dat/g_data_Vz0.H Dat/g_data_Vz0_check.H > result-Vz0
	@echo Vx0
	Attr < result-Vx0  
	@echo Vy0
	Attr < result-Vy0
	@echo Vz0
	Attr < result-Vz0




clean:
	rm -f *.o
	rm -r *.out


burn:
	rm -f *.o
	rm -r *.out
	rm -r Dat/*.H
