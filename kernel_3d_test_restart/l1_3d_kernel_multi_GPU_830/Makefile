CUDIR = /usr/local/cuda-7.0
CULIB = ${CUDIR}/lib64 -lcudart
CUINC = ${CUDIR}/include
CUFLAG = -gencode arch=compute_35,code=sm_35 -Xptxas=-v -fmad=false -use_fast_math #-maxrregcount=107
CFLAG = -fopenmp -O3

CC = g++
CU = nvcc

all: a.out

a.out:	rtmlib-modC.o rtmlib-modC-gpu-multi.o
	${CC} ${CFLAG} rtmlib-modC.o rtmlib-modC-gpu-multi.o -L${CULIB} -o a.out  



rtmlib-modC.o:	rtmlib-modC.c
	${CC} ${CFLAG} -o rtmlib-modC.o -c rtmlib-modC.c



rtmlib-modC-gpu-multi.o: rtmlib-modC-gpu-multi.cu
	${CU} -c rtmlib-modC-gpu-multi.o ${CUFLAG} rtmlib-modC-gpu-multi.cu -I${CUINC}

run:	
	./a.out

clean:
	rm -f *.o
	rm *.out
