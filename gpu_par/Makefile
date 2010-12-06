all:
	gcc -Wall common/main.c common/parser.c common/debug.c -c -std=gnu99
	nvcc quicksort.cu radixsort.cu main.o parser.o debug.o -o main
	rm -f *.o
ready:
	export PATH=/usr/local/cuda/bin/:$PATH
	export LD_LIBRARY_PATH=/usr/local/cuda/lib64/:$LD_LIBRARY_PATH
clean:
	rm -f *~
	rm -f *.o
	rm -f main
