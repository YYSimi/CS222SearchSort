all:
	nvcc common/main.cu common/parser.cu common/debug.cu quicksort.cu radixsort.cu -c -arch sm_11
	nvcc quicksort.o radixsort.o main.o parser.o debug.o -o main
	nvcc heapsort.cu -arch sm_11
	mv a.out heapsort
	rm -f *.o

clean:
	rm -f *~
	rm -f *.o
	rm -f main
	rm -f common/*~
	rm -f common/*.o
