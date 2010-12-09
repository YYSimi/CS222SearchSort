all:
	nvcc common/main.cu common/parser.cu common/debug.cu quicksort.cu -c -arch sm_11
	nvcc quicksort.o main.o parser.o debug.o -o main
	rm -f *.o
clean:
	rm -f *~
	rm -f *.o
	rm -f main
	rm -f common/*~
	rm -f common/*.o
