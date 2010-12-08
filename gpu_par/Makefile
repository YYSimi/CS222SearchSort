all:
	gcc -Wall common/main.c common/parser.c common/debug.c -c -std=gnu99
	nvcc quicksort.cu radixsort.cu main.o parser.o debug.o -o main
	rm -f *.o
clean:
	rm -f *~
	rm -f *.o
	rm -f main
