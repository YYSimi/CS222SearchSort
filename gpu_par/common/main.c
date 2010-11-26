#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "io.h"

int main(int argc, char ** argv){
  if(argc == 1){
    printf("Please specify at least one floating point array to operate upon!\n");
    exit(1);
  }

  int nlists = argc - 1;
  Data * data = malloc(sizeof(Data));
  if(data == NULL){
    fprintf(stderr, "Unable to allocate memory\n");
    exit(1);
  }
  char ** lists;
  lists = malloc(sizeof(char *) * nlists);
  if(lists == NULL){
    fprintf(stderr, "Unable to allocated memory\n");
    exit(1);
  }
  for(int i = 0; i < nlists; i++){
    int len = strlen(argv[i+1]);
    lists[i] = malloc((sizeof(char)*len) + 1);
    if(lists[i] == NULL){
      fprintf(stderr, "Unable to allocate memory\n");
      exit(1);
    }
    strcpy(lists[i], argv[i+1]);
  }
  parser(data, nlists, lists);

  //  print_data(data, 0); //uncomment to check that data was read in properly

  Results * results = malloc(sizeof(Results));
  if(results == NULL){
    fprintf(stderr, "Unable to allocate memory\n");
    exit(1);
  }

  results->ntests = 0;
  results->nlists = data->nlists;

  for(int i = 0; i < data->nlists; i++){
    /* Execute the sorting algorithms here, each one on data->fp_arr[i] */
  }

  free(lists);
  free(data);
  free(results);

  return 0;
}
