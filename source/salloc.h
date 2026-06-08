#ifndef SALLOC_H
#define SALLOC_H

#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include "salloc.h"

/* these three functions serve to do 
 * failed allocations check for me, so that 
 * in other areas of my code I can allocate 
 * memory without checks embedded into the
 * code every time */

/* smalloc is a "safe" malloc, that works by 
 * using malloc with the given size, and crashing
 * with malloc perror if the allocation fails,
 * saves a few lines of code here and there */
void *smalloc(size_t size){
    void *ret = malloc(size);
    if(!ret){
        perror("malloc");
        exit(EXIT_FAILURE);
    }
    return ret; 
}
/* same principle as safe malloc */
void *scalloc(size_t nmemb, size_t size){
    void *ret = calloc(nmemb, size);
    if(!ret){
        perror("calloc");
        exit(EXIT_FAILURE);
    }
    return ret; 
}

/* reallocates memory, checks for fail, returns pointer */
void *srealloc(void *ptr, size_t size){
    void *ret = realloc(ptr, size);
    if(!ret){
        perror("realloc");
        exit(EXIT_FAILURE);
    }
    return ret;
}
#endif
