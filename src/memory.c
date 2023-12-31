/*
  Copyright (c) 2022 William Cotton

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in
  all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
  THE SOFTWARE.
*/

#include "memory.h"
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

size_t roundTo8(size_t value) { return (size_t)((int)(value + 7) & ~7); }

void *nm_malloc(nm_t *nm, size_t size) {
  char *ptr = (char *)nm->freePtr;
  nm->freePtr = (void *)(ptr + roundTo8(size));
  if (nm->freePtr > nm->endPtr) {
    printf("Out of memory\n");
    return NULL;
  }
  memset(ptr, 0, size);
  return ptr;
}

void *nm_realloc(nm_t *nm, void *ptr, size_t size) {
  void *newPtr = nm_malloc(nm, size);
  if (newPtr) {
    memmove(newPtr, ptr, size);
  }
  return newPtr;
}

void *nm_calloc(nm_t *nm, size_t num, size_t size) {
  void *ptr = nm_malloc(nm, num * size);
  return ptr;
}

void nm_free(nm_t *nm) {
  munmap(nm->startPtr, nm->size);
  free(nm);
}

size_t nm_size(nm_t *nm) {
  return (size_t)((int *)nm->freePtr - (int *)nm->startPtr);
}

void nm_print(nm_t *nm) {
  printf("\033[31mnm_size(nm): %zu\033[0m\n", nm_size(nm));
}

nm_t *nm_create(size_t size) {
  nm_t *nm = malloc(sizeof(nm_t));
  nm->size = size;
  nm->freePtr = mmap(NULL, size, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  nm->startPtr = nm->freePtr;
  nm->endPtr = (void *)((char *)nm->startPtr + size);

  return nm;
}
