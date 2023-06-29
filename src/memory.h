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

#ifndef MEMORY_MANAGER_H
#define MEMORY_MANAGER_H

#include <stdint.h>
#include <stdlib.h>

typedef struct memory_manager_malloc_t {
  void *ptr;
} memory_manager_malloc_t;

typedef struct nm_t {
  void *freePtr;
  void *startPtr;
  void *endPtr;
  size_t size;
} nm_t;

void *nm_grow(nm_t *nm, size_t size);
void *nm_malloc(nm_t *nm, size_t size);
void *nm_realloc(nm_t *nm, void *ptr, size_t size);
void *nm_calloc(nm_t *nm, size_t num, size_t size);
void nm_free(nm_t *nm);
size_t nm_size(nm_t *nm);
void nm_print(nm_t *nm);
nm_t *nm_create(size_t size);

#endif // MEMORY_MANAGER_H
