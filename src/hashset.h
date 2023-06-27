#ifndef HASHSET_H
#define HASHSET_H

#include "value.h"
#include <stdbool.h>

#define HASHSET_SIZE 1024

typedef struct HashSetNode {
  Value *value;
  struct HashSetNode *next;
} HashSetNode;

typedef struct HashSet {
  HashSetNode *buckets[HASHSET_SIZE];
  nm_t *nm;
} HashSet;

HashSet *hashset_create(nm_t *nm);
void hashset_add(HashSet *set, Value *value);
bool hashset_contains(HashSet *set, Value *value);

#endif
