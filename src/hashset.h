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
} HashSet;

typedef struct TopoList {
  Value **values;
  int size;
  int capacity;
} TopoList;

HashSet *hashset_create();
void hashset_add(HashSet *set, Value *value);
bool hashset_contains(HashSet *set, Value *value);
void hashset_free(HashSet *set);

TopoList *topolist_create(int capacity);
void topolist_add(TopoList *list, Value *value);
void topolist_free(TopoList *list);
void topographicalSort(Value *v, TopoList *topo, int *index, HashSet *visited);

#endif
