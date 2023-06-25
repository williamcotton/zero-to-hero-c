#ifndef TOPOLOGICAL_SORT_H
#define TOPOLOGICAL_SORT_H

#include "hashset.h"
#include "value.h"

typedef struct TopoList {
  Value **values;
  int size;
  int capacity;
} TopoList;

TopoList *topolist_create(int capacity);
void topolist_add(TopoList *list, Value *value);
void topolist_sort(Value *v, TopoList *topo, int *index, HashSet *visited);
void topolist_free(TopoList *list);

#endif
