#include "topographical_sort.h"
#include <stdlib.h>

TopoList *topolist_create(int capacity, nm_t *nm) {
  TopoList *list = nm_malloc(nm, sizeof(TopoList));
  list->values = nm_malloc(nm, sizeof(Value *) * (size_t)capacity);
  list->size = 0;
  list->capacity = capacity;
  list->nm = nm;
  return list;
}

void topolist_add(TopoList *list, Value *value) {
  if (list->size == list->capacity) {
    list->capacity *= 2;
    list->values = nm_realloc(list->nm, list->values,
                              sizeof(Value *) * (size_t)list->capacity);
  }
  list->values[list->size] = value;
  list->size++;
}

void topolist_sort(Value *v, TopoList *topo, int *index, HashSet *visited) {
  if (hashset_contains(visited, v)) {
    return;
  }
  hashset_add(visited, v);
  for (int i = 0; i < v->num_children; i++) {
    topolist_sort(v->children[i], topo, index, visited);
  }
  topolist_add(topo, v);
  (*index)++;
}
