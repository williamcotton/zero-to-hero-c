#include "topographical_sort.h"
#include <stdlib.h>

TopoList *topolist_create(int capacity) {
  TopoList *list = malloc(sizeof(TopoList));
  list->values = malloc(sizeof(Value *) * capacity);
  list->size = 0;
  list->capacity = capacity;
  return list;
}

void topolist_add(TopoList *list, Value *value) {
  if (list->size == list->capacity) {
    list->capacity *= 2;
    list->values = realloc(list->values, sizeof(Value *) * list->capacity);
  }
  list->values[list->size] = value;
  list->size++;
}

void topolist_free(TopoList *list) {
  free(list->values);
  free(list);
}

void topographicalSort(Value *v, TopoList *topo, int *index, HashSet *visited) {
  if (hashset_contains(visited, v)) {
    return;
  }
  hashset_add(visited, v);
  for (int i = 0; i < v->num_children; i++) {
    topographicalSort(v->children[i], topo, index, visited);
  }
  topolist_add(topo, v);
  (*index)++;
}
