#include "hashset.h"
#include "value.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int hash_value(Value *value) {
  int hash = 0;
  for (size_t i = 0; i < strlen(value->label); i++) {
    hash = (hash * 31 + value->label[i]) % HASHSET_SIZE;
  }
  return hash;
}

HashSet *hashset_create() {
  HashSet *set = malloc(sizeof(HashSet));
  for (int i = 0; i < HASHSET_SIZE; i++) {
    set->buckets[i] = NULL;
  }
  return set;
}

void hashset_add(HashSet *set, Value *value) {
  int index = hash_value(value);
  HashSetNode *node = set->buckets[index];
  while (node != NULL) {
    if (node->value == value) {
      return;
    }
    node = node->next;
  }
  node = malloc(sizeof(HashSetNode));
  node->value = value;
  node->next = set->buckets[index];
  set->buckets[index] = node;
}

bool hashset_contains(HashSet *set, Value *value) {
  int index = hash_value(value);
  HashSetNode *node = set->buckets[index];
  while (node != NULL) {
    if (node->value == value) {
      return true;
    }
    node = node->next;
  }
  return false;
}

void hashset_free(HashSet *set) {
  for (int i = 0; i < HASHSET_SIZE; i++) {
    HashSetNode *node = set->buckets[i];
    while (node != NULL) {
      HashSetNode *next = node->next;
      free(node);
      node = next;
    }
  }
  free(set);
}

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
