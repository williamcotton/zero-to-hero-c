#include "hashset.h"
#include "value.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int hash_value(Value *value) {
  int hash = 0;
  if (value->uuid == NULL) {
    return hash;
  }
  for (size_t i = 0; i < strlen(value->uuid); i++) {
    hash = (hash * 31 + value->uuid[i]) % HASHSET_SIZE;
  }
  return hash;
}

HashSet *hashset_create(nm_t *nm) {
  HashSet *set = nm_malloc(nm, sizeof(HashSet));
  for (int i = 0; i < HASHSET_SIZE; i++) {
    set->buckets[i] = NULL;
  }
  set->nm = nm;
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
  node = nm_malloc(set->nm, sizeof(HashSetNode));
  if (node == NULL) {
    printf("Failed to allocate memory for HashSetNode\n");
    exit(1);
  }
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
