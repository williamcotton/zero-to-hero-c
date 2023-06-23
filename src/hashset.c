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
