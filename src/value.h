#ifndef VALUE_H
#define VALUE_H

#include "memory.h"

#define MAX_CHILDREN 1024 * 10

#define UNUSED __attribute__((unused))

typedef struct Value Value;

typedef struct ValueList {
  Value *value;
  struct ValueList *next;
} ValueList;

struct Value {
  double data;
  char *label;
  char *operation;
  Value **children;
  int num_children;
  double grad;
  void (*backward)(Value *v);
  double v2;
};

typedef struct Vector {
  Value **values;
  int size;
} Vector;

Value *value_create(double data, char *label, nm_t *nm);
Value *value_add(Value *v1, Value *v2, nm_t *nm);
Value *value_subtract(Value *v1, Value *v2, nm_t *nm);
Value *value_multiply(Value *v1, Value *v2, nm_t *nm);
Value *value_divide(Value *v1, Value *v2, nm_t *nm);
Value *value_expv(Value *v, nm_t *nm);
Value *value_tanhv(Value *v, nm_t *nm);
Value *value_power(Value *v1, double v2, nm_t *nm);
void value_print(Value *v, int depth);
void value_backpropagate(Value *output, nm_t *epochNm);
Vector *value_create_vector(double *data, int size, nm_t *nm);
void value_print_nested(Value *v, int depth);

#endif
