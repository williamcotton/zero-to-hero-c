#ifndef VALUE_H
#define VALUE_H

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

Value *value_create(double data, char *label);
Value *value_add(Value *v1, Value *v2);
Value *value_subtract(Value *v1, Value *v2);
Value *value_multiply(Value *v1, Value *v2);
Value *value_divide(Value *v1, Value *v2);
Value *value_expv(Value *v);
Value *value_tanhv(Value *v);
Value *value_power(Value *v1, double v2);
void value_print(Value *v, int depth);
void value_backpropagate_graph(Value *output);
void value_free_graph(Value *output);
void value_free(Value *v);
void value_list_free(ValueList *list);
ValueList *value_list_append(ValueList *list, Value *value);

#endif
