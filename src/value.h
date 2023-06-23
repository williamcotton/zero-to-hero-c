#ifndef VALUE_H
#define VALUE_H

#define MAX_CHILDREN 1024 * 10

typedef struct Value {
  float data;
  char *label;
  char *operation;
  struct Value *children[MAX_CHILDREN];
  float grad;
  // a function called backward that takes in 3 values by reference
  void (*backward)(struct Value *self, struct Value *other, struct Value *out);
} Value;

Value add(Value v1, Value v2);
Value multiply(Value v1, Value v2);
Value tanhv(Value v);
Value initValue(float data, char *label);
void printValue(Value v);

#endif
