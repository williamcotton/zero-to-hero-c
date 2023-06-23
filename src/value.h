#ifndef VALUE_H
#define VALUE_H

#define MAX_CHILDREN 1024 * 10

typedef struct Value {
  float data;
  char *label;
  char *operation;
  struct Value *children[MAX_CHILDREN];
  float grad;
} Value;

Value add(Value v1, Value v2);
Value multiply(Value v1, Value v2);
Value tanhv(Value v);
Value initValue(float data, char *label);
void printValue(Value v);

#endif
