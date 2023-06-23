#ifndef PLOT_H
#define PLOT_H

typedef struct range {
  float start;
  float end;
  float step;
} range;

void generate_data(double *xs, double *ys, int num_points, float (*func)(float),
                   float start, float end, float step);

void ys_equal_f_of_xs(double *ys, float (*func)(float), double *xs,
                      int num_points, range r);

void plot_data(double *xs, double *ys, int num_points, char *title);

#endif
