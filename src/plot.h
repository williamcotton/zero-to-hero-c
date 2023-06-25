#ifndef PLOT_H
#define PLOT_H

typedef struct Range {
  float start;
  float end;
  float step;
} Range;

void generate_data(double *xs, double *ys, int num_points, float (*func)(float),
                   float start, float end, float step);
void plot_gen_ys_f_of_xs(double *ys, float (*func)(float), double *xs,
                         int num_points, Range r);
void plot_data(double *xs, double *ys, int num_points, char *title,
               char *output_path);

#endif
