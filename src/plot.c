#include "plot.h"
#include "../lib/gnuplot_i.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

void generate_data(double *xs, double *ys, int num_points, float (*func)(float),
                   float start, float end, float step) {
  int i;
  float x = start;
  for (i = 0; i < num_points; i++) {
    xs[i] = x;
    ys[i] = func(x);
    x += step;
    if (x > end) {
      break;
    }
  }
}

void ys_equal_f_of_xs(double *ys, float (*func)(float), double *xs,
                      int num_points, range r) {
  generate_data(xs, ys, num_points, func, r.start, r.end, r.step);
}

void plot_data(double *xs, double *ys, int num_points, char *title) {
  gnuplot_ctrl *h;
  h = gnuplot_init();
  gnuplot_setstyle(h, "lines");
  gnuplot_plot_xy(h, xs, ys, num_points, title);
  gnuplot_cmd(h, "set terminal png");
  gnuplot_cmd(h, "set output 'images/plot.png'");
  gnuplot_cmd(h, "replot");
  gnuplot_close(h);
}
