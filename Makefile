CC = clang
CFLAGS = -Wall -Wextra -pedantic -std=gnu2x
DEV_CFLAGS += -Werror -fsanitize=address,undefined,implicit-conversion,float-divide-by-zero,local-bounds,nullability,integer,function,return,signed-integer-overflow,unsigned-integer-overflow -fno-omit-frame-pointer -g -O0
LDFLAGS = -L./lib -lgnuplot_i -L./src
SRCDIR = src
OBJDIR = build
LIBDIR = lib
APP = main
SRCS = $(filter-out $(SRCDIR)/$(APP).c,$(wildcard $(SRCDIR)/*.c))
OBJS = $(patsubst $(SRCDIR)/%.c,$(OBJDIR)/%.o,$(SRCS))
LIBS = $(LIBDIR)/libgnuplot_i.a

all: $(OBJDIR)/$(APP)

$(OBJDIR)/$(APP): $(OBJS) $(LIBS)
	mkdir -p $(OBJDIR)
	$(CC) $(CFLAGS) -o $@ $(SRCDIR)/$(APP).c $(OBJS) $(LDFLAGS) $(DEV_CFLAGS)

$(OBJDIR)/%.o: $(SRCDIR)/%.c $(SRCDIR)/%.h
	$(CC) $(CFLAGS) -c $< -o $@

$(LIBDIR)/libgnuplot_i.a: $(OBJDIR)/gnuplot_i.o
	ar rcs $@ $<

$(OBJDIR)/gnuplot_i.o: $(LIBDIR)/gnuplot_i.c $(LIBDIR)/gnuplot_i.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJDIR)/$(APP) $(OBJS)

leaks:
	leaks --atExit -- ./$(OBJDIR)/$(APP)

run: $(OBJDIR)/$(APP)
	export DISPLAY=:0.0 && ./$(OBJDIR)/$(APP)