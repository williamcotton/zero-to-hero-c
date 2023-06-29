CC = clang
TIDY = clang-tidy
CFLAGS = $(shell cat compile_flags.txt | tr '\n' ' ')
DEV_CFLAGS += -fsanitize=address
LDFLAGS = -L./lib -lgnuplot_i -L./src
SRCDIR = src
OBJDIR = build
LIBDIR = lib
APP = main
SRCS = $(filter-out $(SRCDIR)/$(APP).c,$(wildcard $(SRCDIR)/*.c))
OBJS = $(patsubst $(SRCDIR)/%.c,$(OBJDIR)/%.o,$(SRCS))
LIBS = $(LIBDIR)/libgnuplot_i.a

APP_REQUIREMENTS = $(OBJS) $(LIBS) $(SRCDIR)/$(APP).c

all: $(OBJDIR)/$(APP)

$(OBJDIR)/$(APP): $(APP_REQUIREMENTS)
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

leaks: $(OBJDIR)/$(APP)
	leaks --atExit -- ./$(OBJDIR)/$(APP)

lint:
	$(TIDY) -warnings-as-errors=* $(SRCDIR)/$(APP).c

analyze:
	$(CC) --analyze $(SRCS) $(CFLAGS)

leaks-trace: trace
	leaks --atExit -- ./$(OBJDIR)/$(APP)-trace

$(OBJDIR)/$(APP)-trace: $(APP_REQUIREMENTS)
	mkdir -p $(OBJDIR)
	$(CC) $(CFLAGS) -o $@ $(SRCDIR)/$(APP).c $(OBJS) $(LDFLAGS)

trace: $(OBJDIR)/$(APP)-trace
	codesign -s - -v -f --entitlements debug.plist $(OBJDIR)/$(APP)-trace

run: $(OBJDIR)/$(APP)
	export DISPLAY=:0.0 && ./$(OBJDIR)/$(APP)
