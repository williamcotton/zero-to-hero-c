CC = clang
TIDY = clang-tidy
CFLAGS = $(shell cat compile_flags.txt | tr '\n' ' ')
DEV_CFLAGS += -Werror -fsanitize=address,undefined,implicit-conversion,float-divide-by-zero,local-bounds,nullability,integer,function,return,signed-integer-overflow,unsigned-integer-overflow -fno-omit-frame-pointer -g -O0 -fsanitize-trap
TRACE_CFLAGS += -g3 -O0 
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
	$(CC) --analyze $(SRCS) $(CFLAGS) -Xanalyzer -analyzer-output=text -Xanalyzer -analyzer-checker=core,deadcode,nullability,optin,osx,security,unix,valist -Xanalyzer -analyzer-disable-checker -Xanalyzer security.insecureAPI.DeprecatedOrUnsafeBufferHandling

leaks-trace: trace
	leaks --atExit -- ./$(OBJDIR)/$(APP)-trace

$(OBJDIR)/$(APP)-trace: $(APP_REQUIREMENTS)
	mkdir -p $(OBJDIR)
	$(CC) $(CFLAGS) -o $@ $(SRCDIR)/$(APP).c $(OBJS) $(LDFLAGS) $(TRACE_CFLAGS)

trace: $(OBJDIR)/$(APP)-trace
	codesign -s - -v -f --entitlements debug.plist $(OBJDIR)/$(APP)-trace

run: $(OBJDIR)/$(APP)
	export DISPLAY=:0.0 && ./$(OBJDIR)/$(APP)
