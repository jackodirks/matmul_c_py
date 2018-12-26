CC := gcc
SRCDIR :=src/
INC := -I inc/
LIB := -lOpenCL
LIBDIR := -L lib/
LDFLAGS := -shared -fPIC -pthread
CFLAGS =-fPIC -pthread -std=gnu11 $(INC)
ODIR :=obj/
CFILES=$(wildcard $(SRCDIR)*.c)
_OFILES=$(patsubst %.c,%.o,$(CFILES))
OFILES = $(patsubst $(SRCDIR)%,$(ODIR)%,$(_OFILES))
TARGET := libmatmul
PYTHONEXEC := py/main.py

.PHONY: all clean debug

all: CFLAGS += -O3 -Wall -Wextra -Werror -march=native -funroll-loops
all: LDFLAGS += -s
all: $(TARGET)

debug: CFLAGS +=-Og -ggdb3 -march=znver1
debug: LDFLAGS += 
debug: $(TARGET)

run: all
	@$(PYTHONEXEC)	

$(ODIR) :
	@mkdir -p $(ODIR)

$(ODIR)%.o: $(SRCDIR)%.c | $(ODIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(TARGET): $(OFILES)
	$(CC) -o $@ $^ $(LDFLAGS) $(LIB)

clean:
	@rm -rf $(ODIR)
	@rm $(TARGET)

echo:
	@echo $(OFILES)
