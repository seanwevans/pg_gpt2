EXTENSION = pg_llm
MODULE_big = pg_llm

SRCS := $(wildcard src/*.c)
OBJS := $(SRCS:.c=.o)

DATA = \
sql/pg_llm--0.1.0.sql \
sql/llm_block_forward.sql \
sql/llm_backprop.sql

REGRESS = adamw

PG_CPPFLAGS += -I$(srcdir)/src

PG_CONFIG = pg_config
PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)
