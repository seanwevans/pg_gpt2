EXTENSION = pg_llm
MODULE_big = pg_llm

SRCS := $(wildcard src/*.c)
OBJS := $(SRCS:.c=.o)

DATA = \
sql/pg_llm--0.1.0.sql \
sql/llm_block_forward.sql \
sql/llm_backprop.sql

REGRESS = adamw dropout llm_train_e2e llm_long_sequences llm_backprop_layers llm_backprop_tied llm_sampling llm_weight_sharing
REGRESS_OPTS = --dlpath=$(abs_builddir)

PG_CPPFLAGS += -I$(srcdir)/src

PG_CONFIG ?= pg_config
PGXS := $(shell $(PG_CONFIG) --pgxs 2>/dev/null)

ifeq ($(strip $(PGXS)),)
$(error Could not run pg_config via '$(PG_CONFIG)'. Install the PostgreSQL server development package or set PG_CONFIG.)
endif

ifeq (,$(wildcard $(PGXS)))
$(error PostgreSQL PGXS makefile not found at $(PGXS). Install the server development package (e.g. postgresql-server-dev-16) to build this extension.)
endif

include $(PGXS)
