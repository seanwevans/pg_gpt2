# Developer Guide: Running Regression Tests in Docker

This guide describes how to run the full PostgreSQL regression suite for the
`pg_llm` extension by executing `make installcheck` inside a containerized
PostgreSQL instance. It is aimed at developers who want a reproducible
environment with identical dependencies to the production Docker image while
still exercising every regression test in the `REGRESS` list defined in the
project `Makefile`.

## Overview

`make installcheck` uses PostgreSQL's `pg_regress` harness to spin up a
temporary server, load the extension, and run the SQL scripts in `sql/` against
their corresponding `expected/` outputs. Running the target inside a container
ensures that:

- The same PostgreSQL major version and shared library dependencies are used as
  in production builds.
- Tests do not interfere with any PostgreSQL instance running on the host.
- The environment is self-contained and easily reproducible for CI pipelines.

The steps below reuse the repository's multi-stage `Dockerfile`. We build an
image from the `builder` stage (which already has the compiler toolchain and
`postgresql-server-dev` packages installed), mount the project directory into
that container, and execute the regression suite as the `postgres` superuser.

## 1. Build the development image

First build the container image that contains the build toolchain. The
`--target builder` flag reuses the first stage of the repository `Dockerfile` so
no additional Dockerfile is required.

```bash
docker build --target builder -t pg-gpt2-dev .
```

## 2. Start a disposable container for testing

Launch an interactive container from the image. Mount the repository into the
container at `/workspace` so changes on the host are visible inside the
container. The official PostgreSQL image defaults to the `postgres` user, so we
switch to it after adjusting permissions.

```bash
docker run --rm -it \
  --name pg-gpt2-installcheck \
  -v "$(pwd)":/workspace \
  -w /workspace \
  pg-gpt2-dev \
  bash

# Inside the container shell
chown -R postgres:postgres /workspace
su - postgres
cd /workspace
```

If you prefer to keep the container running for repeated test runs, omit the
`--rm` flag and manually remove it later with `docker rm pg-gpt2-installcheck`.

## 3. Build and install the extension artifacts

Within the container shell (now running as the `postgres` user), build and
install the extension to the container's PostgreSQL installation. The
`DESTDIR` step is unnecessary because the container already provides an isolated
filesystem.

```bash
make clean
make
make install
```

`make install` copies the shared library and SQL files into the PostgreSQL
extension directory reported by `pg_config --pkglibdir`. This is required before
`make installcheck` can load the extension during regression tests.

## 4. Run `make installcheck`

After installation completes, invoke the regression target. The PGXS build
system automatically starts a temporary PostgreSQL cluster in the container,
loads `pg_llm`, and runs every script listed in the `REGRESS` block of the
project `Makefile`.

```bash
make installcheck
```

On success you should see output similar to:

```
=======================
 All 9 tests passed.
=======================
```

Detailed logs are stored under `./regression.diffs` and `./regression.out`. If
any test fails, inspect those files to compare the expected and actual SQL
results.

## 5. Cleaning up

When you are finished, exit the container shell. If you created a persistent
container (without `--rm`), stop and remove it manually:

```bash
exit    # leave the postgres shell
exit    # leave the root shell

# On the host
docker stop pg-gpt2-installcheck
docker rm pg-gpt2-installcheck
```

Because `make installcheck` initializes a temporary data directory under the
project tree, run `make clean` on the host if you want to remove the `tmp_check`
folder and object files generated during the build.

## Troubleshooting

- **Port conflicts:** `pg_regress` chooses a free port automatically inside the
  container, so conflicts with host services are unlikely. If you see port
  binding errors, rerun the container with `--network host` disabled (the
  default) to ensure full isolation.
- **Stale build artifacts:** If you switch PostgreSQL versions or rebuild the
  image, run `make clean` before `make` to avoid linking against old objects.
- **Inspecting the temp cluster:** The temporary data directory lives under
  `./tmp_check/`. You can explore its contents after a run to debug failures,
  but remember that it is deleted on the next successful `make clean`.

Following the steps above yields a deterministic Docker workflow for executing
`make installcheck`, providing full regression coverage for the `pg_llm`
extension without polluting the host environment.
