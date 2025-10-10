#include "pg_llm.h"
#include "executor/spi.h"
#include "lib/stringinfo.h"
#include <limits.h>
#include <string.h>

PG_FUNCTION_INFO_V1(pg_llm_load_bpe_vocab);
Datum pg_llm_load_bpe_vocab(PG_FUNCTION_ARGS)
{
    text *path_t = PG_GETARG_TEXT_P(0);
    text *model_t = PG_GETARG_TEXT_P(1);
    char *path = text_to_cstring(path_t);
    char *model = text_to_cstring(model_t);

    FILE *f = fopen(path, "r");
    StringInfoData buf;
    char chunk[8192];
    size_t nread;
    const char *query =
        "INSERT INTO llm_bpe_vocab(model,token_id,token,bytes) "
        "SELECT $1, value::INT, key, convert_to(key, 'UTF8') "
        "FROM json_each_text($2::json) ON CONFLICT DO NOTHING;";
    Oid argtypes[2] = {TEXTOID, TEXTOID};
    char nulls[2] = {' ', ' '};
    Datum model_datum;
    Datum vocab_json;

    if (!f)
        ereport(ERROR, (errmsg("cannot open %s", path)));

    initStringInfo(&buf);
    while ((nread = fread(chunk, 1, sizeof(chunk), f)) > 0)
        appendBinaryStringInfo(&buf, chunk, nread);

    if (ferror(f))
    {
        fclose(f);
        pfree(buf.data);
        ereport(ERROR, (errmsg("failed to read %s", path)));
    }

    fclose(f);

    model_datum = CStringGetTextDatum(model);
    vocab_json = CStringGetTextDatum(buf.data);

    pfree(buf.data);
    buf.data = NULL;

    if (SPI_connect() != SPI_OK_CONNECT)
    {
        pfree(path);
        pfree(model);
        ereport(ERROR, (errmsg("SPI_connect failed in pg_llm_load_bpe_vocab")));
    }

    PG_TRY();
    {
        Datum values[2];
        int rc;

        values[0] = model_datum;
        values[1] = vocab_json;

        rc = SPI_execute_with_args(query, 2, argtypes, values, nulls, false, 0);
        if (rc != SPI_OK_INSERT)
            ereport(ERROR,
                    (errmsg("failed to insert BPE vocab rows (SPI code %d)",
                            rc)));
    }
    PG_CATCH();
    {
        SPI_finish();
        pfree(DatumGetPointer(model_datum));
        pfree(DatumGetPointer(vocab_json));
        pfree(path);
        pfree(model);
        PG_RE_THROW();
    }
    PG_END_TRY();

    SPI_finish();

    pfree(DatumGetPointer(model_datum));
    pfree(DatumGetPointer(vocab_json));
    pfree(path);
    pfree(model);
    PG_RETURN_VOID();
}

PG_FUNCTION_INFO_V1(pg_llm_load_bpe_merges);
Datum pg_llm_load_bpe_merges(PG_FUNCTION_ARGS)
{
    text *path_t = PG_GETARG_TEXT_P(0);
    text *model_t = PG_GETARG_TEXT_P(1);
    char *path=text_to_cstring(path_t);
    char *model=text_to_cstring(model_t);

    FILE *f = fopen(path, "r");
    char l[256], r[256];
    int rank = 0;
    const char *query =
        "INSERT INTO llm_bpe_merges(model,rank,left,right,pair)"
        "VALUES($1,$2,$3,$4,$5) ON CONFLICT DO NOTHING;";
    Oid argtypes[5] = {TEXTOID, INT4OID, TEXTOID, TEXTOID, TEXTOID};
    char nulls[5] = {' ', ' ', ' ', ' ', ' '};
    Datum model_datum;

    if (!f)
        ereport(ERROR, (errmsg("cannot open %s", path)));

    if (SPI_connect() != SPI_OK_CONNECT)
    {
        fclose(f);
        pfree(path);
        pfree(model);
        ereport(ERROR, (errmsg("SPI_connect failed in pg_llm_load_bpe_merges")));
    }

    model_datum = CStringGetTextDatum(model);

    PG_TRY();
    {
        while (fscanf(f, "%255s %255s", l, r) == 2)
        {
            char pair[512];
            Datum values[5];
            Datum left_text;
            Datum right_text;
            Datum pair_text;
            int rc;
            int written;

            if (l[0] == '#')
                continue;

            written = snprintf(pair, sizeof(pair), "%s %s", l, r);
            if (written < 0 || written >= (int) sizeof(pair))
                ereport(ERROR,
                        (errmsg("BPE merge pair too long for buffer")));

            left_text = CStringGetTextDatum(l);
            right_text = CStringGetTextDatum(r);
            pair_text = CStringGetTextDatum(pair);

            values[0] = model_datum;
            values[1] = Int32GetDatum((int32) rank);
            values[2] = left_text;
            values[3] = right_text;
            values[4] = pair_text;

            rc = SPI_execute_with_args(query, 5, argtypes, values, nulls, false, 0);
            if (rc != SPI_OK_INSERT)
                ereport(ERROR,
                        (errmsg("failed to insert BPE merge rank %d (SPI code %d)",
                                rank, rc)));

            pfree(DatumGetPointer(left_text));
            pfree(DatumGetPointer(right_text));
            pfree(DatumGetPointer(pair_text));

            if (rank == INT_MAX)
                ereport(ERROR,
                        (errmsg("BPE merge rank overflow")));
            rank++;
        }

        pfree(DatumGetPointer(model_datum));

        fclose(f);
        SPI_finish();
    }
    PG_CATCH();
    {
        fclose(f);
        SPI_finish();
        pfree(DatumGetPointer(model_datum));
        pfree(path);
        pfree(model);
        PG_RE_THROW();
    }
    PG_END_TRY();

    pfree(path);
    pfree(model);
    PG_RETURN_VOID();
}
