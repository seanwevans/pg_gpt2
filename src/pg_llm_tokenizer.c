#include "pg_llm.h"
#include "executor/spi.h"
#include <string.h>

PG_FUNCTION_INFO_V1(pg_llm_load_bpe_vocab);
Datum pg_llm_load_bpe_vocab(PG_FUNCTION_ARGS)
{
    text *path_t = PG_GETARG_TEXT_P(0);
    text *model_t = PG_GETARG_TEXT_P(1);
    char *path = text_to_cstring(path_t);
    char *model = text_to_cstring(model_t);

    FILE *f = fopen(path, "r");
    char token[512];
    int id;
    const char *query =
        "INSERT INTO llm_bpe_vocab(model,token_id,token,bytes)"
        "VALUES($1,$2,$3,$4) ON CONFLICT DO NOTHING;";
    Oid argtypes[4] = {TEXTOID, INT4OID, TEXTOID, BYTEAOID};
    bool nulls[4] = {false, false, false, false};
    Datum model_datum;

    if (!f)
        ereport(ERROR, (errmsg("cannot open %s", path)));

    SPI_connect();

    model_datum = CStringGetTextDatum(model);

    while (fscanf(f, " \"%[^\"]\" : %d,", token, &id) == 2)
    {
        Datum values[4];
        Datum token_text;
        bytea *token_bytes;
        int rc;

        Size token_len = strlen(token);

        token_text = CStringGetTextDatum(token);
        token_bytes = bytea_alloc(token_len);
        if (token_len > 0)
            memcpy(VARDATA(token_bytes), token, token_len);

        values[0] = model_datum;
        values[1] = Int32GetDatum((int32) id);
        values[2] = token_text;
        values[3] = PointerGetDatum(token_bytes);

        rc = SPI_execute_with_args(query, 4, argtypes, values, nulls, false, 0);
        if (rc != SPI_OK_INSERT)
            ereport(ERROR,
                    (errmsg("failed to insert BPE vocab row for token %d (SPI code %d)",
                            id, rc)));

        pfree(DatumGetPointer(token_text));
        pfree(token_bytes);
    }

    pfree(DatumGetPointer(model_datum));

    fclose(f);
    SPI_finish();
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
    bool nulls[5] = {false, false, false, false, false};
    Datum model_datum;

    if (!f)
        ereport(ERROR, (errmsg("cannot open %s", path)));

    SPI_connect();

    model_datum = CStringGetTextDatum(model);

    while (fscanf(f, "%255s %255s", l, r) == 2)
    {
        char pair[512];
        Datum values[5];
        Datum left_text;
        Datum right_text;
        Datum pair_text;
        int rc;

        if (l[0] == '#')
            continue;

        snprintf(pair, sizeof(pair), "%s %s", l, r);

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

        rank++;
    }

    pfree(DatumGetPointer(model_datum));

    fclose(f);
    SPI_finish();
    pfree(path);
    pfree(model);
    PG_RETURN_VOID();
}
