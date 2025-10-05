#include "pg_llm.h"

#include "access/htup_details.h"
#include "catalog/pg_type.h"
#include "commands/trigger.h"
#include "executor/spi.h"
#include "utils/array.h"
#include "utils/hsearch.h"
#include "utils/memutils.h"

typedef struct TensorRegistryEntry
{
    bytea   *tensor;
    int      id;
} TensorRegistryEntry;

static HTAB *tensor_registry = NULL;

PG_FUNCTION_INFO_V1(pg_llm_autograd_map_param);

static void
ensure_tensor_registry(void)
{
    if (tensor_registry == NULL)
    {
        HASHCTL ctl;

        MemSet(&ctl, 0, sizeof(HASHCTL));
        ctl.keysize = sizeof(bytea *);
        ctl.entrysize = sizeof(TensorRegistryEntry);
        ctl.hcxt = TopMemoryContext;
        tensor_registry = hash_create("llm tensor registry",
                                      1024,
                                      &ctl,
                                      HASH_ELEM | HASH_CONTEXT);
    }
}

static ArrayType *
build_shape_array(int ndims, const int *dims)
{
    if (ndims <= 0 || dims == NULL)
        return NULL;

    Datum      *elems = (Datum *) palloc(sizeof(Datum) * ndims);
    int16       elmlen;
    bool        elmbyval;
    char        elmalign;

    get_typlenbyvalalign(INT4OID, &elmlen, &elmbyval, &elmalign);
    for (int i = 0; i < ndims; i++)
        elems[i] = Int32GetDatum(dims[i]);

    return construct_array(elems, ndims, INT4OID, elmlen, elmbyval, elmalign);
}

static int
tape_insert(const char *name, int *inputs, int n_in, int output, const char *extra_json)
{
    StringInfoData buf;

static bool autograd_enabled(void);
static int tape_insert(const char *name, int *inputs, int n_in, int output, const char *extra_json) PG_USED_FOR_ASSERTS_ONLY;

static bool
autograd_enabled(void)
{
    bool enabled = false;
    int ret;
    bool isnull;

    ret = SPI_execute("SELECT flag FROM llm_autograd_mode LIMIT 1", true, 1);
    if (ret != SPI_OK_SELECT)
        ereport(ERROR,
                (errmsg("failed to query llm_autograd_mode (SPI_execute returned %d)", ret)));

    if (SPI_processed > 0)
    {
        Datum datum = SPI_getbinval(SPI_tuptable->vals[0],
                                    SPI_tuptable->tupdesc,
                                    1,
                                    &isnull);
        if (!isnull)
            enabled = DatumGetBool(datum);
    }

    return enabled;
}

static int tape_insert(const char *name, int *inputs, int n_in, int output, const char *extra_json)
{
    StringInfoData buf;

    SPI_connect();

    if (!autograd_enabled())
    {
        SPI_finish();
        return output;
    }

    initStringInfo(&buf);
    appendStringInfo(&buf,
                     "INSERT INTO llm_tape(name,inputs,output,extra) VALUES('%s',ARRAY[",
                     name);
    for (int i = 0; i < n_in; i++)
    {
        if (i > 0)
            appendStringInfoChar(&buf, ',');
        appendStringInfo(&buf, "%d", inputs[i]);
    }
    appendStringInfo(&buf, "],%d,'%s')",
                     output,
                     extra_json ? extra_json : "{}");
    SPI_execute(buf.data, false, 0);
    return output;
}

bool
pg_llm_autograd_enabled(void)
{
    bool enabled = false;

    if (SPI_connect() != SPI_OK_CONNECT)
        ereport(ERROR, (errmsg("SPI_connect failed in autograd_enabled")));

    PG_TRY();
    {
        int ret = SPI_execute("SELECT flag FROM llm_autograd_mode LIMIT 1", true, 0);
        if (ret == SPI_OK_SELECT && SPI_processed > 0)
        {
            bool isnull;
            Datum d = SPI_getbinval(SPI_tuptable->vals[0],
                                    SPI_tuptable->tupdesc,
                                    1,
                                    &isnull);
            if (!isnull)
                enabled = DatumGetBool(d);
        }
    }
    PG_CATCH();
    {
        SPI_finish();
        PG_RE_THROW();
    }
    PG_END_TRY();

    SPI_finish();
    return enabled;
}

static int
insert_tensor_rt(bytea *tensor, int ndims, const int *dims, bool requires_grad)
{
    Datum       values[3];
    bool        nulls[3] = {false, false, false};
    Oid         argtypes[3] = {BYTEAOID, INT4ARRAYOID, BOOLOID};
    int         ret;
    bool        isnull;
    Datum       result;
    ArrayType  *shape = build_shape_array(ndims, dims);

    values[0] = PointerGetDatum(tensor);
    if (shape)
        values[1] = PointerGetDatum(shape);
    else
        nulls[1] = true;
    values[2] = BoolGetDatum(requires_grad);

    ret = SPI_execute_with_args(
        "INSERT INTO llm_tensor_rt(data, shape, requires_grad) "
        "VALUES($1,$2,$3) RETURNING id",
        3,
        argtypes,
        values,
        nulls,
        true,
        1);

    if (ret != SPI_OK_INSERT_RETURNING || SPI_processed != 1)
        ereport(ERROR, (errmsg("failed to insert runtime tensor")));

    result = SPI_getbinval(SPI_tuptable->vals[0],
                           SPI_tuptable->tupdesc,
                           1,
                           &isnull);
    if (isnull)
        ereport(ERROR, (errmsg("runtime tensor id is NULL")));

    return DatumGetInt32(result);
}

int
pg_llm_autograd_track_tensor(bytea *tensor, int ndims, const int *dims, bool requires_grad)
{
    bool                    found;
    TensorRegistryEntry    *entry;
    bytea                  *key = tensor;

    ensure_tensor_registry();
    entry = (TensorRegistryEntry *) hash_search(tensor_registry, &key, HASH_FIND, &found);
    if (found)
        return entry->id;

    entry = (TensorRegistryEntry *) hash_search(tensor_registry, &key, HASH_ENTER, &found);
    entry->tensor = tensor;
    entry->id = insert_tensor_rt(tensor, ndims, dims, requires_grad);
    return entry->id;
}

void
pg_llm_autograd_record_tape(const char *name, int *inputs, int n_inputs, int output, const char *extra_json)
{
    tape_insert(name, inputs, n_inputs, output, extra_json);
}

static void
pg_llm_autograd_map_param_impl(const char *model,
                               const char *name,
                               int token_id,
                               bytea *tensor,
                               int ndims,
                               const int *dims)
{
    Datum       values[4];
    bool        nulls[4] = {false, false, false, false};
    Oid         argtypes[4] = {TEXTOID, TEXTOID, INT4OID, INT4OID};
    int         tensor_id;
    int         ret;

    if (model == NULL || name == NULL)
        ereport(ERROR, (errmsg("model and name must be provided for param mapping")));

    tensor_id = pg_llm_autograd_track_tensor(tensor, ndims, dims, true);

    values[0] = CStringGetTextDatum(model);
    values[1] = CStringGetTextDatum(name);
    values[2] = Int32GetDatum(token_id);
    values[3] = Int32GetDatum(tensor_id);

    ret = SPI_execute_with_args(
        "INSERT INTO llm_tensor_map(model,name,token_id,tensor_id) "
        "VALUES($1,$2,$3,$4) "
        "ON CONFLICT (model,name,token_id) DO UPDATE SET tensor_id=EXCLUDED.tensor_id",
        4,
        argtypes,
        values,
        nulls,
        false,
        0);
    if (ret != SPI_OK_INSERT && ret != SPI_OK_UPDATE)
        ereport(ERROR, (errmsg("failed to upsert tensor map")));
}

Datum
pg_llm_autograd_map_param(PG_FUNCTION_ARGS)
{
    text       *model_text;
    text       *name_text;
    int32       token_id;
    bytea      *tensor;
    ArrayType  *shape = NULL;
    int        *dims = NULL;
    int         ndims = 0;

    if (PG_ARGISNULL(0) || PG_ARGISNULL(1) || PG_ARGISNULL(2) || PG_ARGISNULL(3))
        ereport(ERROR,
                (errmsg("pg_llm_autograd_map_param requires non-null model, name, token_id, and tensor")));

    model_text = PG_GETARG_TEXT_PP(0);
    name_text = PG_GETARG_TEXT_PP(1);
    token_id = PG_GETARG_INT32(2);
    tensor = PG_GETARG_BYTEA_PP(3);

    if (PG_NARGS() > 4 && !PG_ARGISNULL(4))
    {
        Datum  *elems;
        bool   *nulls;
        int     nelems;

        shape = PG_GETARG_ARRAYTYPE_P(4);

        if (ARR_NDIM(shape) > 1)
            ereport(ERROR,
                    (errmsg("pg_llm_autograd_map_param expects shape as a 1-D int array")));

        deconstruct_array(shape,
                          INT4OID,
                          sizeof(int32),
                          true,
                          'i',
                          &elems,
                          &nulls,
                          &nelems);

        dims = (int *) palloc(sizeof(int) * nelems);
        for (int i = 0; i < nelems; i++)
        {
            if (nulls[i])
                ereport(ERROR,
                        (errmsg("pg_llm_autograd_map_param does not accept NULL elements in shape")));
            dims[i] = DatumGetInt32(elems[i]);
        }
        ndims = nelems;
    }

    if (SPI_connect() != SPI_OK_CONNECT)
        ereport(ERROR, (errmsg("SPI_connect failed in pg_llm_autograd_map_param")));

    PG_TRY();
    {
        pg_llm_autograd_map_param_impl(text_to_cstring(model_text),
                                       text_to_cstring(name_text),
                                       token_id,
                                       tensor,
                                       ndims,
                                       dims);
    }
    PG_CATCH();
    {
        SPI_finish();
        PG_RE_THROW();
    }
    PG_END_TRY();

    SPI_finish();

    PG_RETURN_VOID();
}
