#include "pg_llm.h"
#include "access/htup_details.h"
#include "lib/stringinfo.h"
#include "utils/array.h"
#include <zlib.h>
#include <limits.h>

static void write_npz_entry(gzFile fp, const char *name, bytea *data, ArrayType *shape_array);
static void gzwrite_exact(gzFile fp, const void *src, Size nbytes, const char *context);

PG_FUNCTION_INFO_V1(pg_llm_export_npz);

Datum
pg_llm_export_npz(PG_FUNCTION_ARGS)
{
    text   *path_t = PG_GETARG_TEXT_P(0);
    text   *model_t = PG_GETARG_TEXT_P(1);
    char   *path = text_to_cstring(path_t);
    char   *model = text_to_cstring(model_t);
    gzFile  fp = NULL;
    Datum   model_value;
    Oid     argtypes[1];
    Datum   values[1];

    if (SPI_connect() != SPI_OK_CONNECT)
    {
        pfree(path);
        pfree(model);
        ereport(ERROR,
                (errmsg("SPI_connect failed in pg_llm_export_npz")));
    }

    PG_TRY();
    {
        int spi_rc;

        fp = gzopen(path, "wb");
        if (!fp)
            ereport(ERROR,
                    (errmsg("could not open %s", path)));

        argtypes[0] = TEXTOID;
        model_value = CStringGetTextDatum(model);
        values[0] = model_value;

        spi_rc = SPI_execute_with_args(
            "SELECT p.name, p.data, t.shape "
            "FROM llm_param p "
            "LEFT JOIN llm_tensor_map m ON (m.model = p.model AND m.name = p.name AND m.token_id = p.token_id) "
            "LEFT JOIN llm_tensor_rt t ON (t.id = m.tensor_id) "
            "WHERE p.model = $1 "
            "ORDER BY p.name, p.token_id",
            1, argtypes, values, NULL, true, 0);

        if (spi_rc != SPI_OK_SELECT)
            ereport(ERROR,
                    (errmsg("SPI_execute_with_args failed with code %d", spi_rc)));

        for (uint64 i = 0; i < SPI_processed; i++)
        {
            HeapTuple   tuple = SPI_tuptable->vals[i];
            TupleDesc   tupdesc = SPI_tuptable->tupdesc;
            bool        isnull;
            Datum       name_d;
            Datum       data_d;
            Datum       shape_d;
            text       *name_t;
            bytea      *data_b;
            ArrayType  *shape_a = NULL;
            char       *name_cstr;
            Pointer     data_ptr;
            Pointer     shape_ptr = NULL;

            name_d = SPI_getbinval(tuple, tupdesc, 1, &isnull);
            if (isnull)
                ereport(ERROR,
                        (errmsg("NULL parameter name encountered")));

            name_t = DatumGetTextPP(name_d);
            name_cstr = text_to_cstring(name_t);

            data_d = SPI_getbinval(tuple, tupdesc, 2, &isnull);
            if (isnull)
                ereport(ERROR,
                        (errmsg("parameter \"%s\" has NULL data", name_cstr)));

            data_ptr = DatumGetPointer(data_d);
            data_b = DatumGetByteaP(data_d);

            shape_d = SPI_getbinval(tuple, tupdesc, 3, &isnull);
            if (!isnull)
            {
                shape_ptr = DatumGetPointer(shape_d);
                shape_a = DatumGetArrayTypeP(shape_d);
            }

            write_npz_entry(fp, name_cstr, data_b, shape_a);

            pfree(name_cstr);
            if ((Pointer) data_b != data_ptr)
                pfree(data_b);
            if (shape_a && (Pointer) shape_a != shape_ptr)
                pfree(shape_a);
        }

        pfree(DatumGetPointer(model_value));

        if (gzclose(fp) != Z_OK)
            ereport(ERROR,
                    (errmsg("gzclose failed while exporting %s", path)));
        fp = NULL;

        SPI_finish();
    }
    PG_CATCH();
    {
        if (fp)
            gzclose(fp);
        SPI_finish();
        pfree(path);
        pfree(model);
        PG_RE_THROW();
    }
    PG_END_TRY();

    pfree(path);
    pfree(model);

    PG_RETURN_VOID();
}

static void
write_npz_entry(gzFile fp, const char *name, bytea *data, ArrayType *shape_array)
{
    Size        name_len;
    Size        payload_bytes;
    Size        elem_size = sizeof(float);
    Size        nelems;
    Size        dims_local[8];
    Size       *dims = dims_local;
    int         ndims = 0;
    bool        dims_allocated = false;
    bool        use_shape = false;
    StringInfoData header;
    Size        base_len;
    Size        pad;
    Size        total_len;
    unsigned char header_len_buf[2];
    unsigned char name_len_buf[2];
    static const unsigned char magic[] = {0x93, 'N', 'U', 'M', 'P', 'Y'};
    static const unsigned char version[] = {1, 0};
    char       *header_buf;

    if (name == NULL)
        ereport(ERROR,
                (errmsg("attempted to export tensor with NULL name")));

    name_len = strlen(name);
    if (name_len == 0)
        ereport(ERROR,
                (errmsg("tensor name must not be empty")));
    if (name_len > UINT16_MAX)
        ereport(ERROR,
                (errmsg("tensor name too long: %s", name)));

    payload_bytes = VARSIZE_ANY_EXHDR(data);

    if (elem_size == 0)
        ereport(ERROR,
                (errmsg("invalid element size for tensor %s", name)));

    if (payload_bytes % elem_size != 0)
        ereport(ERROR,
                (errmsg("tensor %s has %zu bytes which is not divisible by %zu", name,
                        payload_bytes, elem_size)));

    nelems = payload_bytes / elem_size;

    if (shape_array != NULL)
    {
        if (ARR_ELEMTYPE(shape_array) != INT4OID)
            ereport(ERROR,
                    (errmsg("tensor %s has non-int32 shape metadata", name)));

        if (ARR_HASNULL(shape_array))
            ereport(ERROR,
                    (errmsg("tensor %s shape metadata contains NULLs", name)));

        if (ARR_NDIM(shape_array) > 0)
        {
            int         ndims_shape = ArrayGetNItems(ARR_NDIM(shape_array), ARR_DIMS(shape_array));

            if (ndims_shape > 0)
            {
                int32      *shape_vals = (int32 *) ARR_DATA_PTR(shape_array);
                Size        total = 1;

                if (ndims_shape > (int) lengthof(dims_local))
                {
                    dims = (Size *) palloc(sizeof(Size) * ndims_shape);
                    dims_allocated = true;
                }

                ndims = ndims_shape;
                use_shape = true;

                for (int i = 0; i < ndims_shape; i++)
                {
                    if (shape_vals[i] <= 0)
                    {
                        use_shape = false;
                        break;
                    }

                    if (total > SIZE_MAX / (Size) shape_vals[i])
                        ereport(ERROR,
                                (errmsg("tensor %s shape metadata overflows element count", name)));

                    total *= (Size) shape_vals[i];
                    dims[i] = (Size) shape_vals[i];
                }

                if (!use_shape || total != nelems)
                {
                    use_shape = false;
                    ndims = 0;
                }
            }
        }
    }

    if (!use_shape)
    {
        ndims = 1;
        if (dims_allocated)
        {
            pfree(dims);
            dims = dims_local;
            dims_allocated = false;
        }
        dims[0] = nelems;
    }

    initStringInfo(&header);
    appendStringInfoString(&header,
                           "{'descr': '<f4', 'fortran_order': False, 'shape': (");

    for (int i = 0; i < ndims; i++)
    {
        if (i > 0)
            appendStringInfoString(&header, ", ");
        appendStringInfo(&header, "%zu", (size_t) dims[i]);
    }

    if (ndims == 1)
        appendStringInfoString(&header, ",");

    appendStringInfoString(&header, "), }");

    base_len = header.len;

    pad = (16 - ((10 + base_len + 1) % 16)) % 16;
    total_len = base_len + pad + 1;

    if (total_len > UINT16_MAX)
        ereport(ERROR,
                (errmsg("numpy header for tensor %s too large", name)));

    header_buf = (char *) palloc(total_len);
    memcpy(header_buf, header.data, base_len);
    MemSet(header_buf + base_len, ' ', pad);
    header_buf[base_len + pad] = '\n';

    name_len_buf[0] = (unsigned char) (name_len & 0xFF);
    name_len_buf[1] = (unsigned char) ((name_len >> 8) & 0xFF);
    gzwrite_exact(fp, name_len_buf, sizeof(name_len_buf), "tensor name length");
    gzwrite_exact(fp, name, name_len, "tensor name");
    gzwrite_exact(fp, magic, sizeof(magic), "numpy magic");
    gzwrite_exact(fp, version, sizeof(version), "numpy version");

    header_len_buf[0] = (unsigned char) (total_len & 0xFF);
    header_len_buf[1] = (unsigned char) ((total_len >> 8) & 0xFF);
    gzwrite_exact(fp, header_len_buf, sizeof(header_len_buf), "numpy header length");
    gzwrite_exact(fp, header_buf, total_len, "numpy header");

    if (payload_bytes > 0)
        gzwrite_exact(fp, VARDATA_ANY(data), payload_bytes, name);

    pfree(header.data);
    pfree(header_buf);
    if (dims_allocated)
        pfree(dims);
}

static void
gzwrite_exact(gzFile fp, const void *src, Size nbytes, const char *context)
{
    Size        offset = 0;
    const unsigned char *ptr = (const unsigned char *) src;

    while (offset < nbytes)
    {
        int nwritten = gzwrite(fp, ptr + offset, nbytes - offset);

        if (nwritten == 0)
        {
            int errnum;
            const char *msg = gzerror(fp, &errnum);

            if (errnum == Z_OK)
                ereport(ERROR,
                        (errmsg("unexpected short write while writing %s", context)));
            else
                ereport(ERROR,
                        (errmsg("error writing %s: %s", context,
                                 msg ? msg : "unknown error")));
        }
        else if (nwritten < 0)
        {
            int errnum;
            const char *msg = gzerror(fp, &errnum);
            ereport(ERROR,
                    (errmsg("error writing %s: %s", context,
                             msg ? msg : "unknown error")));
        }

        offset += nwritten;
    }
}
