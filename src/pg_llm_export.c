#include "pg_llm.h"
#include "access/htup_details.h"
#include "lib/stringinfo.h"
#include <zlib.h>
#include <limits.h>

static void write_npz_entry(gzFile fp, const char *name, bytea *data);
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

    SPI_connect();

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
            "SELECT name, data FROM llm_param WHERE model = $1 ORDER BY name, token_id",
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
            text       *name_t;
            bytea      *data_b;
            char       *name_cstr;
            Pointer     data_ptr;

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

            write_npz_entry(fp, name_cstr, data_b);

            pfree(name_cstr);
            if ((Pointer) data_b != data_ptr)
                pfree(data_b);
        }

        pfree(DatumGetPointer(model_value));

        gzclose(fp);
        fp = NULL;

        SPI_finish();
    }
    PG_CATCH();
    {
        if (fp)
            gzclose(fp);
        SPI_finish();
        PG_RE_THROW();
    }
    PG_END_TRY();

    pfree(path);
    pfree(model);

    PG_RETURN_VOID();
}

static void
write_npz_entry(gzFile fp, const char *name, bytea *data)
{
    Size        name_len;
    Size        payload_bytes;
    Size        elem_size = sizeof(float);
    Size        nelems;
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

    initStringInfo(&header);
    appendStringInfo(&header,
                     "{'descr': '<f4', 'fortran_order': False, 'shape': (%zu,), }",
                     (size_t) nelems);

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
