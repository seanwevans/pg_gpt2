#include "pg_llm.h"
#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <sys/types.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <zlib.h>

typedef struct npy_header_t
{
    char   *key;           /* tensor name */
    char   *dtype;         /* numpy descriptor string */
    bool    fortran_order;
    int     ndim;
    Size   *shape;         /* array of ndim elements */
    Size    elem_size;     /* size in bytes of a single element */
    Size    payload_bytes; /* total bytes for the tensor payload */
} npy_header_t;

static void free_npy_header(npy_header_t *hdr);
static ssize_t read_npz_entry(gzFile fp, npy_header_t *hdr);
static void gzread_exact(gzFile fp, void *dest, Size nbytes, const char *context);
static void parse_npy_header(const char *header, npy_header_t *hdr);

PG_FUNCTION_INFO_V1(pg_llm_import_npz);

/*
 * pg_llm_import_npz(path TEXT, model TEXT)
 * Reads a .npz or .safetensors file and populates llm_param
 */
Datum pg_llm_import_npz(PG_FUNCTION_ARGS)
{
    text *path_t = PG_GETARG_TEXT_P(0);
    text *model_t = PG_GETARG_TEXT_P(1);
    char *path = text_to_cstring(path_t);
    char *model = text_to_cstring(model_t);
    gzFile fp;

    SPI_connect();

    fp = gzopen(path, "rb");
    if (!fp)
    {
        SPI_finish();
        ereport(ERROR,
                (errmsg("could not open %s", path)));
    }

    PG_TRY();
    {
        while (true)
        {
            npy_header_t hdr;
            ssize_t payload_bytes;
            bytea *data;
            StringInfoData q;
            Oid argtypes[1];
            Datum values[1];

            MemSet(&hdr, 0, sizeof(npy_header_t));

            payload_bytes = read_npz_entry(fp, &hdr);
            if (payload_bytes == 0)
            {
                free_npy_header(&hdr);
                break;
            }

            if (hdr.fortran_order)
                ereport(ERROR,
                        (errmsg("Fortran-ordered arrays are not supported in %s", path)));

            if (hdr.elem_size == 0 || hdr.payload_bytes != (Size) payload_bytes)
                ereport(ERROR,
                        (errmsg("invalid numpy header for entry \"%s\"", hdr.key)));

            if (payload_bytes < 0)
                ereport(ERROR,
                        (errmsg("invalid payload size for entry \"%s\"", hdr.key)));

            if (hdr.payload_bytes != (Size) payload_bytes)
                ereport(ERROR,
                        (errmsg("payload size mismatch for entry \"%s\"", hdr.key)));

            if ((Size) payload_bytes > MaxAllocSize - VARHDRSZ)
                ereport(ERROR,
                        (errmsg("tensor \"%s\" too large to fit in memory", hdr.key)));

            data = (bytea *) palloc(VARHDRSZ + hdr.payload_bytes);
            SET_VARSIZE(data, VARHDRSZ + hdr.payload_bytes);
            gzread_exact(fp, VARDATA(data), hdr.payload_bytes,
                         hdr.key ? hdr.key : "tensor payload");

            initStringInfo(&q);
            appendStringInfo(&q,
                             "INSERT INTO llm_param(model,name,data,grad,m,v,step)"
                             "VALUES('%s','%s',$1,'','', '',0)"
                             "ON CONFLICT (model,name) DO UPDATE SET data=$1;",
                             model, hdr.key);
            argtypes[0] = BYTEAOID;
            values[0] = PointerGetDatum(data);
            SPI_execute_with_args(q.data, 1, argtypes, values, NULL, false, 0);

            pfree(data);
            pfree(q.data);
            free_npy_header(&hdr);
        }

        gzclose(fp);
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

    PG_RETURN_VOID();
}

static void
free_npy_header(npy_header_t *hdr)
{
    if (!hdr)
        return;

    if (hdr->key)
        pfree(hdr->key);
    if (hdr->dtype)
        pfree(hdr->dtype);
    if (hdr->shape)
        pfree(hdr->shape);

    MemSet(hdr, 0, sizeof(npy_header_t));
}

static char *
read_header_string(gzFile fp, Size header_len)
{
    char *header = (char *) palloc(header_len + 1);

    gzread_exact(fp, header, header_len, "numpy header");
    header[header_len] = '\0';

    return header;
}

static Size
dtype_size_from_descr(const char *descr)
{
    const char *p = descr;

    if (p == NULL)
        return 0;

    while (*p && !isdigit((unsigned char) *p))
        p++;

    if (!*p)
        return 0;

    return (Size) strtoul(p, NULL, 10);
}

static void
parse_shape(const char *shape_str, npy_header_t *hdr)
{
    Size   *dims = NULL;
    int     ndims = 0;
    int     capacity = 0;
    const char *p = shape_str;
    long dim = 0;

    while (isspace((unsigned char) *p))
        p++;

    if (*p != '(')
        ereport(ERROR,
                (errmsg("invalid numpy header: malformed shape tuple")));

    p++;
    while (true)
    {
        while (isspace((unsigned char) *p))
            p++;

        if (*p == ')')
        {
            p++;
            break;
        }

        if (!isdigit((unsigned char) *p))
            ereport(ERROR,
                    (errmsg("invalid numpy header: expected dimension length")));

        errno = 0;
        dim = strtol(p, (char **) &p, 10);
        if (errno != 0 || dim < 0)
            ereport(ERROR,
                    (errmsg("invalid numpy header: bad dimension length")));

        if (ndims == capacity)
        {
            capacity = capacity ? capacity * 2 : 4;
            if (dims)
                dims = (Size *) repalloc(dims, capacity * sizeof(Size));
            else
                dims = (Size *) palloc(capacity * sizeof(Size));
        }
        dims[ndims++] = (Size) dim;

        while (isspace((unsigned char) *p))
            p++;

        if (*p == ',')
        {
            p++;
            continue;
        }
        else if (*p == ')')
        {
            p++;
            break;
        }
        else
            ereport(ERROR,
                    (errmsg("invalid numpy header: unexpected character in shape")));
    }

    if (ndims == 0)
    {
        if (dims)
            pfree(dims);
        hdr->shape = NULL;
        hdr->ndim = 0;
        return;
    }

    hdr->shape = dims;
    hdr->ndim = ndims;
}

static void
parse_npy_header(const char *header, npy_header_t *hdr)
{
    const char *descr_key = strstr(header, "'descr':");
    const char *fortran_key = strstr(header, "'fortran_order':");
    const char *shape_key = strstr(header, "'shape':");
    const char *descr_end;
    char quote;
    Size total_elems;

    if (!descr_key || !fortran_key || !shape_key)
        ereport(ERROR,
                (errmsg("invalid numpy header: missing required fields")));

    descr_key += strlen("'descr':");
    while (isspace((unsigned char) *descr_key))
        descr_key++;
    if (*descr_key != '\'' && *descr_key != '"')
        ereport(ERROR,
                (errmsg("invalid numpy header: malformed descr")));
    quote = *descr_key++;
    descr_end = strchr(descr_key, quote);
    if (!descr_end)
        ereport(ERROR,
                (errmsg("invalid numpy header: unterminated descr")));
    hdr->dtype = pnstrdup(descr_key, descr_end - descr_key);
    hdr->elem_size = dtype_size_from_descr(hdr->dtype);
    if (hdr->elem_size == 0)
        ereport(ERROR,
                (errmsg("unsupported numpy dtype \"%s\"", hdr->dtype)));

    fortran_key += strlen("'fortran_order':");
    while (isspace((unsigned char) *fortran_key))
        fortran_key++;
    if (strncmp(fortran_key, "True", 4) == 0)
        hdr->fortran_order = true;
    else if (strncmp(fortran_key, "False", 5) == 0)
        hdr->fortran_order = false;
    else
        ereport(ERROR,
                (errmsg("invalid numpy header: malformed fortran_order")));

    shape_key += strlen("'shape':");
    while (isspace((unsigned char) *shape_key))
        shape_key++;
    parse_shape(shape_key, hdr);

    /* compute payload size */
    total_elems = 1;
    if (hdr->ndim == 0)
    {
        total_elems = 1;
    }
    else
    {
        for (int i = 0; i < hdr->ndim; i++)
        {
            if (hdr->shape[i] == 0)
            {
                hdr->payload_bytes = 0;
                return;
            }

            if (total_elems > SIZE_MAX / hdr->shape[i])
                ereport(ERROR,
                        (errmsg("tensor dimensions overflow for entry \"%s\"", hdr->key)));
            total_elems *= hdr->shape[i];
        }
    }

    if (hdr->elem_size > 0 && total_elems > SIZE_MAX / hdr->elem_size)
        ereport(ERROR,
                (errmsg("tensor payload too large for entry \"%s\"", hdr->key)));

    hdr->payload_bytes = total_elems * hdr->elem_size;
}

static ssize_t
read_npz_entry(gzFile fp, npy_header_t *hdr)
{
    unsigned char name_len_buf[2];
    int      nread;
    char    *header_str;
    unsigned char magic[6];
    unsigned char version[2];
    Size     header_len;
    Size     name_len;

    nread = gzread(fp, name_len_buf, sizeof(name_len_buf));
    if (nread == 0)
        return 0;               /* EOF */
    if (nread != sizeof(name_len_buf))
        ereport(ERROR,
                (errmsg("failed to read npz entry name length")));

    name_len = (Size) name_len_buf[0] | ((Size) name_len_buf[1] << 8);
    if (name_len == 0)
        ereport(ERROR,
                (errmsg("npz entry with empty name")));

    hdr->key = (char *) palloc(name_len + 1);
    gzread_exact(fp, hdr->key, name_len, "tensor name");
    hdr->key[name_len] = '\0';

    gzread_exact(fp, magic, sizeof(magic), "numpy magic");
    if (memcmp(magic, "\x93NUMPY", 6) != 0)
        ereport(ERROR,
                (errmsg("invalid numpy magic for entry \"%s\"", hdr->key)));

    gzread_exact(fp, version, sizeof(version), "numpy version");

    if (version[0] == 1 && version[1] == 0)
    {
        unsigned char header_len_buf_v1[2];

        gzread_exact(fp, header_len_buf_v1, sizeof(header_len_buf_v1),
                     "numpy header length");
        header_len = (Size) header_len_buf_v1[0] |
                     ((Size) header_len_buf_v1[1] << 8);
    }
    else if (version[0] == 2 && version[1] == 0)
    {
        unsigned char header_len_buf_v2[4];

        gzread_exact(fp, header_len_buf_v2, sizeof(header_len_buf_v2),
                     "numpy header length");
        header_len = (Size) header_len_buf_v2[0] |
                     ((Size) header_len_buf_v2[1] << 8) |
                     ((Size) header_len_buf_v2[2] << 16) |
                     ((Size) header_len_buf_v2[3] << 24);
    }
    else
        ereport(ERROR,
                (errmsg("unsupported numpy header version %u.%u for entry \"%s\"",
                        version[0], version[1], hdr->key)));

    header_str = read_header_string(fp, header_len);
    parse_npy_header(header_str, hdr);
    pfree(header_str);

    return (ssize_t) hdr->payload_bytes;
}

static void
gzread_exact(gzFile fp, void *dest, Size nbytes, const char *context)
{
    Size offset = 0;
    unsigned char *ptr = (unsigned char *) dest;

    while (offset < nbytes)
    {
        int nread = gzread(fp, ptr + offset, nbytes - offset);
        if (nread <= 0)
        {
            int errnum;
            const char *msg = gzerror(fp, &errnum);
            if (errnum == Z_OK || errnum == Z_STREAM_END)
                ereport(ERROR,
                        (errmsg("unexpected EOF while reading %s", context)));
            else
                ereport(ERROR,
                        (errmsg("error reading %s: %s", context,
                                 msg ? msg : "unknown error")));
        }
        offset += nread;
    }
}
