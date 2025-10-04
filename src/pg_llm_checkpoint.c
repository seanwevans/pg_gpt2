#include "pg_llm.h"
#include <zlib.h>
#include <json-c/json.h>
#include "numpy/ndarrayobject.h"   /* if available */

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

    SPI_connect();

    gzFile fp = gzopen(path, "rb");
    if (!fp)
        ereport(ERROR,(errmsg("could not open %s",path)));

    /* Iterate entries in zip */
    npy_header_t hdr;
    while (read_npz_entry(fp, &hdr)) {
        const char *name = hdr.key;
        int len = hdr.shape[0]*hdr.shape[1];
        bytea *data = (bytea*) palloc(VARHDRSZ + len*sizeof(float));
        SET_VARSIZE(data, VARHDRSZ + len*sizeof(float));
        gzread(fp, VARDATA(data), len*sizeof(float));

        StringInfoData q; initStringInfo(&q);
        appendStringInfo(&q,
            "INSERT INTO llm_param(model,name,data,grad,m,v,step)"
            "VALUES('%s','%s',$1,'','', '',0)"
            "ON CONFLICT (model,name) DO UPDATE SET data=$1;",
            model,name);
        Oid argtypes[1] = {BYTEAOID};
        Datum values[1] = {PointerGetDatum(data)};
        SPI_execute_with_args(q.data,1,argtypes,values,NULL,false,0);
    }

    SPI_finish();
    PG_RETURN_VOID();
}
