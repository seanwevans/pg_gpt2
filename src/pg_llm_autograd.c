#include "pg_llm.h"
#include "executor/spi.h"
#include "commands/trigger.h"

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
        "INSERT INTO llm_tape(name,inputs,output,extra) VALUES('%s',ARRAY[", name);
    for(int i=0;i<n_in;i++){ if(i>0) appendStringInfoChar(&buf,','); appendStringInfo(&buf,"%d",inputs[i]); }
    appendStringInfo(&buf,"],%d,'%s')",output,extra_json?extra_json:"{}");
    SPI_execute(buf.data,false,0);
    SPI_finish();
    return output;
}
