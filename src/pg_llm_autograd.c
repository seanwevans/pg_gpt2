#include "pg_llm.h"
#include "executor/spi.h"
#include "commands/trigger.h"

static int tape_insert(const char *name, int *inputs, int n_in, int output, const char *extra_json)
{
    SPI_connect();
    StringInfoData buf; initStringInfo(&buf);
    appendStringInfo(&buf,
        "INSERT INTO llm_tape(name,inputs,output,extra) VALUES('%s',ARRAY[", name);
    for(int i=0;i<n_in;i++){ if(i>0) appendStringInfoChar(&buf,','); appendStringInfo(&buf,"%d",inputs[i]); }
    appendStringInfo(&buf,"],%d,'%s')",output,extra_json?extra_json:"{}");
    SPI_execute(buf.data,false,0);
    SPI_finish();
    return output;
}
