#include "pg_llm.h"
#include "executor/spi.h"

PG_FUNCTION_INFO_V1(pg_llm_load_bpe_vocab);
Datum pg_llm_load_bpe_vocab(PG_FUNCTION_ARGS)
{
    text *path_t = PG_GETARG_TEXT_P(0);
    text *model_t = PG_GETARG_TEXT_P(1);
    char *path = text_to_cstring(path_t);
    char *model = text_to_cstring(model_t);

    FILE *f = fopen(path,"r");
    if(!f) ereport(ERROR,(errmsg("cannot open %s",path)));
    SPI_connect();
    StringInfoData q; initStringInfo(&q);

    char token[512]; int id;
    while(fscanf(f," \"%[^\"]\" : %d,",token,&id)==2){
        resetStringInfo(&q);
        appendStringInfo(&q,
            "INSERT INTO llm_bpe_vocab(model,token_id,token,bytes)"
            "VALUES('%s',%d,$1,$2) ON CONFLICT DO NOTHING;",model,id);
        Oid types[2]={TEXTOID,BYTEAOID};
        Datum vals[2]={CStringGetTextDatum(token),
                       PointerGetDatum(cstring_to_text(token))};
        SPI_execute_with_args(q.data,2,types,vals,NULL,false,0);
    }
    fclose(f);
    SPI_finish();
    PG_RETURN_VOID();
}

PG_FUNCTION_INFO_V1(pg_llm_load_bpe_merges);
Datum pg_llm_load_bpe_merges(PG_FUNCTION_ARGS)
{
    text *path_t = PG_GETARG_TEXT_P(0);
    text *model_t = PG_GETARG_TEXT_P(1);
    char *path=text_to_cstring(path_t);
    char *model=text_to_cstring(model_t);

    FILE *f=fopen(path,"r");
    if(!f) ereport(ERROR,(errmsg("cannot open %s",path)));
    SPI_connect();
    char l[256], r[256]; int rank=0;
    while(fscanf(f,"%255s %255s",l,r)==2){
        StringInfoData q; initStringInfo(&q);
        appendStringInfo(&q,
            "INSERT INTO llm_bpe_merges(model,rank,left,right,pair)"
            "VALUES('%s',%d,'%s','%s','%s %s')",
            model,rank,l,r,l,r);
        SPI_execute(q.data,false,0);
        rank++;
    }
    fclose(f);
    SPI_finish();
    PG_RETURN_VOID();
}
