# Model assets

`docker compose` mounts this directory into the database and provisioning
containers as `/mnt/models`. Nothing here is committed.

For a real GPT-2 deployment, put three files in this directory:

| File | Where it comes from |
|------|---------------------|
| `vocab.json` | GPT-2 tokenizer vocabulary |
| `merges.txt` | GPT-2 BPE merge ranks |
| `gpt2-small.npz` | `python scripts/convert_gpt2_checkpoint.py --source gpt2 --output models/gpt2-small.npz` |

Then point the provisioning step at the checkpoint:

```bash
PG_GPT2_WEIGHTS=/mnt/models/gpt2-small.npz docker compose run --rm provision
```

Leaving `PG_GPT2_WEIGHTS` unset provisions a small randomly-initialised model
instead, which is enough to exercise the endpoint end to end.
