# Fine-Tuning Testing Summary

## Changes Made to Match PubMedQA Cookbook

Following the official [PubMedQA Fine-Tuning Cookbook](https://docs.nscale.com/docs/cookbooks/pubmedqa-fine-tuning), we updated:

### 1. Dataset Creation (`finetune.py`)
- ✅ **Removed** `prompt_column` and `answer_column` from dataset creation
- ✅ These fields now only appear in job creation (as per cookbook)
- ✅ Removed `purpose` parameter from file upload

### 2. Configuration (`config.yaml`)
- ✅ Updated to use **Meta-Llama-3.1-8B-Instruct** (cookbook model)
- ✅ Model ID: `5b912da4-4a68-43eb-9224-38239535d934`
- ✅ Hyperparameters match cookbook exactly:
  - batch_size: 32
  - learning_rate: 0.00001
  - n_epochs: 3
  - n_evals: 6
  - weight_decay: 0.03
  - LoRA alpha: 16 (instead of 8)

### 3. Payload Structure
Updated job creation to match cookbook format:
```json
{
  "name": "job-name",
  "base_model_id": "5b912da4-4a68-43eb-9224-38239535d934",
  "dataset": {
    "id": "dataset-id",
    "prompt_column": "question",
    "answer_column": "answer"
  },
  "hyperparameters": { ... }
}
```

## Testing Results

### Test 1: Python Script (After Updates)
```bash
make finetune
```
**Result**: `500 Internal Server Error`

### Test 2: Direct curl Command (Cookbook Format)
```bash
curl -X POST "https://fine-tuning.api.nscale.com/api/v1/organizations/$ORGANIZATION_ID/jobs" \
  -H "Authorization: Bearer $NSCALE_API_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{ <exact payload from cookbook> }'
```
**Result**: `{"code":"internal","message":null}`

### Test 3: Verify Other Endpoints Work
```bash
# List jobs
curl GET /jobs  → ✅ 200 OK

# List datasets  
curl GET /datasets → ✅ 200 OK

# Create dataset
curl POST /datasets → ✅ 200 OK

# Create job
curl POST /jobs → ❌ 500 ERROR
```

## Conclusion

✅ **Our code is correct** - matches official cookbook exactly
✅ **Dataset creation works** - successfully creates datasets  
✅ **Other APIs work** - can list and read resources
❌ **Job creation is broken** - returns 500 error for ALL attempts

This is confirmed to be an **API-side issue**, not a problem with our implementation.

## Evidence

1. ✅ Payload structure matches PubMedQA cookbook exactly
2. ✅ Same error with Python requests and curl
3. ✅ Same error with different models (Llama, DeepSeek)
4. ✅ Same error with old and new datasets
5. ✅ User had successful job earlier today (API worked at 14:57 UTC)
6. ✅ Now nothing works (API broken since ~15:00 UTC)

## Next Steps

**For User:**
1. Contact Nscale Support with `ISSUE_REPORT.md`
2. Reference cookbook example that fails
3. Monitor [Nscale Status Page](https://status.nscale.com) if available
4. Try again in a few hours

**Files to Share with Support:**
- `ISSUE_REPORT.md` - Detailed diagnostics
- `TESTING_SUMMARY.md` - This file
- Successful job ID: `44df5731-ee86-4898-8a8b-49f029792c96`
