# Nscale API Issue Report

## Issue Summary
Consistent `500 Internal Server Error` when creating fine-tuning jobs via API, despite successful job creation earlier today.

**Update**: Tested with exact payload structure from official [PubMedQA Fine-Tuning Cookbook](https://docs.nscale.com/docs/cookbooks/pubmedqa-fine-tuning) - same 500 error persists.

## Environment
- **Organization ID**: `a7467a9d-90d4-4976-8a08-cc5bd358b56e`
- **API Endpoint**: `https://fine-tuning.api.nscale.com/api/v1/organizations/{org_id}/jobs`
- **Date/Time**: 2026-01-14 15:00-15:10 UTC

## Previous Successful Job
- **Job ID**: `44df5731-ee86-4898-8a8b-49f029792c96`
- **Status**: Completed successfully
- **Created**: 2026-01-14T14:57:45.376538Z
- **Completed**: 2026-01-14T14:59:06.077Z
- **Model**: DeepSeek-R1-Distill-Qwen-1.5B (`e082db90-cf4a-4be9-a0e6-b2f8067c1cf6`)

## Tests Performed

### 1. Exact Replica of Successful Job
- Used EXACT same base model ID
- Used EXACT same dataset ID (`6fd13761-dc3c-4403-ad88-869bf8f6731e`)
- Used EXACT same hyperparameters
- **Result**: 500 Internal Server Error

### 2. New Dataset Test
- Created new datasets successfully via `/datasets` endpoint
- Datasets are identical in structure to successful one (same files, same size)
- Attempted job creation with new datasets
- **Result**: 500 Internal Server Error

### 3. Minimal Payload Test
- Used only required fields (name, base_model_id, dataset with minimal info)
- **Result**: 500 Internal Server Error

### 4. Different Models Test
- Tested with Meta-Llama-3.1-8B-Instruct model
- **Result**: 500 Internal Server Error

### 5. Dry Run Test
- Used `dry_run: true` parameter
- **Result**: 500 Internal Server Error

## API Response
```json
{
  "code": "internal",
  "message": null
}
```

## Working Operations
✅ GET `/base-models` - Returns list of available models
✅ GET `/jobs` - Returns job history
✅ GET `/jobs/{id}` - Returns job details
✅ POST `/datasets` - Successfully creates datasets
✅ GET `/datasets` - Returns dataset list
✅ GET `/datasets/{id}` - Returns dataset details

❌ POST `/jobs` - Consistently returns 500 error

## Sample Payloads Tested

### Payload 1: Exact replica of successful job
```json
{
  "name": "replicate_successful_job",
  "base_model_id": "e082db90-cf4a-4be9-a0e6-b2f8067c1cf6",
  "dataset": {
    "id": "6fd13761-dc3c-4403-ad88-869bf8f6731e",
    "prompt_column": "question",
    "answer_column": "answer"
  },
  "hyperparameters": {
    "n_epochs": 1,
    "n_evals": 1,
    "warmup_ratio": 0,
    "batch_size": 4,
    "learning_rate": 0.00001,
    "weight_decay": 0.01,
    "mask_prompt_labels": false,
    "best_checkpoints": false,
    "lora": {
      "enabled": true,
      "r": 8,
      "alpha": 8,
      "dropout": 0,
      "trainable_modules": []
    }
  }
}
```

### Payload 2: Minimal required fields
```json
{
  "name": "minimal_test",
  "base_model_id": "e082db90-cf4a-4be9-a0e6-b2f8067c1cf6",
  "dataset": {
    "id": "6fd13761-dc3c-4403-ad88-869bf8f6731e",
    "answer_column": "answer"
  }
}
```

Both result in the same 500 error.

### Payload 3: Following PubMedQA Cookbook (via curl)
Tested using curl with exact structure from official documentation:

```bash
curl -X POST "https://fine-tuning.api.nscale.com/api/v1/organizations/$ORGANIZATION_ID/jobs" \
  -H "Authorization: Bearer $NSCALE_API_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{
    "name": "curl-test-job",
    "base_model_id": "5b912da4-4a68-43eb-9224-38239535d934",
    "dataset": {
      "id": "0dcf0972-9420-4390-b2cf-94c565f6721a",
      "prompt_column": "question",
      "answer_column": "answer"
    },
    "hyperparameters": {
      "batch_size": 32,
      "best_checkpoints": false,
      "learning_rate": 0.00001,
      "lora": {
        "alpha": 16,
        "dropout": 0,
        "enabled": true,
        "r": 8,
        "trainable_modules": []
      },
      "n_epochs": 3,
      "n_evals": 6,
      "warmup_ratio": 0,
      "weight_decay": 0.03,
      "mask_prompt_labels": false
    }
  }'
```

**Result**: `{"code":"internal","message":null}`

## Conclusion
The issue appears to be on the API server side, as:
1. The exact same configuration that worked 2 hours ago now fails
2. All payload variations tested result in the same 500 error
3. The error occurs even with the exact same dataset used in the successful job
4. **Cookbook example fails**: Exact payload from official PubMedQA cookbook returns 500 error
5. **curl testing**: Direct curl commands (not Python) also fail with same error
6. Other API endpoints work correctly (GET /jobs, GET /datasets, POST /datasets)
7. Datasets are created successfully but cannot be used in job creation
8. Account has only 1 previous job (no quota issues)

## Recommended Actions
1. Check Nscale API server logs for the time period 15:00-15:10 UTC on 2026-01-14
2. Verify job creation endpoint is operational
3. Check for any recent API changes or deployments
4. Verify organization quotas/limits are not being hit (no clear error message if this is the case)

## Contact
Please contact Nscale support with this issue report.
