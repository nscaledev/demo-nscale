# Nscale Setup Guide

## Prerequisites

- Python 3.8+
- Nscale API token
- Nscale Organization ID

## Finding Your Credentials

### 1. API Token

Your API token is already configured in `.env`:
```
NSCALE_TOKEN=eyJhbGci...
```

### 2. Organization ID

**Option A: Use the Helper Script (Recommended)**

```bash
make get-org-id
```

This will attempt to fetch your organizations and help you configure the `.env` file.

**Option B: Manual Configuration**

1. Visit https://console.nscale.com/settings
2. Find your "Organization ID" (looks like: `org-abc123xyz789`)
3. Add it to `.env`:
   ```
   ORGANIZATION_ID=org-abc123xyz789
   ```

## API Endpoints

This demo uses the correct Nscale API endpoints:

- **Fine-tuning**: `https://fine-tuning.api.nscale.com/api/v1/organizations/{organization_id}`
- **Inference**: `https://inference.api.nscale.com/v1`

## Complete Setup Flow

```bash
# 1. Create virtual environment
make setup

# 2. Install dependencies
make install

# 3. Configure organization ID
make get-org-id

# 4. Create sample data
make data

# 5. Start fine-tuning
make finetune-monitor
```

## Troubleshooting

### "ORGANIZATION_ID not found" Error

Make sure your `.env` file has both:
```
NSCALE_TOKEN=your_token_here
ORGANIZATION_ID=your_org_id_here
```

### "Failed to resolve 'api.nscale.com'" Error

This error is now fixed! The demo uses the correct endpoints:
- Fine-tuning: `fine-tuning.api.nscale.com`
- Inference: `inference.api.nscale.com`

### Can't Fetch Organization ID Automatically

The API endpoint for listing organizations may require different permissions.
Use the manual method instead:
1. Visit https://console.nscale.com/settings
2. Copy your organization ID
3. Run: `make get-org-id` and enter it manually

## Quick Test

Verify your setup:

```bash
# Check .env file
cat .env

# Should show:
# NSCALE_TOKEN=...
# ORGANIZATION_ID=...
```

## Ready to Go!

Once your `.env` is configured with both values, you're ready to start:

```bash
make finetune-monitor
```
