# Quick Setup Guide

## Prerequisites

- Python 3.8+
- Nscale API token (get from [console.nscale.com/settings](https://console.nscale.com/settings))

## Quick Start

```bash
# 1. Setup environment
make setup
make install

# 2. Add your API token to .env
echo "NSCALE_TOKEN=your_token_here" > .env

# 3. Auto-fetch organization ID
make get-org-id

# 4. Create sample data
make data

# 5. Run fine-tuning
make finetune-monitor
```

## Step-by-Step Setup

### 1. API Token

Get your API token from: https://console.nscale.com/settings

Add it to `.env`:
```bash
NSCALE_TOKEN=your_token_here
```

### 2. Organization ID (Automatic)

The organization ID is automatically fetched from the identity API:

```bash
make get-org-id
```

This will:
- Call the identity API at `https://identity.nks.europe-west4.nscale.com/api/v1/acl`
- List all organizations associated with your token
- Automatically select the first organization
- Update `.env` with `ORGANIZATION_ID`

**Example output:**
```
Fetching organizations from identity API...

Found 2 organization(s):
  1. abc-123-def-456
     Projects: 5, Endpoints: 3
  2. xyz-789-ghi-012
     Projects: 2, Endpoints: 1

Selecting organization: abc-123-def-456

✓ .env file updated successfully!
  ORGANIZATION_ID=abc-123-def-456
```

## Verify Setup

```bash
# Check credentials are configured
cat .env

# Should show:
# NSCALE_TOKEN=...
# ORGANIZATION_ID=...

# Test inference (doesn't require org ID)
python inference.py --list-models
```

## Troubleshooting

**Missing Organization ID**
```bash
make get-org-id
```

**Python Module Errors**
```bash
source venv/bin/activate
pip install -r requirements.txt
```

**File Not Found**
```bash
make data  # Create sample datasets
```

For more details, see [README.md](README.md).
