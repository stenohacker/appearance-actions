# Appearance Actions Deploy Bundle

This folder is meant to be its own small GitHub repo.

Use this if your existing repo is messy or confusing.

## Contents

- `app/` - FastAPI backend
- `requirements.txt`
- `render.yaml`
- `openapi.yaml` - GPT Action schema

## Simplest workflow

1. Create a new empty GitHub repo named something like `appearance-actions`.
2. Upload the contents of this folder to that repo.
3. In Render, create a new `Blueprint` from that repo.
4. After deploy, open your Render URL and confirm `/healthz` works.
5. In your GPT, add a new Action and paste `openapi.yaml`.
6. Replace the placeholder server URL in `openapi.yaml` with your Render URL.

## What to replace

In `openapi.yaml`, replace:

`https://YOUR-ACTION-HOST.example.com`

with your actual Render URL, for example:

`https://appearance-actions-backend.onrender.com`
