---
sidebar_position: 18
title: "TAMU AI Chat"
description: "Set up the Texas A&M OpenAI-compatible endpoint in Hermes, discover protected models, maximize known context windows, and review local per-model usage"
---

# TAMU AI Chat

This Hermes branch adds a guided setup for Texas A&M's OpenAI-compatible AI service. You choose the production or preview endpoint, paste one API key, select a `protected.*` model, and Hermes handles the custom-provider settings that are otherwise easy to miss.

The integration is additive. OpenRouter, Nous Portal, Anthropic, local models, and every other normal Hermes option remain available.

## Before you start

You need:

- A working Hermes installation from the `codex/tamu-quick-setup` branch.
- A Texas A&M account authorized for TAMU AI Chat.
- An API key for the endpoint you intend to use. Obtain and manage keys through the authenticated [TAMU AI Chat documentation portal](https://docs.tamus.ai/).

Choose the endpoint that issued your key:

| Environment | Inference base URL | Use it for |
|---|---|---|
| Production | `https://chat-api.tamu.ai/openai` | Normal, supported use |
| Preview | `https://chat-api.preview.tamu.ai/openai` | Testing preview models and behavior |

Do not assume that a preview key works in production, or vice versa. If a key is rejected, verify that you selected the matching environment.

## Run the quick setup

Open a terminal and run:

```bash
hermes tamu setup
```

The same quick path is also available from the main setup command:

```bash
hermes setup tamu
# or: hermes setup --tamu
```

The wizard will:

1. Ask whether you want production or preview.
2. Request the API key with masked input.
3. Call the endpoint's authenticated `GET /models` route;
4. keep all model IDs beginning with `protected.`;
5. let you search and select a default model;
6. save the model catalog and known context-window metadata;
7. set a 32,768-token response cap by default; and
8. enable Hermes' stream-only compatibility for TAMU auxiliary and Mixture-of-Agents calls.

![TAMU quick setup terminal showing a masked key, 43 discovered preview models, selected Gemini model, 1,048,576-token context window, and successful configuration](/img/guides/tamu-ai-chat/setup-preview.png)

The screenshot is a clean-profile rehearsal using a fake key. The model count reflects that preview catalog at the time of capture; your live count and available models may differ.

### Set the environment or model directly

You can skip the endpoint prompt:

```bash
hermes tamu setup --environment production
hermes tamu setup --environment preview
```

If you already know the exact live model ID:

```bash
hermes tamu setup \
  --environment preview \
  --model "protected.gemini-2.5-pro"
```

Hermes rejects a typed model that is not present in the authenticated catalog, which prevents a stale or misspelled model from becoming the default.

## Where the key is stored

The secret is stored locally in the active Hermes profile's `.env` file, not in `config.yaml`:

```text
HERMES_TAMU_PRODUCTION_API_KEY=...
HERMES_TAMU_PREVIEW_API_KEY=...
```

Only the relevant variable is written. The provider configuration references the variable by name. Hermes does not print the value in `status`, write it into the model catalog, or include it in local usage reports.

Re-run `hermes tamu setup` to keep or replace an existing key.

## Verify the result

Run a local configuration check:

```bash
hermes tamu status
```

Add `--check` to authenticate to the selected endpoint and verify that the default model is still visible:

```bash
hermes tamu status --check
```

![TAMU status terminal showing production unconfigured and preview active with a saved key, 43 models, and stream compatibility enabled](/img/guides/tamu-ai-chat/status-preview.png)

`status` never displays the key. The live check only calls the TAMU model catalog.

## See every available model

```bash
hermes tamu models
```

Query a specific endpoint when it is not the active one:

```bash
hermes tamu models --environment production
hermes tamu models --environment preview --json
```

The list comes from the current authenticated endpoint, not a frozen model list shipped with Hermes.

## Context windows and response limits

TAMU's OpenAI-compatible model catalog may omit context-window metadata. The setup therefore uses this order:

1. context metadata returned by TAMU, when present;
2. a context value you previously saved for that same model; then
3. a conservative, model-specific value already known to Hermes.

Unknown models remain on Hermes' runtime auto-detection path. The setup does not assign an invented context value merely to make the number larger.

The **context window** and **response-token cap** are different:

- Context window: total working space for instructions, conversation, tool results, and the response.
- Response-token cap: maximum tokens Hermes requests for one model response.

The default response cap is 32,768 tokens. If a particular TAMU model rejects that cap, re-run setup with a smaller value:

```bash
hermes tamu setup --max-output-tokens 8192
```

To leave every per-model context value to runtime detection:

```bash
hermes tamu setup --no-context-overrides
```

## Local usage by model and agent task

Hermes already records each routed API call in its local `state.db`. This branch adds a TAMU-filtered view:

```bash
hermes tamu usage --days 30
```

![Local TAMU usage terminal showing separate main, MoA reference, and compression rows with API-call and token totals](/img/guides/tamu-ai-chat/usage-preview.png)

The screenshot uses synthetic usage rows. A real report separates:

- endpoint environment;
- exact model ID;
- main-agent calls;
- MoA reference calls; and
- auxiliary tasks such as compression, vision analysis, or title generation.

Filter by a Hermes source or export structured JSON:

```bash
hermes tamu usage --days 7 --source cli
hermes tamu usage --days 30 --json
```

This report is local-only. Running it does not transmit analytics to Texas A&M, Nous Research, or another service. It reports tokens and API calls, not dollars, because Hermes does not have an authoritative TAMU price table and TAMU access is currently described in terms of daily token allowances.

## What the setup writes

The generated configuration is equivalent to:

```yaml
model:
  provider: custom:tamu-preview
  default: protected.gemini-2.5-pro
  context_length: 1048576
  max_tokens: 32768

providers:
  tamu-preview:
    name: TAMU AI Chat (Preview)
    api: https://chat-api.preview.tamu.ai/openai
    key_env: HERMES_TAMU_PREVIEW_API_KEY
    transport: chat_completions
    default_model: protected.gemini-2.5-pro
    discover_models: true
    models:
      protected.gemini-2.5-pro:
        context_length: 1048576

auxiliary:
  stream_only_base_urls:
    - chat-api.preview.tamu.ai
```

The actual `models` map contains every model discovered during setup. Existing unrelated configuration is preserved.

## Data handling and current limitations

Texas A&M describes TAMU AI Chat as a university-approved, multi-model service with an OpenAI-compatible API. Review the current [TAMU Technology Services service page](https://www.it.tamu.edu/services/services-by-category/communication-and-collaboration/tamu-ai-chat.html) before using institutional data.

At the time this guide was written, the official page states that University-Confidential data and FERPA data are allowed, HIPAA use requires caution, and several Restricted Data categories are not allowed—including export- or IRB-controlled data, government ID numbers, personal health and financial records, and sensitive personal information. The service page also notes that sessions are not used to train external models but may be accessed by the service owner for troubleshooting and service improvements.

Image generation is not currently available through TAMU AI Chat. Configure a separate supported image provider in Hermes for image creation; do not assume that selecting a multimodal TAMU chat model enables image generation.

## Troubleshooting

### The key is rejected

- Confirm that preview or production matches where the key was issued.
- Re-run `hermes tamu setup`, choose to replace the saved key, and paste it again.
- Run `hermes tamu status --check`.

### No `protected.*` models are found

The key authenticated, but the catalog did not expose the expected protected models. Confirm your TAMU account/API entitlement and check the authenticated documentation portal. Hermes deliberately does not fall back to an unprotected model list.

### Main chat works but MoA or compression fails

Re-run `hermes tamu setup`. It adds the selected TAMU hostname to `auxiliary.stream_only_base_urls`, which makes non-interactive auxiliary requests use a streamed request and aggregate the result locally.

### A model reports the wrong context size

TAMU's model list may not publish that field. Correct the selected model's `model.context_length` and its `providers.tamu-<environment>.models.<model>.context_length` value in `config.yaml`, then preserve it by re-running setup. Existing positive operator values take precedence over Hermes' metadata match.

### Usage is empty

Usage begins after Hermes makes attributed calls through the TAMU base URL. Older sessions created before per-model accounting, or calls made outside Hermes, cannot be reconstructed by this command.
