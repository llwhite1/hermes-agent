---
sidebar_position: 18
title: "TAMU AI Chat"
description: "Set up Standard or Preview TAMU AI Chat access in Hermes, discover protected models, use Preview image generation, and review local per-model usage"
---

# TAMU AI Chat

This Hermes branch adds a guided setup for Texas A&M's OpenAI-compatible AI service. Choose **Standard access** or **Preview access**, paste the matching API key, and Hermes discovers the `protected.*` models available to your account.

The integration is additive. OpenRouter, Nous Portal, Anthropic, local models, and every other normal Hermes option remain available.

:::info Why is the branch named `codex/tamu-quick-setup`?
This is a branch inside the **Hermes Agent repository**. `codex/` is only the development branch namespace; it does not mean the feature belongs to the Codex application. A future release branch can use a public-facing name such as `feature/tamu-ai-chat`.
:::

## Choose your access

You need a Texas A&M account authorized for TAMU AI Chat and an API key for the service you intend to use.

| Setup choice | Inference base URL | Best for |
|---|---|---|
| **Standard access** | `https://chat-api.tamu.ai/openai` | Normal supported TAMUS use |
| **Preview access** | `https://chat-api.preview.tamu.ai/openai` | Early-access models and Preview-only capabilities, including image output |

Use the [Standard documentation](https://docs.tamus.ai/docs/prod/) or [Preview documentation](https://docs.tamus.ai/docs/preview/) to obtain the appropriate access. Do not assume that a Preview key works with Standard access, or the reverse.

Preview-hosted models are a pilot service. Capacity, settings, and availability may change, so Hermes checks the authenticated live catalog instead of assuming that every documented model is enabled for every key.

## Run the quick setup

Open a terminal and run:

```bash
hermes tamu setup
```

The same quick path is available from the main setup command:

```bash
hermes setup tamu
# or: hermes setup --tamu
```

The wizard will:

1. ask whether you have Standard or Preview access;
2. request the API key with masked input;
3. call the endpoint's authenticated `GET /models` route;
4. keep model IDs beginning with `protected.`;
5. exclude embedding and image-output models from the main-agent picker;
6. save TAMUS-specific context metadata for known models;
7. clamp the requested response cap to the model's documented output limit;
8. enable stream compatibility for TAMU auxiliary and Mixture-of-Agents calls; and
9. for Preview access, optionally configure an image model for `image_generate`.

![TAMU quick setup terminal showing Preview access, protected model discovery, a selected main model, maximum documented context, and an optional Preview image model](/img/guides/tamu-ai-chat/setup-preview.png)

The screenshot is a clean-profile rehearsal using a fake key and cached catalog. Your live model count and available models may differ.

### Skip individual prompts

Use the clearer access names in scripts:

```bash
hermes tamu setup --access standard
hermes tamu setup --access preview
```

The older `--environment production|preview` spelling remains supported for compatibility.

If you know the exact IDs exposed by your key:

```bash
hermes tamu setup \
  --access preview \
  --model "protected.Laguna-S-2.1" \
  --image-model "protected.gemini-3.1-flash-lite-image"
```

Use `--no-image-setup` when you want Preview text models without an image-setup prompt. Hermes rejects a typed model that is absent from the authenticated catalog.

## Preview image generation

The authenticated [Preview model documentation](https://docs.tamus.ai/docs/preview/models/) currently documents five image-output models:

| Model ID | Useful starting point |
|---|---|
| `protected.gemini-3.1-flash-lite-image` | Fast, efficient iteration; Hermes default |
| `protected.gemini-3.1-flash-image` | Higher-quality Gemini generation and conversational editing |
| `protected.gpt-image-1-mini` | Cost-efficient GPT image generation and editing |
| `protected.gpt-image-1.5` | Strong instruction following and adherence |
| `protected.gpt-image-2` | Highest-fidelity GPT generation and editing |

After Preview setup, Hermes' normal `image_generate` tool uses the selected model. Text-to-image and one-source image editing are supported. Generated data URLs are decoded into local files under `~/.hermes/cache/images/`.

TAMUS currently serves these models through the streaming `POST /openai/chat/completions` route. Direct OpenAI Images passthrough is disabled, so the Hermes `tamu-preview` image plugin deliberately uses chat completions and extracts the returned image from the stream.

:::caution Preview availability
Documentation describes the Preview catalog, while your authenticated `GET /models` response determines what your key can use. The setup only offers documented image models that are visible in that live response.
:::

## Where the key is stored

The secret is stored locally in the active Hermes profile's `.env` file, not in `config.yaml`:

```text
HERMES_TAMU_PRODUCTION_API_KEY=...
HERMES_TAMU_PREVIEW_API_KEY=...
```

Only the relevant variable is written. Configuration references the variable by name. Hermes does not print its value in status output, save it in the model catalog, or include it in usage reports.

## Verify the result

Run a local configuration check:

```bash
hermes tamu status
```

Add `--check` to verify the live main and image models:

```bash
hermes tamu status --check
```

![TAMU status terminal showing Standard access unconfigured and Preview access active, including the selected image model and live visibility checks](/img/guides/tamu-ai-chat/status-preview.png)

`status` never displays the key. The live check only calls the TAMU model catalog.

## See every available model

```bash
hermes tamu models
```

Query a specific endpoint when it is not active:

```bash
hermes tamu models --environment production
hermes tamu models --environment preview --json
```

The human-readable list labels image-output and embedding models so they are not confused with main-agent models. The list itself always comes from the authenticated endpoint.

## Context windows and response limits

The model catalog may omit limits, so setup uses this order:

1. a positive operator value already saved for that exact model;
2. context metadata returned by the TAMU catalog;
3. TAMUS' documented model-specific limit; then
4. generic Hermes metadata when no TAMUS-specific value is known.

TAMUS-specific values win over generic model-family guesses. For example, TAMUS currently documents Claude Sonnet 4.6 at 200,000 input tokens even though the same family may have a larger window elsewhere.

Unknown models remain on runtime auto-detection. Hermes does not invent a large context number.

The **context window** and **response-token cap** are different:

- Context window: working space for instructions, conversation, tool results, and the response.
- Response-token cap: maximum tokens Hermes requests for one response.

The requested response cap defaults to 32,768. If that is larger than TAMUS' documented output limit for the selected model, setup automatically clamps it and tells you. You can request a different cap:

```bash
hermes tamu setup --max-output-tokens 8192
```

To leave per-model context values to runtime detection:

```bash
hermes tamu setup --no-context-overrides
```

Setup normally preserves a positive value you deliberately saved. To replace older saved values with the current catalog or TAMUS-documented limits—and therefore raise a model to its highest currently documented window—run:

```bash
hermes tamu setup --access preview --refresh-context-limits
```

## Local usage by model and agent task

Hermes already records routed language-model calls in its local `state.db`. This branch adds a TAMU-filtered view and a separate prompt-free image-call ledger:

```bash
hermes tamu usage --days 30
```

![Local TAMU usage terminal showing main, MoA, compression, and Preview image-generation rows with API-call, token, and image totals](/img/guides/tamu-ai-chat/usage-preview.png)

The report separates exact model IDs and tasks such as main-agent calls, MoA references, compression, and `image_generate`. The image ledger stores only timestamp, model ID, success/failure type, input-image count, and output-image count—never prompts, image contents, or API keys.

Filter language-model rows by a Hermes source or export JSON:

```bash
hermes tamu usage --days 7 --source cli
hermes tamu usage --days 30 --json
```

Image calls are not currently source-attributed, so `--source` omits them and says so. Image responses do not currently expose authoritative token usage; the report counts calls and generated images without inventing token or dollar estimates.

## What setup writes

A Preview setup with images is equivalent to:

```yaml
model:
  provider: custom:tamu-preview
  default: protected.Laguna-S-2.1
  context_length: 1048576
  max_tokens: 32768

providers:
  tamu-preview:
    name: TAMU AI Chat — Preview access
    api: https://chat-api.preview.tamu.ai/openai
    key_env: HERMES_TAMU_PREVIEW_API_KEY
    transport: chat_completions
    default_model: protected.Laguna-S-2.1
    discover_models: true
    models:
      protected.Laguna-S-2.1:
        context_length: 1048576

image_gen:
  provider: tamu-preview
  model: protected.gemini-3.1-flash-lite-image
  use_gateway: false
  tamu_preview:
    model: protected.gemini-3.1-flash-lite-image

auxiliary:
  stream_only_base_urls:
    - chat-api.preview.tamu.ai
```

The actual `models` map contains every protected model discovered during setup. Existing unrelated configuration is preserved.

## Advanced: Anthropic-compatible route

TAMUS also documents an Anthropic-compatible endpoint for tools such as Claude Code. Its Standard base is `https://chat-api.tamu.ai/api`, with the same TAMU API key supplied as `ANTHROPIC_AUTH_TOKEN` and a protected Claude model name. That route is useful for Anthropic-native clients; Hermes' guided TAMU provider uses the broader OpenAI-compatible catalog so one setup can discover Claude, Gemini, GPT, open, and image models together.

See the authenticated [TAMUS Claude Code setup](https://docs.tamus.ai/docs/prod/api-tool/tool_integrations/setup-claude-code-cli-with-tamus-ai-chat) for the current native-client variables.

## Data handling

Texas A&M describes TAMU AI Chat as a university-approved, multi-model service with an OpenAI-compatible API. Review the current [TAMU Technology Services service page](https://www.it.tamu.edu/services/services-by-category/communication-and-collaboration/tamu-ai-chat.html) before using institutional data. Preview access should be treated as a pilot even when its underlying model is also available elsewhere.

## Troubleshooting

### The key is rejected

- Confirm that Standard or Preview matches where the key was issued.
- Re-run setup, choose to replace the saved key, and paste it again.
- Run `hermes tamu status --check`.

### No `protected.*` models are found

The key authenticated, but the catalog did not expose expected protected models. Confirm your TAMU entitlement and authenticated documentation. Hermes deliberately does not fall back to an unprotected list.

### An image model is documented but not offered

Run `hermes tamu models --environment preview`. Your key's live catalog is authoritative for availability. A documented Preview model may be temporarily unavailable or not enabled for that account.

### Main chat works but MoA or compression fails

Re-run setup. It adds the selected hostname to `auxiliary.stream_only_base_urls`, allowing auxiliary requests to use streaming and aggregate the result locally.

### A model reports the wrong context size

Check the current TAMUS model documentation. A positive value you explicitly saved for an exact model is preserved across setup runs; remove or correct that override if the documented limit changed.

### Usage is empty

Usage begins after Hermes makes calls through a TAMU route. Older sessions, calls outside Hermes, and past image calls made before the local image ledger existed cannot be reconstructed.
