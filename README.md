# ComfyUI Visual AI Asset Pipeline

A node-based AI execution backend for generating visual assets and delivering them to modern web applications. ComfyUI provides the workflow runtime; API-format workflow exports make those visual pipelines callable from application backends, batch jobs, and content-production systems.

## Project Overview

This repository combines ComfyUI's visual workflow engine with an integration pattern for programmatic asset generation:

- **Visual pipeline authoring:** Compose model loading, prompt conditioning, sampling, decoding, and output nodes into reusable graphs.
- **Workflow templates:** The [`blueprints/`](blueprints/) directory contains visual workflow templates for image, video, audio, and other tasks.
- **Headless execution:** Submit an API-format workflow graph to ComfyUI's REST API without driving the browser interface.
- **Application integration:** A frontend can submit generation requests to its application backend, which can populate workflow inputs, enqueue work, and return generated assets to the client.

The model and custom-node requirements depend on the selected workflow. This repository provides the ComfyUI runtime and templates; install the required model files and any custom nodes before executing a workflow.

## Headless API Integration

### Export a workflow

Build or open a workflow in ComfyUI, then select **File > Export (API)**. This exports the executable node graph used by the API, rather than the regular UI workflow layout. Save the export in a location your backend can load, such as `workflows/text_to_image_api.json`. The `workflows/` directory is a suggested location for your API exports; the visual templates themselves are in `blueprints/`.

API exports use node IDs as JSON keys. Before integrating a workflow, identify the positive text-encoder node and the `KSampler` node in the export. Their IDs and input names are workflow-specific.

### Submit, monitor, and retrieve an image

The example below demonstrates the server-side flow: inject a request's prompt and seed into an exported graph, submit it to `POST /prompt`, poll `GET /history/{prompt_id}`, and retrieve the generated image through `GET /view`.

```python
import json
import random
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

COMFY = "http://127.0.0.1:8188"
POSITIVE_PROMPT_NODE = "6"  # Replace with the positive text-encoder node ID.
KSAMPLER_NODE = "3"         # Replace with the KSampler node ID.

# These values can come from a validated frontend request.
user_prompt = "A cinematic mountain landscape at sunrise"
seed = random.SystemRandom().randrange(0, 2**64)

workflow = json.loads(
    Path("workflows/text_to_image_api.json").read_text(encoding="utf-8")
)
workflow[POSITIVE_PROMPT_NODE]["inputs"]["text"] = user_prompt
workflow[KSAMPLER_NODE]["inputs"]["seed"] = seed

request = Request(
    f"{COMFY}/prompt",
    data=json.dumps({"prompt": workflow}).encode("utf-8"),
    headers={"Content-Type": "application/json"},
)
with urlopen(request, timeout=30) as response:
    prompt_id = json.load(response)["prompt_id"]

deadline = time.monotonic() + 600
while time.monotonic() < deadline:
    with urlopen(f"{COMFY}/history/{prompt_id}", timeout=30) as response:
        history = json.load(response)

    result = history.get(prompt_id)
    if result:
        status = result.get("status", {}).get("status_str")
        if status == "error":
            raise RuntimeError(f"ComfyUI execution failed: {result['status']}")
        if status == "success":
            image = next(
                (
                    image
                    for output in result.get("outputs", {}).values()
                    for image in output.get("images", [])
                ),
                None,
            )
            if image is None:
                raise RuntimeError("Workflow completed without an image output")

            query = urlencode(
                {
                    "filename": image["filename"],
                    "subfolder": image["subfolder"],
                    "type": image["type"],
                }
            )
            with urlopen(f"{COMFY}/view?{query}", timeout=60) as response:
                Path("generated.png").write_bytes(response.read())
            print(f"Saved generated.png (seed={seed})")
            break

    time.sleep(1)
else:
    raise TimeoutError(f"Prompt {prompt_id} did not finish within 10 minutes")
```

For event-driven progress rather than polling, connect to ComfyUI's `/ws` endpoint and correlate execution events with the returned prompt ID. The repository's [`script_examples/`](script_examples/) directory includes REST and WebSocket API examples.

### Production integration pattern

Keep the browser-facing API separate from the model worker:

1. The frontend sends prompt parameters to an application backend.
2. The backend validates inputs, selects an approved workflow, and maps request values to its API graph.
3. A worker submits the graph to ComfyUI and tracks the prompt ID until completion.
4. The application returns an asset URL or result metadata to the frontend.

For production deployments, put authentication, rate limiting, request validation, queue management, retries, and asset retention in the application or worker layer. Keep ComfyUI on a private network rather than exposing its execution API directly to untrusted clients. Multiple workers can be managed by the deployment platform; GPU capacity and model residency determine practical concurrency.

## Run ComfyUI

Install the requirements for your platform and GPU, then start the server from the repository root:

```powershell
python main.py
```

The local server listens at `http://127.0.0.1:8188` by default. Place model files in their corresponding `models/` subdirectories and use workflow-specific setup instructions for additional custom nodes.
