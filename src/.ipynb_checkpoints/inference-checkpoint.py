import asyncio, json, time
from typing import Dict, Any, List, Optional
from postprocessing import PostProcessor
from api_client import OpenAICompatClient

SYSTEM = "You are a precise Python coding assistant. Reply with code only."

def make_instr(header: str, def_src: str) -> str:
    return header.rstrip() + "\n\n" + def_src.rstrip() + "\n<sol>\n"

def build_payload(client: OpenAICompatClient, instr: str, stop=None, **gen):
    if client.use_chat:
        payload = dict(messages=[{"role":"system","content":SYSTEM},{"role":"user","content":instr}], **gen)
    else:
        payload = dict(prompt=instr, **gen)
    if stop: payload["stop"] = stop
    return payload

def extract_text(client: OpenAICompatClient, data: Dict[str, Any]) -> str:
    ch = data["choices"][0]
    return (ch.get("message") or {}).get("content") or ch.get("text") or ""

def generate_one(client: OpenAICompatClient, header: str, ex: Dict[str, Any], **gen) -> Dict[str, Any]:
    def_src = PostProcessor.extract_def_from_prompt(ex["prompt"], ex["entry_point"])
    instr = make_instr(header, def_src)
    data = client.complete(instr, system=SYSTEM, **gen)
    text = extract_text(client, data)
    body = PostProcessor.normalize_body(text)
    return {
        "task_id": ex["task_id"],
        "prompt": ex["prompt"],
        "entry_point": ex["entry_point"],
        "canonical_solution": ex["canonical_solution"],
        "test": ex["test"],
        "raw_text": text,
        "completion": body,
        "usage": data.get("usage", {}),
    }

# Async batch (Jupyter-safe helper below)
async def generate_many_async(client: OpenAICompatClient, header: str, ds, concurrency: int, stop=None, **gen):
    import aiohttp
    url = f"{client.api_base}/chat/completions" if client.use_chat else f"{client.api_base}/completions"
    hdr = client.headers
    sem = asyncio.Semaphore(concurrency)
    async with aiohttp.ClientSession() as session:
        async def _one(ex):
            def_src = PostProcessor.extract_def_from_prompt(ex["prompt"], ex["entry_point"])
            instr = make_instr(header, def_src)
            payload = {"model": client.model, **build_payload(client, instr, stop=stop, **gen)}
            async with sem, session.post(url, headers=hdr, json=payload, timeout=gen.get("timeout",180)) as r:
                r.raise_for_status()
                data = await r.json()
                text = extract_text(client, data)
                body = PostProcessor.normalize_body(text)
                return {
                    "task_id": ex["task_id"],
                    "prompt": ex["prompt"],
                    "entry_point": ex["entry_point"],
                    "canonical_solution": ex["canonical_solution"],
                    "test": ex["test"],
                    "raw_text": text,
                    "completion": body,
                    "usage": data.get("usage", {}),
                }
        tasks = [asyncio.create_task(_one(ex)) for ex in ds]
        return await asyncio.gather(*tasks)

# Jupyter-safe runner
def run_coro(coro):
    try:
        import nest_asyncio, asyncio
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        else:
            nest_asyncio.apply()
            return loop.run_until_complete(coro)
    except Exception:
        import asyncio
        return asyncio.run(coro)
