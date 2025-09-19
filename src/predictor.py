#!/usr/bin/env python3
import aiohttp
import asyncio
import json
from postprocessing import PostProcessor

async def infer_async(
    ds,
    api_base: str,
    model_id: str,
    token: str,
    use_chat: bool,
    header: str,
    profile: dict,
):
    """
    Run inference asynchronously over a dataset.
    
    Args:
        ds: dataset (list of examples, each with prompt/entry_point/etc.)
        api_base: server URL (http://host:port/v1)
        model_id: model name (string)
        token: auth token for API
        use_chat: whether to call /chat/completions or /completions
        header: prompt header string to prepend
        profile: dict with inference parameters:
            - temperature
            - top_p
            - max_tokens
            - stop
            - concurrency
    """
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    results = []

    async def infer_one(session, ex):
        def_src = PostProcessor.extract_def_from_prompt(ex["prompt"], ex["entry_point"])
        instr = header.rstrip() + "\n\n" + f"{def_src.rstrip()}\n<sol>\n"

        payload = {
            "model": model_id,
            "max_tokens": profile.get("max_tokens", 512),
            "temperature": profile.get("temperature", 0.2),
            "top_p": profile.get("top_p", 1.0),
        }
        if profile.get("stop"):
            payload["stop"] = profile["stop"]

        if use_chat:
            payload["messages"] = [
                {"role": "system", "content": "You are a precise Python coding assistant. Reply with code only."},
                {"role": "user", "content": instr},
            ]
            url = f"{api_base}/chat/completions"
        else:
            payload["prompt"] = instr
            url = f"{api_base}/completions"

        async with session.post(url, headers=headers, json=payload, timeout=180) as resp:
            data = await resp.json()
            choice = data["choices"][0]
            text = (choice.get("message") or {}).get("content") or choice.get("text") or ""
            return {
                "task_id": ex["task_id"],
                "prompt": ex["prompt"],
                "entry_point": ex["entry_point"],
                "canonical_solution": ex["canonical_solution"],
                "test": ex["test"],
                "raw_text": text,
            }

    conn = aiohttp.TCPConnector(limit=profile.get("concurrency", 8), ttl_dns_cache=300)
    timeout = aiohttp.ClientTimeout(total=None)

    async with aiohttp.ClientSession(connector=conn, timeout=timeout) as session:
        tasks = [infer_one(session, ex) for ex in ds]
        for fut in asyncio.as_completed(tasks):
            try:
                res = await fut
                results.append(res)
            except Exception as e:
                results.append({"error": str(e)})

    return results
