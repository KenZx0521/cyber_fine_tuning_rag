"""端到端 smoke test：載入模型 → 純文字資安問答生成。

這是「能載入地端」的驗證進入點（對齊 build_dataset.py 的 argparse + main(argv) 風格）。

用法：
    uv run python -m modeling.smoke_test
    uv run python -m modeling.smoke_test --no-generate      # 只載入＋診斷
    uv run python -m modeling.smoke_test --max-new-tokens 128
"""

from __future__ import annotations

import argparse
import time

import torch

from . import config
from .loader import load_model_and_processor, report_diagnostics


def _build_messages() -> list[dict]:
    """純文字 chat：system + 一題資安問題。content 用多模態的 list 形式以契合 VLM 模板。"""
    return [
        {"role": "system", "content": config.SECURITY_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [{"type": "text", "text": config.SMOKE_USER_PROMPT}],
        },
    ]


def run(max_new_tokens: int, do_generate: bool, max_gpu_mem: str | None = None) -> None:
    print(f"載入模型：{config.MODEL_ID}")
    max_memory = (
        {0: max_gpu_mem, "cpu": config.CPU_MEM_BUDGET} if max_gpu_mem else None
    )
    t0 = time.perf_counter()
    model, processor = load_model_and_processor(max_memory=max_memory)
    print(f"載入耗時：{time.perf_counter() - t0:.1f}s")

    report_diagnostics(model)

    if not do_generate:
        print("\n[--no-generate] 只載入，未生成。")
        return

    # device_map 模型用 input embedding 所在裝置作輸入落點（首個運算在此），比 model.device 穩健。
    input_device = model.get_input_embeddings().weight.device
    inputs = processor.apply_chat_template(
        _build_messages(),
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(input_device)

    print("\n生成中…")
    t1 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    generated = out[0][inputs["input_ids"].shape[-1] :]
    decode = getattr(processor, "decode", None) or processor.tokenizer.decode
    text = decode(generated, skip_special_tokens=True)
    elapsed = time.perf_counter() - t1
    n_tok = generated.shape[-1]

    print("=" * 60)
    print("生成結果")
    print("=" * 60)
    print(f"問題：{config.SMOKE_USER_PROMPT}\n")
    print(text.strip())
    print(f"\n（{n_tok} tokens、{elapsed:.1f}s、{n_tok / elapsed:.1f} tok/s）")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="模型載入 + 純文字生成 smoke test")
    parser.add_argument(
        "--max-new-tokens", type=int, default=config.SMOKE_MAX_NEW_TOKENS
    )
    parser.add_argument(
        "--no-generate", action="store_true", help="只載入並印診斷，不生成"
    )
    parser.add_argument(
        "--max-gpu-mem",
        default=None,
        help='手動指定 GPU 權重上限（如 "68GiB"）；預設依可用 VRAM 自動預留 headroom',
    )
    args = parser.parse_args(argv)
    run(
        args.max_new_tokens,
        do_generate=not args.no_generate,
        max_gpu_mem=args.max_gpu_mem,
    )


if __name__ == "__main__":
    main()
