"""把 LoRA adapter merge 回 bf16 base，並（可選）驗證生成。

  merge：  uv run python -m training.merge_adapter --adapter <dir> --out <dir>
  驗證：   加 --verify（merge 後載入 merged 模型跑一題資安生成，重用 smoke_test 同題）

QLoRA 標準做法：在 bf16（非量化）base 上 merge —— LoRA adapter 權重本身是 bf16，
合進 bf16 base 不引入 4-bit 量化誤差（切勿在 4-bit base 上 merge）。
base 預設以 device_map="cpu" 載入：merge 是純權重運算、不需 GPU，且避免 offload
狀態下的 device mismatch。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from . import config


def merge_adapter(adapter_dir: str | Path, out_dir: str | Path, device_map: str = "cpu") -> Path:
    """以 bf16 載入 base → 套 adapter → merge_and_unload → 存完整權重。"""
    from peft import PeftModel

    from modeling.loader import load_model_and_processor

    print(f">> 以 bf16 載入 base（device_map={device_map}）…")
    base, processor = load_model_and_processor(device_map=device_map)

    print(f">> 套用 adapter：{adapter_dir}")
    peft_model = PeftModel.from_pretrained(base, str(adapter_dir))

    print(">> merge_and_unload（在 bf16 base 上 merge）…")
    merged = peft_model.merge_and_unload()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f">> 存 merged 權重：{out_dir}")
    merged.save_pretrained(str(out_dir))
    processor.save_pretrained(str(out_dir))
    return out_dir


def verify_generation(model_dir: str | Path, max_new_tokens: int = 128) -> str:
    """載入 merged 模型，用 smoke_test 同題（資安問答）跑一題生成，確認連貫。"""
    import torch

    from modeling.config import SMOKE_USER_PROMPT
    from modeling.loader import load_model_and_processor
    from modeling.smoke_test import _build_messages

    print(f">> 載入 merged 模型驗證生成：{model_dir}")
    model, processor = load_model_and_processor(model_id=str(model_dir))
    input_device = model.get_input_embeddings().weight.device
    inputs = processor.apply_chat_template(
        _build_messages(),
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(input_device)

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    generated = out[0][inputs["input_ids"].shape[-1] :]
    decode = getattr(processor, "decode", None) or processor.tokenizer.decode
    text = decode(generated, skip_special_tokens=True)

    print("=" * 60)
    print("merged 模型生成結果")
    print("=" * 60)
    print(f"問題：{SMOKE_USER_PROMPT}\n")
    print(text.strip())
    return text


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", default=str(config.ADAPTER_DIR), help="LoRA adapter 目錄")
    ap.add_argument("--out", default=str(config.MERGED_DIR), help="merged 權重輸出目錄")
    ap.add_argument("--device-map", default="cpu", help='base 載入 device_map（merge 建議 "cpu"）')
    ap.add_argument("--verify", action="store_true", help="merge 後載入 merged 模型跑一題生成驗證")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    out_dir = merge_adapter(args.adapter, args.out, device_map=args.device_map)
    if args.verify:
        verify_generation(out_dir, max_new_tokens=args.max_new_tokens)


if __name__ == "__main__":
    main()
