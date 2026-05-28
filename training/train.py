"""QLoRA 訓練進入點。

  正式訓練：   uv run python -m training.train
  煙霧測試：   uv run python -m training.train --smoke
  自訂：       uv run python -m training.train --batch-size 4 --max-length 3072

組裝流程：bf16 載入 + LoRA → 載入資料 + per-source 加權（長度分組）→ WeightedSFTTrainer → 訓練 → 存 adapter。
情境（已拍板）：2 epochs + α=0.5 / cap=3（primus/general 權重=0 排除取樣），沿用既有 sampler_weights.json。
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# bs=3 峰值 VRAM 貼近 95GiB 上限；expandable_segments 減少碎片化以避免 OOM。
# 須在 torch 初始化 CUDA 前設定（torch 於各函式內延遲 import，故此處 module-load 即生效）。
# 已用環境變數設定者優先（setdefault 不覆寫）。
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from . import config


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true",
                    help="煙霧測試：極小子集 + 少步，快速驗證整條訓練路徑能跑通")
    ap.add_argument("--epochs", type=int, default=None, help="覆寫 num_train_epochs")
    ap.add_argument("--batch-size", type=int, default=None, help="覆寫 per_device_train_batch_size")
    ap.add_argument("--grad-accum", type=int, default=None, help="覆寫 gradient_accumulation_steps")
    ap.add_argument("--max-length", type=int, default=None, help="覆寫 max_length（截斷上限）")
    ap.add_argument("--output-dir", default=None, help="覆寫輸出目錄")
    ap.add_argument("--no-weighted", action="store_true",
                    help="關閉 per-source 加權取樣（除錯/對比用）")
    ap.add_argument("--extended-targets", action="store_true",
                    help="LoRA 額外納入 30 層 linear_attn 投影（進階實驗）")
    ap.add_argument("--no-wandb", action="store_true",
                    help="關閉 wandb 上報（離線/除錯；report_to=[]，不需 WANDB_API_KEY）")
    return ap.parse_args(argv)


def configure_wandb(args: argparse.Namespace) -> None:
    """載入 .env 並設定 wandb 環境變數（self-hosted）。--no-wandb 時直接略過。"""
    if args.no_wandb:
        return
    import os

    from dotenv import load_dotenv

    load_dotenv()  # override=False：優先序 shell env > .env > config 預設
    os.environ.setdefault("WANDB_PROJECT", config.WANDB_PROJECT)
    os.environ.setdefault("WANDB_BASE_URL", config.WANDB_BASE_URL)
    os.environ.setdefault("WANDB_LOG_MODEL", "false")  # 不把 checkpoint 當 artifact 上傳
    if not os.environ.get("WANDB_API_KEY"):
        raise SystemExit(
            "缺 WANDB_API_KEY：請把 .env.example 複製成 .env 並填入金鑰"
            f"（從 {config.WANDB_BASE_URL}/authorize 取得），或用 --no-wandb 關閉上報。"
        )


def build_sft_config(args: argparse.Namespace):
    """從 config 常數 + CLI 覆寫組 SFTConfig。"""
    from datetime import datetime

    from trl import SFTConfig

    smoke = args.smoke
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    default_dir = config.OUTPUT_DIR / "smoke" if smoke else config.RUN_DIR
    output_dir = args.output_dir or str(default_dir)
    kw = dict(
        output_dir=output_dir,
        bf16=True,
        bf16_full_eval=True,  # eval 也走 bf16（不 upcast fp32），更快更省 VRAM
        prediction_loss_only=True,  # eval 只需 eval_loss；不累積 248k-vocab logits（防 OOM、加速）
        per_device_train_batch_size=args.batch_size or (1 if smoke else config.PER_DEVICE_TRAIN_BATCH_SIZE),
        gradient_accumulation_steps=args.grad_accum or (1 if smoke else config.GRADIENT_ACCUMULATION_STEPS),
        per_device_eval_batch_size=config.PER_DEVICE_EVAL_BATCH_SIZE,
        learning_rate=config.LEARNING_RATE,
        lr_scheduler_type=config.LR_SCHEDULER_TYPE,
        warmup_ratio=config.WARMUP_RATIO,
        weight_decay=config.WEIGHT_DECAY,
        max_grad_norm=config.MAX_GRAD_NORM,
        # grad ckpt 在 prepare_model_for_kbit_training 與此處一致設定（use_reentrant=False，冪等）。
        gradient_checkpointing=config.GRADIENT_CHECKPOINTING,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=args.max_length or config.MAX_LENGTH,
        packing=False,  # 破壞 per-example 權重語意與多輪 mask 邊界，故不採用
        shuffle_dataset=False,  # 隨機性交給 WeightedRandomSampler，避免雙重 shuffle
        assistant_only_loss=True,  # 只在 assistant token 算 loss（依訓練模板的 {% generation %}）
        optim=config.OPTIM,
        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=config.SAVE_TOTAL_LIMIT,
        report_to=([] if args.no_wandb else config.REPORT_TO),
        project=config.WANDB_PROJECT,  # 明確指定 wandb 專案，避免被 SFTConfig 預設 'huggingface' 蓋過
        seed=config.SEED,
    )
    if smoke:
        kw.update(
            max_steps=config.SMOKE_MAX_STEPS,
            num_train_epochs=1,
            logging_steps=1,
            eval_steps=config.SMOKE_SAVE_STEPS,
            save_steps=config.SMOKE_SAVE_STEPS,
            dataloader_num_workers=0,
            load_best_model_at_end=False,
            run_name=f"smoke-{ts}",
        )
    else:
        epochs = args.epochs or config.NUM_TRAIN_EPOCHS
        kw.update(
            num_train_epochs=epochs,
            logging_steps=config.LOGGING_STEPS,
            eval_steps=config.EVAL_STEPS,
            save_steps=config.SAVE_STEPS,
            # num_workers=0：資料已預先 tokenize，取批僅需索引+padding（極輕）。用 worker 子進程會在
            # 「CUDA 已初始化後 fork」觸發不確定性的 dataloader 死鎖（實測 step 0 卡死 15 分、4 個 worker
            # 活著卻不產 batch）；單進程載入對此 GPU-bound（~5s/step）工作吞吐影響 <5%，換取穩定。
            dataloader_num_workers=0,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            run_name=f"qlora-cyber-{epochs}ep-{ts}",
        )
    return SFTConfig(**kw)


def main(argv=None):
    args = parse_args(argv)
    configure_wandb(args)  # 載入 .env + 設定 wandb 環境變數（須在 trainer.train() 前）

    from .data import build_example_weights, load_sampler_weights, load_sft_datasets
    from .lora_loader import (
        assert_only_lora_trainable,
        load_lora_model_and_processor,
        report_model,
    )
    from .weighted_trainer import WeightedSFTTrainer

    target_modules = config.TARGET_MODULES_WITH_LINEAR_ATTN if args.extended_targets else None
    print(">> 載入 bf16 模型並掛 LoRA …")
    model, processor = load_lora_model_and_processor(target_modules=target_modules)
    report_model(model)
    assert_only_lora_trainable(model)

    # 純文字 SFT 走 tokenizer（assistant_only_loss 依賴 tokenizer.apply_chat_template
    # 的 return_assistant_tokens_mask）。設自訂訓練模板（含 {% generation %} 標記）。
    tokenizer = getattr(processor, "tokenizer", processor)
    tokenizer.chat_template = config.TRAIN_TEMPLATE_PATH.read_text(encoding="utf-8")

    print(">> 載入資料 …")
    train_ds, val_ds = load_sft_datasets()
    if args.smoke:
        train_ds = train_ds.select(range(min(config.SMOKE_SUBSET, len(train_ds))))
        val_ds = val_ds.select(range(min(config.SMOKE_SUBSET, len(val_ds))))
    print(f"   train={len(train_ds)}  val={len(val_ds)}")

    example_weights = None
    if not args.no_weighted:
        weights_map = load_sampler_weights()
        example_weights = build_example_weights(train_ds["source"], weights_map)
        print(f"   已套 per-source 加權取樣（{len(weights_map)} 來源）")

    sft_config = build_sft_config(args)
    trainer = WeightedSFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        example_weights=example_weights,
        group_by_length=config.GROUP_BY_LENGTH,
        mega_batch_mult=config.LENGTH_GROUP_MEGA_BATCH_MULT,
    )

    print(">> 開始訓練 …")
    trainer.train()

    final_dir = Path(sft_config.output_dir) / "final-adapter"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f">> 完成。adapter 已存：{final_dir}")
    return trainer


if __name__ == "__main__":
    main()
