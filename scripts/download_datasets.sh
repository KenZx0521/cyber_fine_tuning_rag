#!/usr/bin/env bash
#
# download_datasets.sh — Fetch & organise the cybersecurity corpus described in dataset.md
#
# Layout produced under data/:
#   fine-tuning/  -> training data (Primus-Instruct, AttackQA)
#   rag/          -> knowledge base, kept fresh by re-running (MITRE ATT&CK, NVD, CISA KEV, CWE)
#   eval/         -> held-out benchmarks, NEVER train on these (SecQA, CyberSecEval)
#
# Re-run any time to re-sync the RAG feeds (NVD updates daily, KEV on change).
#
# Gated dataset (Primus-Instruct) requires a Hugging Face token with the dataset
# terms accepted. Provide it via the environment — it is NEVER stored in this file:
#   export HF_TOKEN=hf_xxx          # token for account that accepted the gate
#   ./scripts/download_datasets.sh
#
# Optional: pass one or more groups to download only those:
#   ./scripts/download_datasets.sh rag            # only RAG sources
#   ./scripts/download_datasets.sh fine-tuning eval
#
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="$ROOT/data"
NVD_BASE="https://nvd.nist.gov/feeds/json/cve/2.0"
NVD_FIRST_YEAR=2002
NVD_LAST_YEAR="$(date +%Y)"

log()  { printf '\033[1;36m[%s]\033[0m %s\n' "$(date +%H:%M:%S)" "$*"; }
ok()   { printf '  \033[1;32mOK\033[0m   %s\n' "$*"; }
fail() { printf '  \033[1;31mFAIL\033[0m %s\n' "$*"; }

have() { command -v "$1" >/dev/null 2>&1; }

# Resolve a Hugging Face CLI; the modern binary is `hf`, older installs expose `huggingface-cli`.
hf_cli() {
  if have hf; then hf "$@";
  elif have huggingface-cli; then huggingface-cli "$@";
  else fail "no Hugging Face CLI found (pip install -U huggingface_hub)"; return 1; fi
}

hf_pull() {  # $1=repo  $2=dest
  HF_HUB_DISABLE_TELEMETRY=1 hf_cli download "$1" --repo-type dataset --local-dir "$2"
}

# git clone or fast-forward an existing shallow checkout.
git_sync() {  # $1=url  $2=dest  [extra clone args...]
  local url="$1" dest="$2"; shift 2
  if [ -d "$dest/.git" ]; then
    git -C "$dest" pull --ff-only --depth 1 2>/dev/null || git -C "$dest" fetch --depth 1
  else
    git clone --depth 1 "$@" "$url" "$dest"
  fi
}

# ----------------------------------------------------------------------------- fine-tuning
fetch_finetuning() {
  log "fine-tuning datasets"
  mkdir -p "$DATA/fine-tuning"

  # Primus-Instruct (GATED — needs HF_TOKEN whose account accepted the terms at
  # https://huggingface.co/datasets/trendmicro-ailab/Primus-Instruct)
  if [ -n "${HF_TOKEN:-}" ]; then
    if HF_TOKEN="$HF_TOKEN" hf_pull "trendmicro-ailab/Primus-Instruct" "$DATA/fine-tuning/primus-instruct"; then
      ok "Primus-Instruct"
    else
      fail "Primus-Instruct (accept the gate on the dataset page, then retry)"
    fi
  else
    fail "Primus-Instruct skipped — set HF_TOKEN to download the gated dataset"
  fi

  # AttackQA (open)
  hf_pull "sambanovasystems/attackqa" "$DATA/fine-tuning/attackqa" && ok "AttackQA" || fail "AttackQA"
}

# ----------------------------------------------------------------------------- RAG
fetch_rag() {
  log "RAG knowledge-base sources"
  mkdir -p "$DATA/rag/nvd" "$DATA/rag/cwe" "$DATA/rag/cisa-kev"

  # MITRE ATT&CK — latest STIX 2.1 collections (Enterprise, Mobile, ICS) + index.
  # Direct download of the current collections is far smaller/faster than cloning
  # the full repo, which carries every historical ATT&CK version (~128 MB of noise
  # for RAG). Re-running overwrites with the freshest collections.
  mkdir -p "$DATA/rag/mitre-attack-stix"
  local mitre_ok=1 raw="https://raw.githubusercontent.com/mitre-attack/attack-stix-data/master"
  for col in enterprise-attack mobile-attack ics-attack; do
    curl -sSfL -o "$DATA/rag/mitre-attack-stix/${col}.json" "$raw/${col}/${col}.json" || mitre_ok=0
  done
  curl -sSfL -o "$DATA/rag/mitre-attack-stix/index.json" "$raw/index.json" || mitre_ok=0
  [ "$mitre_ok" = 1 ] && ok "MITRE ATT&CK STIX (latest collections)" || fail "MITRE ATT&CK STIX"

  # CISA KEV — git mirror (CSV/JSON/schema) + the live catalog feed
  git_sync https://github.com/cisagov/kev-data.git "$DATA/rag/cisa-kev" \
    && ok "CISA KEV mirror" || fail "CISA KEV mirror"
  curl -sSfL -o "$DATA/rag/cisa-kev/known_exploited_vulnerabilities.json" \
    https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json \
    && ok "CISA KEV live catalog" || fail "CISA KEV live catalog"

  # CWE — full weakness catalog (XML) + comprehensive view (CSV)
  curl -sSfL -o "$DATA/rag/cwe/cwec_latest.xml.zip" \
    https://cwe.mitre.org/data/xml/cwec_latest.xml.zip && ok "CWE XML" || fail "CWE XML"
  curl -sSfL -o "$DATA/rag/cwe/cwe-comprehensive-2000.csv.zip" \
    https://cwe.mitre.org/data/csv/2000.csv.zip && ok "CWE CSV" || fail "CWE CSV"

  # NVD — JSON 2.0 year feeds (one-time bulk) + modified/recent (incremental sync)
  local yr n=0
  for (( yr=NVD_FIRST_YEAR; yr<=NVD_LAST_YEAR; yr++ )); do
    if curl -sSfO --output-dir "$DATA/rag/nvd" "$NVD_BASE/nvdcve-2.0-${yr}.json.gz"; then
      curl -sSfO --output-dir "$DATA/rag/nvd" "$NVD_BASE/nvdcve-2.0-${yr}.meta" 2>/dev/null
      n=$((n+1))
    fi
  done
  curl -sSfO --output-dir "$DATA/rag/nvd" "$NVD_BASE/nvdcve-2.0-modified.json.gz" 2>/dev/null
  curl -sSfO --output-dir "$DATA/rag/nvd" "$NVD_BASE/nvdcve-2.0-recent.json.gz"   2>/dev/null
  [ "$n" -gt 0 ] && ok "NVD ($n year feeds + modified/recent)" || fail "NVD feeds"
}

# ----------------------------------------------------------------------------- eval
fetch_eval() {
  log "eval / benchmark datasets (DO NOT TRAIN ON THESE)"
  mkdir -p "$DATA/eval"

  # SecQA — multiple-choice security knowledge (open)
  hf_pull "zefang-liu/secqa" "$DATA/eval/secqa" && ok "SecQA" || fail "SecQA"

  # CyberSecEval — sparse checkout of just CybersecurityBenchmarks from PurpleLlama
  local dest="$DATA/eval/cyberseceval-purplellama"
  if [ -d "$dest/.git" ]; then
    git -C "$dest" pull --ff-only 2>/dev/null || true
  else
    git clone --depth 1 --filter=blob:none --sparse \
      https://github.com/meta-llama/PurpleLlama.git "$dest" \
      && git -C "$dest" sparse-checkout set CybersecurityBenchmarks
  fi
  [ -d "$dest/CybersecurityBenchmarks" ] && ok "CyberSecEval" || fail "CyberSecEval"
}

# ----------------------------------------------------------------------------- main
main() {
  local groups=("$@")
  [ "${#groups[@]}" -eq 0 ] && groups=(fine-tuning rag eval)
  for g in "${groups[@]}"; do
    case "$g" in
      fine-tuning) fetch_finetuning ;;
      rag)         fetch_rag ;;
      eval)        fetch_eval ;;
      *) fail "unknown group: $g (use: fine-tuning | rag | eval)" ;;
    esac
  done
  log "done — see data/ for the organised corpus"
}

main "$@"
