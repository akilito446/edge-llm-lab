# edge-llm-lab

Measuring small LLMs on a laptop GPU (GTX 1650, 4 GB) with llama.cpp.

## Run a benchmark

```powershell
.\.venv\Scripts\Activate.ps1
python bench.py --model $HOME\models\qwen2.5-3b-q4_k_m.gguf --ngl 99 --note "baseline"
```

Each run appends one row to `results.csv`.

## What each column means

| Column | Meaning |
| --- | --- |
| load_s | Seconds until the model is loaded and ready |
| ttft_ms | Time to first token: request sent → first token back |
| prefill_tps | Prompt tokens read per second |
| decode_tps | New tokens generated per second |
| vram_mb | Extra GPU memory while the model is loaded |
| ram_mb | Peak RAM of the llama-server process |

## Results

| Model | Quant | ngl | decode tok/s | prefill tok/s | TTFT ms | VRAM MB |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-3B Instruct | Q4_K_M | 99 | 44.1 | 178.2 | 963 | 2131 |

Hardware: laptop, NVIDIA GTX 1650 (4 GB), AMD Ryzen 7 4800H, 16 GB RAM, Windows.
Runtime: llama.cpp b11669, CUDA 12.4 build. Context 4096, 128 generated tokens, median of 3 runs.

## Findings so far

- Decode is memory-bandwidth bound: 44.1 tok/s × 2.1 GB of weights ≈ 93 GB/s moved per second.
- Prefill is the slow part on this card (no tensor cores): a 169-token prompt takes ~0.95 s, which is almost all of the time to first token.
