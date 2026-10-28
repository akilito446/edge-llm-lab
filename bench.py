"""
edge-llm-lab benchmark: measure one GGUF model on llama-server.

What it measures, and how:
  load_s        seconds from launching llama-server until /health says the model is ready
  ttft_ms       wall-clock time from sending the request to the first streamed token
  prefill_tps   prompt tokens processed per second   (llama-server's own timer)
  decode_tps    generated tokens per second          (llama-server's own timer)
  vram_mb       extra GPU memory used while the model is loaded (peak minus baseline)
  ram_mb        peak resident memory of the llama-server process

Usage (PowerShell, from the repo folder, venv active):
  python bench.py --model $HOME\\models\\qwen2.5-3b-q4_k_m.gguf --ngl 99
"""

import argparse
import csv
import json
import os
import shutil
import statistics
import subprocess
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

import psutil

# A fixed prompt, so every run and every model gets the same prefill work.
PROMPT = (
    "You are the onboard assistant of an agricultural robot. "
    "Explain, step by step, how the robot should decide whether a tomato is ripe "
    "enough to harvest, using colour, size and firmness readings from its sensors. "
) * 4


def gpu_mem_used_mb():
    """Total memory used on NVIDIA GPU 0, in MB, or None without nvidia-smi.

    On Windows, nvidia-smi cannot report memory per process, so we read the
    total for the card and subtract a baseline taken before the server starts.
    """
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", "0"],
            capture_output=True, text=True, timeout=5,
        )
        return float(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


class PeakSampler(threading.Thread):
    """Polls RAM of one process and total GPU memory every `every` seconds, keeping the peaks."""

    def __init__(self, pid, every=0.25):
        super().__init__(daemon=True)
        self.proc = psutil.Process(pid)
        self.every = every
        self.peak_ram_mb = 0.0
        self.peak_vram_mb = None
        self._stop = threading.Event()

    def run(self):
        while not self._stop.is_set():
            try:
                rss = self.proc.memory_info().rss / 2**20
                self.peak_ram_mb = max(self.peak_ram_mb, rss)
            except psutil.Error:
                break
            vram = gpu_mem_used_mb()
            if vram is not None:
                self.peak_vram_mb = vram if self.peak_vram_mb is None else max(self.peak_vram_mb, vram)
            self._stop.wait(self.every)

    def stop(self):
        self._stop.set()
        self.join()


def wait_until_ready(url, proc, timeout_s):
    start = time.perf_counter()
    while time.perf_counter() - start < timeout_s:
        if proc.poll() is not None:
            raise RuntimeError("llama-server exited during load; see server.log")
        try:
            with urllib.request.urlopen(url + "/health", timeout=2) as r:
                if r.status == 200:
                    return time.perf_counter() - start
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass  # 503 while loading, or not listening yet
        time.sleep(0.2)
    raise TimeoutError(f"model not ready after {timeout_s} s")


def one_run(url, n_predict):
    """Send one streamed request. Returns (ttft_ms, timings dict from the server)."""
    body = json.dumps({
        "prompt": PROMPT,
        "n_predict": n_predict,
        "stream": True,
        "cache_prompt": False,   # re-process the prompt every run, or prefill looks free
        "ignore_eos": True,      # always generate exactly n_predict tokens
        "temperature": 0,
    }).encode()
    req = urllib.request.Request(url + "/completion", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft_ms, timings = None, None
    try:
        r = urllib.request.urlopen(req, timeout=600)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"server refused the request ({e.code}): {e.read().decode()[:300]}") from None
    with r:
        for raw in r:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            chunk = json.loads(line[5:])
            if ttft_ms is None and chunk.get("content"):
                ttft_ms = (time.perf_counter() - t0) * 1000
            if chunk.get("stop"):
                timings = chunk.get("timings")
    if timings is None:
        raise RuntimeError("server sent no timings; check the llama.cpp version")
    return ttft_ms, timings


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="path to a .gguf file")
    p.add_argument("--llama-dir", default=str(Path.home() / "llama.cpp"),
                   help="folder holding llama-server(.exe)")
    p.add_argument("--ngl", type=int, default=99, help="layers on the GPU (99 = all)")
    p.add_argument("--ctx", type=int, default=4096, help="context size")
    p.add_argument("--n-predict", type=int, default=128, help="tokens to generate per run")
    p.add_argument("--runs", type=int, default=3, help="measured runs (after 1 warm-up)")
    p.add_argument("--port", type=int, default=8090)
    p.add_argument("--out", default="results.csv", help="CSV file to append a row to")
    p.add_argument("--note", default="", help="free text saved with the row")
    a = p.parse_args()

    exe = Path(a.llama_dir) / ("llama-server.exe" if os.name == "nt" else "llama-server")
    if not exe.exists():
        raise SystemExit(f"not found: {exe}  (use --llama-dir)")
    if not Path(a.model).exists():
        raise SystemExit(f"not found: {a.model}")

    url = f"http://127.0.0.1:{a.port}"
    base_vram = gpu_mem_used_mb()

    log = open("server.log", "w")
    proc = subprocess.Popen(
        [str(exe), "-m", a.model, "-ngl", str(a.ngl), "-c", str(a.ctx),
         "--host", "127.0.0.1", "--port", str(a.port)],
        stdout=log, stderr=subprocess.STDOUT,
    )
    sampler = PeakSampler(proc.pid)
    sampler.start()
    try:
        print(f"loading {Path(a.model).name} (ngl={a.ngl}, ctx={a.ctx}) ...")
        load_s = wait_until_ready(url, proc, timeout_s=300)
        print(f"ready in {load_s:.1f} s; warm-up run ...")
        one_run(url, n_predict=16)

        ttfts, prefill, decode = [], [], []
        for i in range(a.runs):
            ttft, tm = one_run(url, a.n_predict)
            ttfts.append(ttft)
            prefill.append(tm["prompt_per_second"])
            decode.append(tm["predicted_per_second"])
            print(f"  run {i + 1}: ttft {ttft:7.1f} ms | prefill {tm['prompt_per_second']:8.1f} tok/s "
                  f"({tm['prompt_n']} tok) | decode {tm['predicted_per_second']:6.1f} tok/s "
                  f"({tm['predicted_n']} tok)")
    finally:
        sampler.stop()
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()

    vram_mb = None
    if sampler.peak_vram_mb is not None and base_vram is not None:
        vram_mb = sampler.peak_vram_mb - base_vram

    row = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "model": Path(a.model).name,
        "ngl": a.ngl,
        "ctx": a.ctx,
        "n_predict": a.n_predict,
        "runs": a.runs,
        "load_s": round(load_s, 2),
        "ttft_ms": round(statistics.median(ttfts), 1),
        "prefill_tps": round(statistics.median(prefill), 1),
        "decode_tps": round(statistics.median(decode), 1),
        "vram_mb": None if vram_mb is None else round(vram_mb),
        "ram_mb": round(sampler.peak_ram_mb),
        "note": a.note,
    }

    new_file = not Path(a.out).exists()
    with open(a.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new_file:
            w.writeheader()
        w.writerow(row)

    print("\nmedian of", a.runs, "runs")
    for k in ("load_s", "ttft_ms", "prefill_tps", "decode_tps", "vram_mb", "ram_mb"):
        print(f"  {k:12} {row[k]}")
    print(f"appended to {a.out}")


if __name__ == "__main__":
    main()
