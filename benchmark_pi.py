"""
Measure speed and memory on the Raspberry Pi for the technical report.

    python benchmark_pi.py

Prints a table and saves results_pi.json next to this file.
"""
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path

import psutil

ram_start = psutil.Process().memory_info().rss / 1e6
t0 = time.time()
import pi_app as app  # noqa: E402  (loads all three models)

load_total = time.time() - t0


def cpu_temp():
    try:
        return subprocess.run(["vcgencmd", "measure_temp"], capture_output=True, text=True).stdout.strip()
    except Exception:
        return "n/a"


res = {
    "device": {"machine": platform.machine(), "python": platform.python_version(), "cpu_count": psutil.cpu_count(),
               "total_ram_gb": round(psutil.virtual_memory().total / 1e9, 2), "os": platform.platform(),
               "temp_start": cpu_temp()},
    "load_seconds": {"sms_model": round(app.SMS_LOAD_S, 2), "network_pte": round(app.NET_LOAD_S, 2),
                     "llama_pte": round(app.LLAMA_LOAD_S, 2), "all": round(load_total, 2)},
    "file_mb": {f.name: round(f.stat().st_size / 1e6, 2) for f in app.MODELS.rglob("*") if f.is_file()},
}

# SMS detector: 200 runs
msgs = (app.SAMPLE_SMS * 50)[:200]
app.check_sms(msgs[0])
sms_ms = [app.check_sms(m)["ms"] for m in msgs]
res["sms_detector_ms"] = {"mean": round(statistics.mean(sms_ms), 2), "p95": round(sorted(sms_ms)[189], 2)}

# Network detector (.pte): all sample flows
app.check_flow(app.SAMPLE_FLOWS.iloc[[0]])  # warm-up
flows = [app.check_flow(app.SAMPLE_FLOWS.iloc[[i]]) for i in range(len(app.SAMPLE_FLOWS))]
net_ms = [f["ms"] for f in flows]
correct = sum(f["label"] == t for f, t in zip(flows, app.SAMPLE_FLOWS["Attack Type"]))
res["network_detector_ms"] = {"mean": round(statistics.mean(net_ms), 3), "p95": round(sorted(net_ms)[int(0.95 * len(net_ms))], 3),
                              "sample_accuracy": f"{correct}/{len(flows)}"}

# Llama: 3 SMS + 2 network explanations
runs = []
if app.LLAMA:
    for s in app.SAMPLE_SMS[:3]:
        text, info = app.explain_sms(app.check_sms(s))
        runs.append({"type": "sms", **{k: round(v, 2) for k, v in info.items()}, "text": text})
    for i in [0, 20]:
        text, info = app.explain_flow(app.check_flow(app.SAMPLE_FLOWS.iloc[[i]]))
        runs.append({"type": "network", **{k: round(v, 2) for k, v in info.items()}, "text": text})
    res["llama"] = {
        "seconds_mean": round(statistics.mean(r["seconds"] for r in runs), 2),
        "tokens_per_s_mean": round(statistics.mean(r["tokens_per_s"] for r in runs), 2),
        "first_token_s_mean": round(statistics.mean(r["first_token_s"] for r in runs), 2),
        "runs": runs,
    }
res["ram_mb"] = {"python_start": round(ram_start, 0), "after_everything": round(app.ram_mb(), 0)}
res["device"]["temp_end"] = cpu_temp()

Path(__file__).with_name("results_pi.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
print("\n================ RESULTS (copy into your report) ================")
print(f"Device: {res['device']['machine']}, {res['device']['total_ram_gb']} GB RAM, Python {res['device']['python']}")
print(f"SMS detector:      {res['sms_detector_ms']['mean']} ms mean, {res['sms_detector_ms']['p95']} ms p95")
print(f"Network detector:  {res['network_detector_ms']['mean']} ms mean (ExecuTorch .pte), sample accuracy {res['network_detector_ms']['sample_accuracy']}")
if app.LLAMA:
    print(f"Llama 3.2 1B:      {res['llama']['tokens_per_s_mean']} tokens/s, first token {res['llama']['first_token_s_mean']} s, "
          f"{res['llama']['seconds_mean']} s per explanation")
print(f"Model load time:   {res['load_seconds']}")
print(f"RAM used:          {res['ram_mb']['after_everything']} MB")
print(f"CPU temperature:   {res['device']['temp_start']} -> {res['device']['temp_end']}")
print("Saved results_pi.json")
