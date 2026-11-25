"""
Scam & Threat Shield - Raspberry Pi 5 app (runs fully on the Pi, no internet).

Uses your Colab outputs UNCHANGED:
  ~/soc/models/scam_detector.joblib          <- sms_model_artifacts.zip  (scikit-learn 1.6.1)
  ~/soc/models/network/network_mlp.pte       <- network_artifacts.zip    (ExecuTorch, no scikit-learn needed)
  ~/soc/models/network/network_meta.json     <- network_artifacts.zip
  ~/soc/models/network/sample_flows.csv      <- network_artifacts.zip
  ~/soc/models/llama3_2_1b.pte               <- Google Drive / llama_pte  (Llama 3.2 1B Instruct)
  ~/soc/models/tokenizer.model               <- Google Drive / llama_pte
(Any sub-folder inside models/ works: the app searches for each file by name.)

Run:   python pi_app.py
Open:  http://<pi-ip-address>:7860  from a phone or laptop on the same Wi-Fi
"""
import json
import os
import re
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psutil
import torch

import executorch.extension.llm.custom_ops.custom_ops  # noqa: F401  (Llama KV-cache / attention ops)
import executorch.kernels.quantized  # noqa: F401  (4-bit kernels)
from executorch.extension.llm.runner import GenerationConfig, TextLLMRunner
from executorch.runtime import Runtime
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

MODELS = Path(__file__).resolve().parent / "models"
PROC = psutil.Process()


def find(name):
    """Find a file anywhere inside models/ (e.g. models/network/...), so folder layout doesn't matter."""
    hits = sorted(MODELS.rglob(name), key=lambda p: len(p.parts))
    return hits[0] if hits else None


def ram_mb():
    return PROC.memory_info().rss / 1e6


# ============================ 1. SMS detector ============================
def clean_text(text):  # identical to the Colab training notebook
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " urltoken ", text)
    text = re.sub(r"[£$₹]|\brs\.?(?=\s?\d)", " moneytoken ", text)
    text = re.sub(r"\b\d{10,13}\b", " phonetoken ", text)
    text = re.sub(r"\b\d{4,6}\b", " shortcodetoken ", text)
    return text


RED_FLAGS = {
    "Contains a link": r"(https?://|www\.|bit\.ly|tinyurl|\b[a-z0-9-]+\.(com|in|co|net|org|xyz|top|info|ly)\b)",
    "Asks for OTP / PIN / password": r"\b(otp|pin|cvv|password|passcode)\b",
    "Mentions KYC / account block": r"\b(kyc|pan card|aadhaa?r|account (will be )?(blocked|suspended|deactivated)|blocked)\b",
    "Mentions UPI / payment request": r"\b(upi|paytm|phonepe|gpay|google pay|collect request)\b",
    "Prize / lottery / free offer": r"\b(won|winner|prize|lottery|jackpot|free|reward|cashback|claim)\b",
    "Urgent language": r"\b(urgent|immediately|now|today only|within 24 ?h(ou)?rs?|last chance|expire[sd]?|act fast)\b",
    "Money amount": r"(₹|rs\.?\s?\d|inr\s?\d|£\s?\d|\$\s?\d|\b\d{3,}\s?(rs|rupees|inr)\b)",
    "Phone / short code to call or text": r"(\b\d{5,6}\b|\b0?\d{10,11}\b|\+\d{10,13})",
}
PRETTY = {"shortcodetoken": "<short number>", "phonetoken": "<phone number>",
          "moneytoken": "<money symbol>", "urltoken": "<link>"}

t0 = time.time()
if not find("scam_detector.joblib"):
    raise SystemExit("models/scam_detector.joblib not found - unzip sms_model_artifacts.zip into models/")
_bundle = joblib.load(find("scam_detector.joblib"))
SMS_MODEL = _bundle["pipeline"]
SMS_NAME = _bundle.get("model_name", "SMS model")
_vec, _clf = SMS_MODEL.named_steps["tfidf"], SMS_MODEL.named_steps["clf"]
_names = np.array(_vec.get_feature_names_out())
if hasattr(_clf, "calibrated_classifiers_"):  # Linear SVM (calibrated) chosen in Colab
    _weights = np.mean([c.estimator.coef_.ravel() for c in _clf.calibrated_classifiers_], axis=0)
elif hasattr(_clf, "coef_"):
    _weights = _clf.coef_.ravel()
else:
    _weights = None
SMS_LOAD_S = time.time() - t0


def check_sms(text, top_k=5):
    t = time.perf_counter()
    x = _vec.transform([clean_text(text)])
    score = float(_clf.predict_proba(x)[0, 1])
    words = []
    if _weights is not None:
        row = x.tocoo()
        contrib = row.data * _weights[row.col]
        for i in np.argsort(-contrib):
            if contrib[i] <= 0 or len(words) >= top_k:
                break
            w = _names[row.col[i]]
            if w.startswith("char__"):
                continue
            w = w.removeprefix("word__")
            if w in ENGLISH_STOP_WORDS:
                continue
            for k, v in PRETTY.items():
                w = w.replace(k, v)
            words.append(w)
    label = "Scam" if score >= 0.5 else "Suspicious" if score >= 0.3 else "Safe"
    flags = [n for n, p in RED_FLAGS.items() if re.search(p, text.lower())]
    return {"text": text, "label": label, "score": score, "trigger_words": words,
            "red_flags": flags, "ms": (time.perf_counter() - t) * 1000}


# ====================== 2. Network detector (ExecuTorch .pte) ======================
t0 = time.time()
for _f in ["network_meta.json", "network_mlp.pte", "sample_flows.csv"]:
    if not find(_f):
        raise SystemExit(f"{_f} not found - unzip network_artifacts.zip into models/network/")
META = json.loads(find("network_meta.json").read_text())
FEATURES, CLASSES, NORMAL = META["features"], META["classes"], META["normal_label"]
MEAN, STD = np.array(META["normal_slog_mean"]), np.array(META["normal_slog_std"])
TYPICAL = np.sign(MEAN) * np.expm1(np.abs(MEAN))
NET = Runtime.get().load_program(str(find("network_mlp.pte"))).load_method("forward")
SAMPLE_FLOWS = pd.read_csv(find("sample_flows.csv"))
NET_LOAD_S = time.time() - t0


def check_flow(row_df):
    x = row_df[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(dtype=np.float32)[:1]
    t = time.perf_counter()
    p = NET.execute([torch.tensor(x)])[0].numpy()[0]
    ms = (time.perf_counter() - t) * 1000
    z = (np.sign(x[0]) * np.log1p(np.abs(x[0].astype(float))) - MEAN) / STD
    k = int(p.argmax())
    top = np.argsort(-np.abs(z))[:4]
    return {"label": CLASSES[k], "confidence": float(p[k]), "is_attack": CLASSES[k] != NORMAL,
            "port": int(x[0, FEATURES.index("Destination Port")]), "ms": ms,
            "unusual": [(FEATURES[j], float(x[0, j]), float(TYPICAL[j])) for j in top]}


# ====================== 3. Llama 3.2 1B (ExecuTorch) ======================
t0 = time.time()
ram_before_llama = ram_mb()
LLAMA = None
if find("llama3_2_1b.pte") and find("tokenizer.model"):
    LLAMA = TextLLMRunner(str(find("llama3_2_1b.pte")), str(find("tokenizer.model")))
LLAMA_LOAD_S = time.time() - t0

SMS_SYSTEM = ("You are a mobile security assistant. A scam detector has already analysed an SMS. "
              "In 2 or 3 short sentences, explain to a non-technical person why the message is or is not "
              "risky, then give one clear action starting with 'Action:'. Do not invent facts.")
NET_SYSTEM = ("You are a network security assistant on a home or small-office router. A detector has already "
              "classified one network flow. In 2 or 3 short sentences, explain to a non-expert what the verdict "
              "means and which measurements look unusual, then give one clear step starting with 'Action:'. "
              "Do not invent facts.")
NET_ACTIONS = {"DoS": "Block or rate-limit the source IP and check the targeted service.",
               "DDoS": "Contact your ISP and enable DDoS protection.",
               "Port Scanning": "Block the scanning IP and close ports you do not use.",
               "Brute Force": "Block the IP, enable account lockout, use strong passwords or SSH keys.",
               "Web Attacks": "Check web logs, patch the website, add a web application firewall.",
               "Bots": "Disconnect the device, scan it for malware and change its passwords."}


def chat(system, user):
    return ("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n" + system +
            "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n" + user +
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")


def ask_llama(system, user, max_new_tokens=80):
    """Returns (text, info dict with seconds / tokens / tokens_per_s / first_token_s)."""
    LLAMA.reset()
    pieces, info = [], {}

    def on_stats(s):
        info["tokens"] = s.num_generated_tokens
        info["prompt_tokens"] = s.num_prompt_tokens
        info["first_token_s"] = (s.first_token_ms - s.inference_start_ms) / 1000
        dt = (s.inference_end_ms - s.prompt_eval_end_ms) / 1000
        info["tokens_per_s"] = s.num_generated_tokens / dt if dt > 0 else 0.0

    t = time.time()
    LLAMA.generate(chat(system, user), GenerationConfig(echo=False, max_new_tokens=max_new_tokens, temperature=0.0),
                   token_callback=pieces.append, stats_callback=on_stats)
    info["seconds"] = time.time() - t
    text = re.sub(r"<\|[a-z_]+\|>", "", "".join(pieces).split("<|eot_id|>")[0]).strip()
    return text, info


def explain_sms(r):
    if LLAMA:
        user = (f'SMS: "{r["text"][:400]}"\nDetector verdict: {r["label"]} (scam probability {r["score"]:.0%})\n'
                f'Words that raised the score: {", ".join(r["trigger_words"]) or "none"}\n'
                f'Red flags found: {", ".join(r["red_flags"]) or "none"}\nExplain and recommend an action.')
        text, info = ask_llama(SMS_SYSTEM, user)
        if text:
            return text, info
    if r["label"] == "Safe":
        return "This message looks safe.\nAction: No action needed. Never share an OTP or PIN.", {}
    return ("This message looks like a scam: " + ", ".join(r["red_flags"][:3]).lower() + ".\n"
            "Action: Do not click links, call back or share OTP/PIN. Report it at sancharsaathi.gov.in or call 1930."), {}


def explain_flow(r):
    if LLAMA:
        feats = "; ".join(f"{n} = {v:,.0f} (normal is about {t:,.0f})" for n, v, t in r["unusual"])
        user = (f"Destination port: {r['port']}\nDetector verdict: {r['label']} (confidence {r['confidence']:.0%})\n"
                f"Most unusual measurements compared with normal traffic: {feats}\nExplain and recommend an action.")
        text, info = ask_llama(NET_SYSTEM, user)
        if text:
            return text, info
    if not r["is_attack"]:
        return f"Normal traffic on port {r['port']}.\nAction: No action needed.", {}
    return f"{r['label']} detected on port {r['port']}.\nAction: {NET_ACTIONS.get(r['label'], 'Investigate.')}", {}


def perf_line(det_ms, info):
    s = f"Detector {det_ms:.1f} ms"
    if info:
        s += (f" · Llama {info['seconds']:.1f} s ({info['tokens']} tokens, {info['tokens_per_s']:.1f} tokens/s, "
              f"first token {info['first_token_s']:.1f} s)")
    else:
        s += " · template explanation (Llama not loaded)"
    return s + f" · RAM {ram_mb():.0f} MB · on Raspberry Pi, offline"


print(f"Loaded SMS model ({SMS_NAME}) in {SMS_LOAD_S:.1f}s, network .pte in {NET_LOAD_S:.1f}s, "
      f"Llama {'in %.1fs' % LLAMA_LOAD_S if LLAMA else 'NOT FOUND'} | RAM {ram_mb():.0f} MB")

# ============================ 4. Web UI (Gradio) ============================
SAMPLE_SMS = [
    "Dear customer, your SBI account KYC is pending and will be blocked today. Update now: http://sbi-kyc-update.xyz",
    "Congratulations! You have won Rs 25,00,000 in KBC lottery. Call 9876543210 to claim your prize.",
    "Your electricity connection will be disconnected tonight at 9.30pm. Contact officer 8123456789 immediately.",
    "Hi amma, I'll reach home by 7. Keep dinner ready.",
]


def ui_sms(text):
    if not text or not text.strip():
        return "Please paste a message.", "", ""
    r = check_sms(text)
    exp, info = explain_sms(r)
    icon = {"Scam": "🚨", "Suspicious": "⚠️", "Safe": "✅"}[r["label"]]
    verdict = (f"## {icon} {r['label']} ({r['score']:.0%} scam probability)\n"
               f"**Trigger words:** {', '.join(r['trigger_words']) or 'none'}\n\n"
               f"**Red flags:** {', '.join(r['red_flags']) or 'none'}")
    return verdict, exp, perf_line(r["ms"], info)


def ui_net(idx, file):
    df = pd.read_csv(file) if file else SAMPLE_FLOWS
    idx = int(min(max(idx or 0, 0), len(df) - 1))
    r = check_flow(df.iloc[[idx]])
    exp, info = explain_flow(r)
    truth = f" · true label: {df['Attack Type'].iloc[idx]}" if "Attack Type" in df.columns else ""
    verdict = f"## {'🚨' if r['is_attack'] else '✅'} {r['label']} ({r['confidence']:.0%})\nPort {r['port']}{truth}"
    table = pd.DataFrame(r["unusual"], columns=["Measurement", "This flow", "Normal is about"]).round(1)
    return verdict, table, exp, perf_line(r["ms"], info)


if __name__ == "__main__":
    import gradio as gr

    with gr.Blocks(title="Scam & Threat Shield") as demo:
        gr.Markdown("# 🛡️ Scam & Threat Shield\nRuns on this **Raspberry Pi 5**, offline. "
                    "Detector decides, **Llama 3.2 1B on ExecuTorch** explains and recommends.")
        with gr.Tab("📱 SMS"):
            inp = gr.Textbox(label="SMS text", lines=4, value=SAMPLE_SMS[0])
            gr.Examples(SAMPLE_SMS, inp)
            b1 = gr.Button("Check", variant="primary")
            o1, o2, o3 = gr.Markdown(), gr.Textbox(label="🦙 Why, and what to do", lines=5), gr.Markdown()
            b1.click(ui_sms, inp, [o1, o2, o3])
        with gr.Tab("🌐 Network"):
            n = gr.Number(label=f"Sample flow number (0–{len(SAMPLE_FLOWS) - 1})", value=0, precision=0)
            f = gr.File(label="Or upload a flows CSV (same 52 columns)", file_types=[".csv"])
            b2 = gr.Button("Check", variant="primary")
            p1, p2, p3, p4 = (gr.Markdown(), gr.Dataframe(label="Most unusual measurements"),
                              gr.Textbox(label="🦙 Why, and what to do", lines=5), gr.Markdown())
            b2.click(ui_net, [n, f], [p1, p2, p3, p4])
    # 0.0.0.0 = reachable from other devices on the same Wi-Fi
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
