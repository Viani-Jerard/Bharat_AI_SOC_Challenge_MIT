# Raspberry Pi 5: step by step (uses your files unchanged)

**What you need**
- Raspberry Pi 5 (4 GB or 8 GB), power supply, **32 GB+ microSD card**, a fan or heatsink
- Your files:
  - `sms_model_artifacts.zip` (from the SMS notebook)
  - `network_artifacts.zip` (from the CICIDS notebook)
  - `llama3_2_1b.pte` and `tokenizer.model` (Google Drive → `llama_pte`)
  - `pi_app.py`, `benchmark_pi.py`, `requirements-pi.txt` (this folder)

Final layout on the Pi:
```
~/soc/
├── pi_app.py
├── benchmark_pi.py
├── requirements-pi.txt
└── models/
    ├── scam_detector.joblib        (from sms_model_artifacts.zip)
    ├── llama3_2_1b.pte
    ├── tokenizer.model
    └── network/                    (unzip network_artifacts.zip here)
        └── ... network_mlp.pte, network_meta.json, sample_flows.csv ...
```
The app searches inside `models/` for each file by name, so an extra sub-folder from unzipping is fine.

---

## Part 1: Set up the Pi (skip if it already boots to a desktop)

1. On your laptop, install **Raspberry Pi Imager** from raspberrypi.com/software.
2. Put the microSD card in your laptop and open Imager.
3. Click **Choose Device → Raspberry Pi 5**.
4. Click **Choose OS → Raspberry Pi OS (64-bit)**. It must be **64-bit**.
5. Click **Choose Storage** and pick your SD card.
6. Click **Next → Edit settings**. Set a username (for example `pi`) and password, and enter your **Wi-Fi name and password**. On the **Services** tab, tick **Enable SSH**. Click **Save → Yes**.
7. When it finishes, put the card in the Pi and switch it on. Wait about 2 minutes.

---

## Part 2: Copy your files to the Pi

**Easiest (USB stick):**
1. Copy all the files listed above onto a USB stick.
2. Plug it into the Pi. It appears under `/media/<your-username>/<stick-name>/`.

Then open **Terminal** on the Pi (black icon, top bar) and run these one at a time (change `STICK` to your stick's name):
```bash
mkdir -p ~/soc/models/network
cp /media/$USER/STICK/pi_app.py /media/$USER/STICK/benchmark_pi.py /media/$USER/STICK/requirements-pi.txt ~/soc/
cp /media/$USER/STICK/llama3_2_1b.pte /media/$USER/STICK/tokenizer.model ~/soc/models/
unzip -o /media/$USER/STICK/sms_model_artifacts.zip -d ~/soc/models/
unzip -o /media/$USER/STICK/network_artifacts.zip -d ~/soc/models/network/
```

*(Alternative from a Windows laptop on the same Wi-Fi: `scp llama3_2_1b.pte pi@raspberrypi.local:~/soc/models/`.)*

Check that everything is there:
```bash
find ~/soc -name "*.pte" -o -name "*.joblib" -o -name "*.json" -o -name "*.model" -o -name "*.csv"
```
You should see `llama3_2_1b.pte`, `tokenizer.model`, `scam_detector.joblib`, `network_mlp.pte`, `network_meta.json` and `sample_flows.csv`.

---

## Part 3: Install the software (one time, 10–20 minutes)

Run these in the Pi Terminal, **one line at a time**:

1. Go to the folder:
   ```bash
   cd ~/soc
   ```
2. Create a Python environment (skip if you already made one; just activate it in the next step):
   ```bash
   python3 -m venv venv
   ```
3. Turn it on. You'll see `(venv)` at the start of the line:
   ```bash
   source venv/bin/activate
   ```
4. Install the small CPU-only PyTorch first (avoids a 3 GB GPU download the Pi can't use):
   ```bash
   pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
   ```
   If this line gives an error, skip it. Step 5 will still work; it just downloads more.
5. Install everything else:
   ```bash
   pip install -r requirements-pi.txt
   ```
6. Check it worked:
   ```bash
   python -c "import sklearn, executorch, gradio; print('OK, scikit-learn', sklearn.__version__)"
   ```
   It must say **scikit-learn 1.6.1**, the same version your SMS model was trained with.

> We checked that executorch 1.6.0, torch 2.15.0 and scikit-learn 1.6.1 all have Raspberry Pi (aarch64) builds for Python 3.11 and 3.13. On the Pi, executorch does **not** force a newer scikit-learn (it only does that on x86 PCs), so 1.6.1 stays.

---

## Part 4: Run the app

1. In the Pi Terminal (with `(venv)` showing):
   ```bash
   cd ~/soc && python pi_app.py
   ```
2. Wait until you see `Loaded SMS model … Llama in …s` and `Running on local URL: http://0.0.0.0:7860`. Loading Llama takes a few seconds.
3. On the Pi, open the browser and go to **http://localhost:7860**.
4. Click **Check**. The first Llama answer is slower; after that it's quicker.

**Open it from your phone or laptop (same Wi-Fi):**
1. In a **second** Pi Terminal, run `hostname -I`. It prints something like `192.168.1.42`.
2. On your phone, open **http://192.168.1.42:7860** (use your number).

At the bottom of each result the app shows **detector time, Llama time, tokens/second and RAM**. That's live proof it runs on the Pi.

To stop the app, press **Ctrl + C** in the Terminal.

---

## Part 5: Measure for the report

1. Stop the app (Ctrl + C), then run:
   ```bash
   cd ~/soc && source venv/bin/activate && python benchmark_pi.py
   ```
2. It takes 1–3 minutes and prints a **RESULTS** block like this:
   ```
   SMS detector:      … ms mean
   Network detector:  … ms mean (ExecuTorch .pte)
   Llama 3.2 1B:      … tokens/s, first token … s, … s per explanation
   RAM used:          … MB
   CPU temperature:   … -> …
   ```
3. Copy those numbers into your report's results table. Everything is also saved in `~/soc/results_pi.json`.
4. Take a photo or screen recording of the app running on the Pi for the demo video.

---

## If something goes wrong

| Message | Fix |
|---|---|
| `scam_detector.joblib not found` | Run the `unzip … sms_model_artifacts.zip -d ~/soc/models/` line again |
| `network_meta.json not found` | Run the `unzip … network_artifacts.zip -d ~/soc/models/network/` line again |
| Sidebar says Llama **NOT FOUND** | Check `ls ~/soc/models` shows `llama3_2_1b.pte` and `tokenizer.model` |
| `ModuleNotFoundError` | You forgot `source venv/bin/activate`. Run it, then try again |
| scikit-learn warning "InconsistentVersionWarning" | Run `pip install scikit-learn==1.6.1` |
| `No matching distribution found for executorch` | You're on 32-bit Raspberry Pi OS. Re-flash with **64-bit** (Part 1) |
| Pi freezes / very slow | Use the 8 GB Pi, close the browser on the Pi and open the app from your phone instead, add a fan |
| Phone can't open the page | Phone and Pi must be on the same Wi-Fi; check the IP with `hostname -I` |
