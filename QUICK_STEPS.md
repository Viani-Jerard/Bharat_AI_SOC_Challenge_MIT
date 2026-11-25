# Quick Steps: Run the Whole Project

## Part A: Google Colab 
1. `sms_spam_detector_colab.ipynb` → Run all → gives `sms_model_artifacts.zip`
2. `cicids_network_threat_detector.ipynb` → Run all → gives `network_artifacts.zip`
3. `llama_export_colab.ipynb` → Run all → gives `llama3_2_1b.pte` + `tokenizer.model` (in Drive → `llama_pte`)

## Part B: Copy to USB pen drive
4. `sms_model_artifacts.zip`
5. `network_artifacts.zip`
6. `llama3_2_1b.pte` and `tokenizer.model`
7. `raspberry-pi-app.zip`

## Part C: Raspberry Pi 5 setup
8. Flash **Raspberry Pi OS 64-bit** with Raspberry Pi Imager (turn on Wi-Fi).
9. Boot the Pi, open **Terminal**, plug in the USB drive.
10. `ls /media/$USER` → note your USB name (use it as `USB` below).

## Part D: Put files in place (type in Terminal)
```
mkdir -p ~/soc/models/network && cd ~/soc
U=/media/$USER/USB
unzip $U/raspberry-pi-app.zip -d app_tmp && cp app_tmp/*/pi_app.py app_tmp/*/benchmark_pi.py app_tmp/*/requirements-pi.txt . 2>/dev/null || cp app_tmp/pi_app.py app_tmp/benchmark_pi.py app_tmp/requirements-pi.txt .
cp $U/llama3_2_1b.pte $U/tokenizer.model models/
unzip $U/sms_model_artifacts.zip -d models/
unzip -j $U/network_artifacts.zip -d models/network/
```

## Part E: Install (one time)
```
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-pi.txt
```

## Part F: Run
11. `cd ~/soc && source venv/bin/activate`
12. `python pi_app.py`
13. `hostname -I` → on your phone open `http://<IP>:7860`
14. SMS tab → type a message → see verdict + Llama explanation.
15. Network tab → pick a sample flow → see attack type + explanation.

## Part G: Report numbers
16. `python benchmark_pi.py` → copy numbers into README section 5.4.



