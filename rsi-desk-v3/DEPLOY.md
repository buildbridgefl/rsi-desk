# Getting a URL for RSI Desk

## Option 1 — Local only (free, 2 minutes, no domain)
```
pip install -r requirements.txt
streamlit run app.py
```
Opens at http://localhost:8501. On your phone, use
http://<your-computer-ip>:8501 over the same wifi. Only works when
your computer is on.

## Option 2 — Streamlit Community Cloud (free, permanent URL, no domain)
1. Push app.py, rsi_backtest.py, requirements.txt to a GitHub repo
2. share.streamlit.io -> New app -> pick the repo -> Deploy
3. You get https://yourname-rsi-desk.streamlit.app

Works on your phone from anywhere. Add it to your home screen and it
behaves like a native app. Free tier sleeps after inactivity and takes
~30s to wake.

Make the repo private if you don't want the code public. Set a password
via .streamlit/secrets.toml if you don't want strangers opening it.

## Option 3 — Custom domain (~$12/year)
Only needed if you want rsidesk.com instead of the free subdomain.
Buy at Cloudflare or Namecheap, then point a CNAME at your host.
Purely cosmetic for a personal tool. Streamlit Cloud does not support
custom domains on the free tier — you'd need Render, Railway, or Fly.io
(~$5-7/mo) for that.

## Recommendation
Start with Option 1 today. Move to Option 2 when you want it on your
phone. Skip Option 3 unless you're putting it in front of other people.
