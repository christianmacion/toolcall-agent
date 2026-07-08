# Deploy — toolcall-agent

**TL;DR:** a public URL in ~5 minutes on Streamlit Community Cloud (free). Works with **no API key** (offline deterministic planner replays full traces); add `ANTHROPIC_API_KEY` for live Claude tool-use.

## 0. Prerequisites
- GitHub account + a Streamlit Community Cloud account ([share.streamlit.io](https://share.streamlit.io), sign in with GitHub — free).
- *(Optional, live mode only)* an Anthropic API key.

## 1. Own public GitHub repo
```bash
cd "06_projects/toolcall-agent"
git init && git add . && git commit -m "toolcall-agent: tool-calling agent with traces + recovery"
gh repo create toolcall-agent --public --source=. --push
```

## 2. Deploy on Streamlit Community Cloud
1. [share.streamlit.io](https://share.streamlit.io) → **Create app** → **Deploy from GitHub**.
2. Repo `<you>/toolcall-agent` · Branch `main` · **Main file path: `app.py`**.
3. *(Optional, live tool-use)* **Advanced settings → Secrets**:
   ```toml
   ANTHROPIC_API_KEY="sk-ant-..."
   ```
   (Streamlit exposes secrets as env vars; the app's `os.getenv` reads it.)
4. **Deploy** → permanent URL `https://<app>.streamlit.app`.

## What a reviewer sees
Pick a task → watch the agent's trace stream **span by span** (thought → tool → args → result → latency → cost). Flip the **fault-injection** toggle and watch retry/backoff actually recover. Open the **Eval** tab: **100% tool-correctness, recovered from 6/6 injected failures, avg 1.2 steps/task** on 20 tasks.

## Run locally
```bash
pip install -r requirements.txt
python -m agent.score        # prints the scorecard, writes scorecard.json
streamlit run app.py
```

## Alternative host: Hugging Face Spaces (SDK: Streamlit). Add the key under **Settings → Variables and secrets** for live mode.

---
*Christian Macion — AI / Agent Engineer.*
