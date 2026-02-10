# Deploying FinOracle to Streamlit Cloud: The Roadmap

To make this project work on a cloud platform like Streamlit Cloud (where you cannot control the machine), you need to change three key things: Authentication, Environment, and (ideally) Architecture.

## 1. 🔑 Authentication (The Hardest Part)
**Current:** Uses `DesktopSession` (requires Eikon/Workspace app running on the PC).
**Required:** `PlatformSession` (API Key + Machine ID + Password).

*   **Action:** You must switch `fetch_data.py` to use `rd.open_session(app_key=..., ...)` with environment variables for secrets.
*   **Limitation:** You need a Refinitiv license that allows **Data Platform (RDP)** access, not just Desktop access.

## 2. 📦 Environment (No More Subprocess)
**Current:** Main app calls a specific `.venv` python executable.
**Required:** Single environment defined by `requirements.txt`.

*   **Action:**
    1.  Merge `Forecast/FinCast-fts/requirements.txt` into the root `requirements.txt`.
    2.  Rewrite `finoracle_wrapper.py` to **import** `finoracle_bridge` directly instead of running it as a subprocess.
    3.  Delete the `.venv` folder from the project (git ignore it).

## 3. 🏗️ Architecture (Heavy Lifting)
**Current:** FinCast loads a heavy PyTorch model into memory on demand.
**Problem:** Streamlit Cloud has memory limits (often ~1-3GB). FinCast + ECE might crash it.

*   **Solution A (Easy but Risky):** Try to run it all-in-one. If it crashes, you need more RAM.
*   **Solution B (Pro):**host the **FinCast Inference API** separately (e.g., on a GPU instance like Modal, AWS Lambda, or a dedicated VPS) and have your Streamlit app just send a JSON request to it.

## 📝 Immediate "To-Do" for a Cloud Demo
If you just want to show it to someone else **on their machine**:
1.  They need to install Python & Git.
2.  They need to clone your repo.
3.  They need to run `install.bat` (which you would create) to set up the `.venv` locally.
4.  They need their own Refinitiv Workspace running.

For **true** web hosting, you strictly need to solve **#1 (Auth)** and **#2 (Env)** first.
