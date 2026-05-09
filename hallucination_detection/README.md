# LLM-as-a-Judge Demo

Live demo of the Trajel hallucination judge. Runs on Google Colab — no local setup needed.

---

## Setup

### 1. Open the notebook in Colab

Download `demo.ipynb` from this folder and upload it to [colab.research.google.com](https://colab.research.google.com), or open it directly:

**colab.research.google.com/github/aishani-rachakonda/genai-using-llms-final-project/blob/main/hallucination_detection/demo.ipynb**

### 2. Upload `llm_judge_prompt.py`

In Colab, click the **Files** icon (📁) in the left sidebar and upload `llm_judge_prompt.py` from this folder. The notebook imports it directly.

### 3. Add your Anthropic API key

In Colab, click the **Secrets** icon (🔑) in the left sidebar:
- Click **Add new secret**
- Name: `ANTHROPIC_API_KEY`
- Value: your key from [console.anthropic.com](https://console.anthropic.com)
- Toggle **Notebook access** on

### 4. Run cells in order

| Cell | What it does |
|---|---|
| Step 1 | Installs `anthropic`, loads API key |
| Step 2 | Defines the trajectory |
| Step 3 | Sends trajectory to the LLM judge |
| Step 4 | Prints verdict with ground-truth comparison |
