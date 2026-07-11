# Frequently Asked Questions

## 🔍 Common Issues and Solutions

---

### 💬 Q1: Why am I failing to parse LLM output format / How to select the right LLM?

<div class="faq-answer">
<p><strong>Solution:</strong> Small language models often struggle to follow prompts and generate responses in the expected format. For better results, we recommend using large reasoning models such as:</p>

<ul>
  <li>DeepSeek-R1 671B</li>
  <li>OpenAI o-series models</li>
  <li>Claude 3.7 Sonnet</li>
</ul>

<p>These models provide superior reasoning capabilities and are more likely to produce correctly formatted outputs.</p>
</div>

---

### 🌐 Q2: "We couldn't connect to 'https://huggingface.co'" error

<div class="faq-answer">
<p><strong>Error Message:</strong></p>
<div class="error-message">
OSError: We couldn't connect to 'https://huggingface.co' to load this file, couldn't find it in the cached files and it looks like GPTCache/paraphrase-albert-small-v2 is not the path to a directory containing a file named config.json.
Checkout your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offline-mode'.
</div>

<p><strong>Solution:</strong> This issue is typically caused by network access problems to Hugging Face. Try these solutions:</p>

<details>
<summary><strong>Network Issue? Try Using a Mirror</strong></summary>

```bash
export HF_ENDPOINT=https://hf-mirror.com
```
</details>

<details>
<summary><strong>Permission Issue? Set Up a Personal Token</strong></summary>

```bash
export HUGGING_FACE_HUB_TOKEN=xxxx
```
</details>
</div>

---

### 📓 Q3: DeepSearcher doesn't run in Jupyter notebook

<div class="faq-answer">
<p><strong>Solution:</strong> This is a common issue with asyncio in Jupyter notebooks. Install <code>nest_asyncio</code> and add the following code to the top of your notebook:</p>

<div class="code-steps">
<p><strong>Step 1:</strong> Install the required package</p>

```bash
pip install nest_asyncio
```

<p><strong>Step 2:</strong> Add these lines to the beginning of your notebook</p>

```python
import nest_asyncio
nest_asyncio.apply()
```
</div>
</div>
</div> 