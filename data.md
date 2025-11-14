# Thread Summarization and Topic Segmentation (Code-Mixed + English):

We study **thread summarization and topic segmentation** across multiple conversation styles and languages. To enable consistent training/evaluation and clean ablations, we normalize all sources to a **single JSONL “one-thread-per-line” schema** and keep a lightweight `domain` tag (dialogue/news/email).

---

## 1. Unified File Format:

All datasets are converted to **JSON Lines** (`.jsonl`), **one thread per line**:

```json
{
  "thread_id": "source:unique_id",
  "source": "cs-sum | crocosum | dialogsum | kaggle_email",
  "domain": "dialogue | news | email",
  "title": "optional",
  "messages": [
    {
      "mid": 0,
      "parent": null,
      "author": "Speaker/Role",
      "time": "ISO8601 or null",
      "lang": null,
      "text": "utterance (or article body)"
    }
  ],
  "summary": "gold reference summary for the thread",
  "refs": ["gold summary (room for multiple refs later)"],
  "meta": {}
}
```

**Why this format?**

* Works for **multi-turn dialogues** (many short messages), **emails** (threaded messages with timestamps), and **news articles** (single long message).
* Separates inputs (`messages`) from labels (`summary`) cleanly.
* `domain` enables **domain-aware prompts**, balanced sampling, and **per-domain reporting** without changing code.
* Future-proof: easily add scraped Reddit/YouTube or meetings later (Maybe for fine-tuning)

---

## 2. Datasets:

| Dataset                              | Link                                                                                                                                                                   | Why this dataset?                                                                                                                                     |
| ------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Kaggle Email Thread Summary**      | [https://www.kaggle.com/datasets/marawanxmamdouh/email-thread-summary-dataset/data](https://www.kaggle.com/datasets/marawanxmamdouh/email-thread-summary-dataset/data) | Real **multi-message email threads** with human **thread summaries**. Covers asynchronous business conversation structure; complements dialogue/news. |
| **CS-Sum (Code-Switched Dialogues)** | [https://huggingface.co/datasets/SkAndMl/cs-sum](https://huggingface.co/datasets/SkAndMl/cs-sum)                                                                       | **Code-mixed** multi-turn dialogues (e.g., zh-en). Directly tests robustness to code-switching in conversational summarization.                       |
| **DialogSum (English Dialogues)**    | [https://github.com/cylnlp/dialogsum/tree/main](https://github.com/cylnlp/dialogsum/tree/main)                                                                         | Large, clean **English dialogues** with concise abstractive summaries; strong baseline for dialogue summarization.                                    |
| **CroCoSum (Cross-lingual News)**    | [https://huggingface.co/datasets/ruochenz/CroCoSum](https://huggingface.co/datasets/ruochenz/CroCoSum)                                                                 | **Articles** (title+body) with cross-lingual/code-mixed summaries. Broad topics; non-dialogue structure stresses generalization.                      |

> Intentional mix: code-mixed + monolingual, and dialogue + email + news. This tests whether a single summarizer can generalize across **domains** and **language conditions**.

---

## 3. What each dataset contains:

### Kaggle Email Thread Summary (email, EN):

* **Content:** Real email threads with message metadata (`from`, `subject`, `timestamp`, `body`) and gold **thread summaries**.
* **Raw:** Two JSON files: message-level details and thread-level summaries.
* **Why:** Models thread-level discourse and longer horizons; unlike short chats.

### CS-Sum (dialogue, code-mixed):

* **Content:** Code-switched dialogues (`dialogue` / `cs_dialogue`) with gold summaries.
* **Raw:** Text blocks with `#Person#:` turn markers; one combined split.
* **Why:** Directly targets code-mix summarization in multi-turn settings.

### DialogSum (dialogue, EN):

* **Content:** Multi-turn English dialogues with succinct abstractive summaries.
* **Raw:** JSONL **train/dev/test** provided.
* **Why:** Clean dialogue baseline, complements CS-Sum.

### CroCoSum (news, cross-lingual/code-mixed):

* **Content:** Articles referenced via `src_docs.json` + post summaries (often non-EN or mixed).
* **Raw:** JSONL `train/val/test` + `src_docs.json` mapping `links → {title, body}`.
* **Why:** Long-form, non-dialogue inputs with cross-lingual targets.

---

## 4. Examples:

<details><summary><b>Kaggle Email</b></summary>

```json
{
  "thread_id": "kaggle:3188",
  "source": "kaggle_email",
  "domain": "email",
  "title": "CNG/Peoples Natural Gas",
  "messages": [
    {
      "mid": 0,
      "parent": null,
      "author": "Chris Germany",
      "time": "2000-09-21T08:32:00Z",
      "lang": null,
      "text": "Wade, for July and August, give me the actual volumes and price..."
    }
  ],
  "summary": "The thread discusses volumes/pricing for Peoples Natural Gas and next steps.",
  "refs": ["The thread discusses volumes/pricing for Peoples Natural Gas and next steps."],
  "meta": {"num_messages": 2}
}
```

</details>

<details><summary><b>CS-Sum (code-mixed dialogue)</b></summary>

```json
{
  "thread_id": "cs-sum:cs-sum_0",
  "source": "cs-sum",
  "domain": "dialogue",
  "title": null,
  "messages": [
    {"mid": 0, "parent": null, "author": "Person1", "time": null, "lang": null,
     "text": "你是不是需要 help with something?"},
    {"mid": 1, "parent": null, "author": "Person2", "time": null, "lang": null,
     "text": "我不知道要去哪里 to get my ballot."}
  ],
  "summary": "Person1 helps Person2 obtain a ballot and explains next steps.",
  "refs": ["Person1 helps Person2 obtain a ballot and explains next steps."],
  "meta": {}
}
```

</details>

<details><summary><b>DialogSum (dialogue, EN)</b></summary>

```json
{
  "thread_id": "dialogsum:0",
  "source": "dialogsum",
  "domain": "dialogue",
  "title": "get a check-up",
  "messages": [
    {"mid": 0, "parent": null, "author": "#Person1#", "time": null, "lang": null,
     "text": "Hi, Mr. Smith. I'm Doctor Hawkins. Why are you here today?"},
    {"mid": 1, "parent": null, "author": "#Person2#", "time": null, "lang": null,
     "text": "…"}
  ],
  "summary": "A patient visits Dr. Hawkins for a check-up and explains symptoms.",
  "refs": ["A patient visits Dr. Hawkins for a check-up and explains symptoms."],
  "meta": {}
}
```

</details>

<details><summary><b>CroCoSum (news)</b></summary>

```json
{
  "thread_id": "crocosum:68021",
  "source": "crocosum",
  "domain": "news",
  "title": "黑客利用 Slack 和社会工程技术窃取 EA 游戏源代码",
  "messages": [
    {"mid": 0, "parent": null, "author": "ARTICLE", "time": null, "lang": "en",
     "text": "How Hackers Used Slack to Break into EA Games ..."}
  ],
  "summary": "（中文/夹杂英文的摘要…）",
  "refs": ["（中文/夹杂英文的摘要…）"],
  "meta": {}
}
```

</details>

---

## 5. Where we collected each dataset & how we prepped it:

* **Kaggle Email Thread Summary**
  **Link:** [https://www.kaggle.com/datasets/marawanxmamdouh/email-thread-summary-dataset/data](https://www.kaggle.com/datasets/marawanxmamdouh/email-thread-summary-dataset/data)
  **Prep:** Loaded `email_thread_details.json` (bodies + metadata) and `email_thread_summaries.json`; grouped messages by `thread_id`; **sorted by `timestamp`**; stripped quoted **“Original Message”** blocks and trailing signatures; built unified JSONL per thread with `domain: "email"`; created an **80/10/10** train/dev/test split **by thread** (seed=42).

* **CS-Sum**
  **Link:** [https://huggingface.co/datasets/SkAndMl/cs-sum](https://huggingface.co/datasets/SkAndMl/cs-sum)
  **Prep:** Parsed **`cs_dialogue` (preferred) or `dialogue` (fallback)** into `(speaker, text)` turns via a `#Speaker#:` regex; emitted unified JSONL; since only one split is provided, we created an **80/10/10** split (seed=42).

* **DialogSum**
  **Link:** [https://github.com/cylnlp/dialogsum/tree/main](https://github.com/cylnlp/dialogsum/tree/main)
  **Prep:** Used the official JSONL **train/dev/test** files; performed direct field mapping to our unified JSONL schema; **no resplitting**.

* **CroCoSum**
  **Link:** [https://huggingface.co/datasets/ruochenz/CroCoSum](https://huggingface.co/datasets/ruochenz/CroCoSum)
  **Prep:** Pulled official **train/val/test** via `hf_hub_download`; resolved article text from `src_docs.json` using each sample’s `links` (**title + body** → one `"ARTICLE"` message); summary set to `post_text` (**often Chinese / code-mixed**); wrote unified JSONL per split.

---

## 6. Splits & directory layout:

We **preserve official splits** where provided; if absent, we create **80/10/10** on thread IDs (no leakage).

Final layout after normalization:

```
data/
└─ splits/
   ├─ kaggle_train.jsonl
   ├─ kaggle_dev.jsonl
   ├─ kaggle_test.jsonl
   ├─ cs_sum_train.jsonl
   ├─ cs_sum_dev.jsonl
   ├─ cs_sum_test.jsonl
   ├─ dialogsum.train.jsonl
   ├─ dialogsum.dev.jsonl
   ├─ dialogsum.test.jsonl
   ├─ croco_train.jsonl
   ├─ croco_dev.jsonl
   └─ croco_test.jsonl
```

---

## 7. How to run:

1. Open the project notebook. (https://colab.research.google.com/drive/1_D4TZ9mFJXHmiEdFp7XzdNSRJRb2Oflq?usp=sharing)
2. **Upload** your "initial_path" data.
3. **Run all cells.** The notebook:

   * Loads the four datasets from the links above,
   * Converts each to the unified JSONL schema,
   * Uses official splits or creates **80/10/10**,
   * Writes all files under `data/splits/`,
   * Prints **summary statistics**.

**Using a Google Drive bundle**

We provide a Drive folder with:

* `initial_data/` — raw JSON/JSONL used by the notebook,
* `data_splits.tar.gz` — gzipped archive of `data/splits/`.

---

## 8. Statistics:

```
Kaggle Email Threads:
  Train  3,326 items | msgs: 17,321 (avg 5.2/item)
  Dev      415 items | msgs:  2,129 (avg 5.1/item)
  Test     417 items | msgs:  2,190 (avg 5.3/item)
  Total  4,158 items | msgs: 21,640

CS-Sum (Code-Switched):
  Train  2,584 items | msgs: 26,219 (avg 10.1/item)
  Dev      323 items | msgs:  3,098 (avg  9.6/item)
  Test     325 items | msgs:  3,110 (avg  9.6/item)
  Total  3,232 items | msgs: 32,427

CroCoSum (News):
  Train 12,989 items | msgs: 12,989 (avg 1.0/item)
  Dev    2,784 items | msgs:  2,784 (avg 1.0/item)
  Test   2,784 items | msgs:  2,784 (avg 1.0/item)
  Total 18,557 items | msgs: 18,557

DialogSum (Dialogue):
  Train 12,460 items | msgs: 118,292 (avg 9.5/item)
  Dev      500 items | msgs:   4,690 (avg 9.4/item)
  Test     500 items | msgs:   4,853 (avg 9.7/item)
  Total 13,460 items | msgs: 127,835

------------------------------------------------------------
Combined Total: 39,407 items | msgs: 200,459
------------------------------------------------------------
```

These numbers confirm the expected **80/10/10** splits (where applicable) and that thread/message structure was preserved after normalization.

---

## 9. Links:

* **Kaggle Email Thread Summary:** [https://www.kaggle.com/datasets/marawanxmamdouh/email-thread-summary-dataset/data](https://www.kaggle.com/datasets/marawanxmamdouh/email-thread-summary-dataset/data)
* **CS-Sum:** [https://huggingface.co/datasets/SkAndMl/cs-sum](https://huggingface.co/datasets/SkAndMl/cs-sum)
* **DialogSum:** [https://github.com/cylnlp/dialogsum/tree/main](https://github.com/cylnlp/dialogsum/tree/main)
* **CroCoSum:** [https://huggingface.co/datasets/ruochenz/CroCoSum](https://huggingface.co/datasets/ruochenz/CroCoSum)

---

**Colab notebook:** https://colab.research.google.com/drive/1_D4TZ9mFJXHmiEdFp7XzdNSRJRb2Oflq?usp=sharing

**Google Drive bundle:** https://drive.google.com/drive/folders/17VVqdi-ZQVWNYhtiaVRE0AtlhqbX3-jc?usp=sharing