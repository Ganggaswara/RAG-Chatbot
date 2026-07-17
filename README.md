# 🏛️ Asisten Hukum Indonesia — Fine-Tuned LLaMA-3.2 + RAG Chatbot

Chatbot AI berbahasa Indonesia yang mampu menjawab pertanyaan seputar regulasi ketenagakerjaan dan perizinan usaha, dibangun di atas model **LLaMA-3.2-3B** yang di-*fine-tune* khusus, dipadukan dengan pipeline **Retrieval-Augmented Generation (RAG)** untuk menjawab berdasarkan dokumen hukum resmi — bukan halusinasi.

Proyek ini terdiri dari dua tahap utama: **fine-tuning model bahasa** dan **RAG chatbot** yang menggunakan model hasil fine-tuning tersebut sebagai otak jawabannya.

---

## ✨ Fitur Utama

- 🧠 **Model LLaMA-3.2-3B Instruct** di-fine-tune dengan LoRA (Unsloth) di atas dataset instruksi Bahasa Indonesia
- 🔬 **Dua eksperimen training** (learning rate & rank LoRA berbeda) dengan pemilihan model terbaik otomatis berdasarkan `eval_loss`
- 💾 **Auto-resume checkpoint** dari Google Drive — aman dari disconnect Colab
- 📚 **Parent-Child Retriever** — chunk besar untuk konteks LLM, chunk kecil untuk pencarian vektor presisi tinggi
- 🔀 **Hybrid retrieval**: FAISS (semantic search) + BM25 (keyword search)
- 🧭 **Semantic Router** berbasis keyword untuk menyaring pertanyaan di luar topik hukum
- 🎯 **Metadata enrichment** per dokumen (nomor peraturan, tahun, topik)
- 🖥️ **Antarmuka Gradio** siap pakai dengan contoh pertanyaan

---

## 🏗️ Arsitektur Sistem

```
┌─────────────────────────────┐         ┌──────────────────────────────┐
│   TAHAP 1: FINE-TUNING       │         │   TAHAP 2: RAG CHATBOT        │
│                              │         │                                │
│  Dataset:                    │         │  4 Dokumen Hukum (PDF):        │
│  alpaca-gpt4-indonesian      │         │  • PP No.5/2021 (Perizinan)    │
│         │                    │         │  • PP No.35/2021 (PKWT)        │
│         ▼                    │         │  • PP No.51/2023 (Pengupahan)  │
│  Base: LLaMA-3.2-3B-Instruct │         │  • UU No.6/2023 (Cipta Kerja)  │
│         │                    │         │         │                      │
│         ▼                    │         │         ▼                      │
│  LoRA Fine-Tuning (Unsloth)  │         │  Parent-Child Splitter         │
│  • Eksperimen A: lr=2e-4,r=16│         │  (1500 / 400 karakter)         │
│  • Eksperimen B: lr=1e-4,r=32│         │         │                      │
│         │                    │         │         ▼                      │
│         ▼                    │────────▶│  Embedding: BAAI/bge-m3        │
│  Pilih model terbaik         │  model  │  Index: FAISS + BM25 Hybrid    │
│  (eval_loss terendah)        │         │         │                      │
│         │                    │         │         ▼                      │
│         ▼                    │         │  Semantic Router (filter topik)│
│  Push ke HuggingFace Hub     │         │         │                      │
└─────────────────────────────┘         │         ▼                      │
                                         │  LlamaRAGLLM Wrapper           │
                                         │  (custom chat template)        │
                                         │         │                      │
                                         │         ▼                      │
                                         │  Gradio UI (Chat + Sumber)     │
                                         └──────────────────────────────┘
```

---

## 📦 Tahap 1 — Fine-Tuning Model

| Komponen | Detail |
|---|---|
| **Base model** | `unsloth/Llama-3.2-3B-Instruct-bnb-4bit` |
| **Metode** | LoRA fine-tuning via Unsloth (4-bit quantization) |
| **Dataset** | [`Ichsan2895/alpaca-gpt4-indonesian`](https://huggingface.co/datasets/Ichsan2895/alpaca-gpt4-indonesian) (split 90/10 train-val) |
| **Chat template** | Llama-3 |
| **Checkpointing** | Auto-resume dari Google Drive setiap eksperimen |
| **Eksperimen A** | `learning_rate=2e-4`, `lora_r=16`, `lora_alpha=16` |
| **Eksperimen B** | `learning_rate=1e-4`, `lora_r=32`, `lora_alpha=32` |
| **Seleksi model** | Model dengan `eval_loss` terendah dipilih otomatis |
| **Output** | Model digabung (`merged_16bit`) dan diunggah ke HuggingFace Hub |

**Model hasil fine-tuning:**
🔗 [Ganggas/legal-chatbot-llama3-indo-3b-final](https://huggingface.co/Ganggas/legal-chatbot-llama3-indo-3b-final)

---

## 📚 Tahap 2 — RAG Chatbot Hukum

### Dokumen Sumber
| Peraturan | Topik |
|---|---|
| PP No. 5 Tahun 2021 | Perizinan Berusaha |
| PP No. 35 Tahun 2021 | Ketenagakerjaan (PKWT) |
| PP No. 51 Tahun 2023 | Pengupahan |
| UU No. 6 Tahun 2023 | Cipta Kerja |

### Pipeline Retrieval
1. **Parent-Child Splitting** — dokumen dipecah jadi chunk besar (1500 karakter, untuk konteks jawaban) dan chunk kecil (400 karakter, untuk pencarian vektor yang lebih presisi)
2. **Embedding** — `BAAI/bge-m3` (multilingual, kuat untuk Bahasa Indonesia)
3. **Hybrid Search** — kombinasi FAISS (semantic) dan BM25 (keyword) agar hasil retrieval lebih relevan dan tahan terhadap variasi istilah hukum
4. **Semantic Router** — menyaring pertanyaan yang tidak relevan dengan topik hukum sebelum masuk ke pipeline RAG
5. **Custom LLM Wrapper (`LlamaRAGLLM`)** — memastikan chat template Llama-3 diterapkan dengan benar saat inferensi
6. **Prompt Engineering ketat** — model diwajibkan mengutip pasal/ayat spesifik dan tidak boleh menjawab di luar konteks dokumen

### Contoh Pertanyaan yang Bisa Dijawab
- *"Apa saja sanksi administratif bagi pelaku usaha yang melanggar perizinan berusaha?"*
- *"Berapa batas maksimal waktu lembur pekerja per hari?"*
- *"Apa syarat perjanjian kerja waktu tertentu (PKWT)?"*
- *"Bagaimana ketentuan pesangon dalam UU Cipta Kerja?"*

---

## 🛠️ Tech Stack

**Fine-Tuning:** `unsloth` · `transformers` · `trl` · `peft` · `bitsandbytes` · `accelerate` · `xformers` · `wandb`

**RAG & Retrieval:** `langchain` · `langchain-community` · `langchain-huggingface` · `faiss-cpu` · `rank-bm25` · `sentence-transformers`

**Document Processing:** `pypdf` · `pymupdf`

**Interface:** `gradio`

**Environment:** Google Colab (GPU T4)

---

## 🚀 Cara Menjalankan

### 1. Clone & Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Fine-Tuning (opsional — model sudah tersedia di HuggingFace)
Jalankan `Fine-tuning_submission_PGABL_....ipynb` di Google Colab (GPU T4), lalu set secrets `WANDB_API_KEY` dan `HF_TOKEN` di Colab Secrets.

### 3. Jalankan RAG Chatbot
Jalankan `RAG_Submission_PGABL_....ipynb` di Google Colab:
1. Mount Google Drive berisi dokumen PDF hukum
2. Jalankan seluruh cell untuk membangun index FAISS + BM25
3. Chatbot akan tersedia melalui antarmuka Gradio

---

## 📁 Struktur Proyek

```
.
├── Fine-tuning_submission_PGABL_....ipynb   # Notebook fine-tuning LLaMA-3.2-3B
├── RAG_Submission_PGABL_....ipynb           # Notebook RAG chatbot hukum
├── requirements.txt                          # Dependencies gabungan
├── link_huggingface.txt                      # Link model hasil fine-tuning
└── README.md
```

---

## 👤 Author

**I Gusti Bagus Pradnyana Gangga Swara**
Mahasiswa Ilmu Komputer, Universitas Pendidikan Ganesha
🔗 [GitHub](https://github.com/Ganggaswara) · [LinkedIn](https://www.linkedin.com/in/ganggaswara11) · [HuggingFace Model](https://huggingface.co/Ganggas/legal-chatbot-llama3-indo-3b-final)

---

## 📄 Lisensi

Proyek ini dibuat untuk keperluan submission Dicoding — *Development of LLM-based Generative AI*.
