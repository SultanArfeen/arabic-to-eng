# Arabic to English — Transformer-Based Neural Machine Translation  

## Overview  
This project implements a **Transformer-based neural machine translation (NMT) model** to translate Arabic to English. The model is trained from scratch using the **ATHAR dataset** and deployed as a **Streamlit web app** on Hugging Face Spaces.  

## Project Highlights  
- **Dataset:** Arabic-English translation pairs from ATHAR (or fallback datasets from Hugging Face).  
- **Architecture:** Standard **Transformer encoder-decoder** with custom embeddings and positional encodings.  
- **Training:** Implemented using **PyTorch**, with progress tracking and model checkpointing.  
- **Deployment:** A **Streamlit UI** that provides real-time Arabic-to-English translation.  

## Installation  
Clone the repository and install dependencies:  

```bash
git clone https://github.com/yourusername/arabic-to-english-nmt.git
cd arabic-to-english-nmt
pip install -r requirements.txt
```

Usage
1. Train the Model
To train the model from scratch, run:

python train.py
2. Run the Streamlit App
To launch the interactive translation app, use:

streamlit run app.py
Dataset
The ATHAR dataset is used for training, containing parallel Arabic-English sentences. If unavailable, an alternative dataset from Hugging Face is used.

Model Architecture
Transformer-based encoder-decoder
Custom token embeddings
Positional encodings
Trained in PyTorch
Deployment
The trained model is deployed on Hugging Face Spaces, allowing users to input Arabic text and get instant English translations.

Challenges and Learnings
Preprocessing Arabic text for deep learning models.
Managing large-scale datasets efficiently.
Fine-tuning hyperparameters for improved translation accuracy.
Future Improvements
Fine-tune on a larger dataset for better accuracy.
Add support for dialectal Arabic.
Optimize inference speed for real-time translations.
