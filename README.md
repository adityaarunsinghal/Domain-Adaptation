# Task Transfer and Domain Adaptation for Zero-Shot Question Answering [(Paper Link)](https://github.com/adityaarunsinghal/Domain-Adaptation/blob/b5199448881c3a1aec97e7599acd8cd8cda936f0/Singhal,%20Shimshoni,%20Pan,%20Sheng%20-%20DomainQA.pdf)

This is the code for the "Task Transfer and Domain Adaptation for Zero-Shot Question Answering" paper at the NAACL 2022 workshop on Deep Learning for Low-Resource NLP (DeepLo 2022). This project is a collaboration between IBM Research and NYU.

## Using Same-Domain Labelled Data to Improve Pretrained Language Model Performance on Question Answering

Aditya Singhal, 
David Shimshoni, 
Alex Sheng, 
Xiang Pan,
Avi Sil, 
Sara Rosenthal

## Abstract

Pretrained language models have shown success in various areas of natural language processing, including reading comprehension tasks. However, when applying machine learning methods to new domains, labeled data may not always be available. To address this, we use supervised pretraining on source-domain data to reduce sample complexity on domain-specific downstream tasks. We evaluate zero-shot performance on domain-specific reading comprehension tasks by combining task transfer with domain adaptation to fine-tune a pretrained model with no labelled data from the target task. Our approach outperforms Domain-Adaptive Pretraining on downstream domain-specific reading comprehension tasks in 3 out of 4 domains.
