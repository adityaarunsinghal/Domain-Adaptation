# Domain-Adaptation (Final Paper Link)

This is the Spring 2021 collaboration project between a few NYU Data Science students and the IBM NLP Research group. 

## Using Same-Domain Labelled Data to Improve Pretrained Language Model Performance on Question Answering

Aditya Singhal, 
David Shimshoni, 
Alex Sheng, 
Xiang Pan,
Avi Sil, 
Sara Rosenthal

### Motivation
Successfully fine-tuning for a task can be costly and labor intensive. This sort of quality and quantity isn’t always available for specific tasks in smaller domains. But the benefits of domain-specific finetuning has been clearly shown in papers like Lee et. al 2019 and Gururangan et. al 2020. The latter demonstrated that performing domain adaptive pretraining (DAPT) using an unsupervised corpus can improve a language model’s downstream performance on supervised learning tasks in the same domain. We think the additional use of sparse, domain-specific, labelled data from unrelated supervised tasks (like NER) for Domain Adaptation could be helpful to improve performance in domains where domain data is in short supply for the required task (QA). We will try continued training and also multi-task learning to see if the labelled data helps. A successful implementation will allow the field to better pool their resources and also use the existing tools more effectively in times of need; such as the QA system developed to answer medical questions during the COVID-19 crisis [11]. 

### Data Collection
We explore the performance of this compositional adaptation approach in the Movies, News and Biomedical science domains. Specifically, our target domain-specific QA tasks are MoviesQA [14], NewsQA [13] and BioASQ [12]. For fine-tuning a large pre-trained LM on question answering, we will be using the general question answering dataset SQuAD 2.0 [10]. The data for DAPT on RoBERTa-Base [7] will use data from– A) Movies: IMDB [8], Cornell Movie Dialogues [1], Cornell Movie Review Polarity [9], B) News: RealNews Corpus [15] and C) Biomedical: CTD-Pfizer Corpus [2]. The labelled, domain specific data will all be from these different NER datasets: MIT Movie Corpus [6], CoNLL 2003 News NER [12] and BC4CHEMD NER [4]. All these datasets are publicly available. 

### Modeling and Analysis
The RoBERTa-Base model is initialized to pretrained weights, fine-tuned in either sequential or multitask regime on SQuAD 2.0 along with a domain-specific task, and evaluated on the target domain-specific QA task in a zero-shot setting without any training on the target task. One of the baselines will be to take generic RoBERTa and fine-tune it on SQuAD 2.0 using HuggingFace. Then we do the same with a domain-specific RoBERTa obtained through continued LM to get a second baseline. The experimental model will be obtained though both sequential and multitask learning (MTL) using the Jiant library, with the baselines + NER + SQuAD 2.0. Our team hasn’t done MTL before but it is the only part of our project that requires engineering and Jiant will help. All of these model variations will be evaluated on the mentioned domain-specific QA evaluation sets, and the Macro-F1 score will be reported. 

### Collaboration Statement
Alex, David Adi, and Xiang Pan will handle training and evaluation for the Bio, News, Movies, Covid domains respectively. David took charge in research, collaboration and formatting, Adi was instrumental with planning and technical management and Alex helped with literature review and idea formation, Xiang Pan take charge of the final paper writing and idea formation. This team is mentored by IBM’s Avi Sil and Sara Rosenthal. We intend to aggregate our results later on for a publication.

#### References

[1] Danescu-Niculescu-Mizil, C., & Lee, L. (2011). Chameleons in imagined conversations: A new approach to understanding coordination of linguistic style in dialogs. https://www.cs.cornell.edu/~cristian/papers/chameleons.pdf

[2] Davis, A. P., Wiegers, T. C., Roberts, P. M., King, B. L., Lay, J. M., Lennon-Hopkins, K., ... & Mattingly, C. J. (2013). A CTD–Pfizer collaboration: manual curation of 88 000 scientific articles text mined for drug–disease and drug–phenotype interactions. Database, 2013.

[3] Gururangan, S., Marasović, A., Swayamdipta, S., Lo, K., Beltagy, I., Downey, D., Smith, N., & Allen. (2020). Don’t Stop Pretraining: Adapt Language Models to Domains and Tasks (pp. 8342–8360). https://www.aclweb.org/anthology/2020.acl-main.740.pdf

[4] Krallinger, M., Leitner, F., Rabal, O., Vazquez, M., Oyarzabal, J., & Valencia, A. (2015). CHEMDNER: The drugs and chemical names extraction challenge. Journal of cheminformatics, 7(1), 1-11.

[5] Lee, J., Yoon, W., Kim, S., Kim, D., Kim, S., So, C. H., & Kang, J. (2019). BioBERT: a pre-trained biomedical language representation model for biomedical text mining. Bioinformatics. https://doi.org/10.1093/bioinformatics/btz682

[6] Liu, J., Pasupat, P., Wang, Y., Cyphers, S., & Glass, J. (2013). Query Understanding Enhanced By Hierarchical Parsing Structures. https://groups.csail.mit.edu/sls/publications/2013/Liu_ASRU_2013.pdf

[7] Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., Levy, O., Lewis, M., Zettlemoyer, L., & Stoyanov, V. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach. ArXiv.org. https://arxiv.org/abs/1907.11692

[8] Maas, A., Daly, R., Pham, P., Huang, D., Ng, A., & Potts, C. (2011). Learning Word Vectors for Sentiment Analysis. https://ai.stanford.edu/~amaas/papers/wvSent_acl2011.pdf

[9] Pang, B., Lee, L., & Vaithyanathan, S. (2019). Thumbs up? Sentiment Classification using Machine Learning Techniques. ArXiv.org. https://arxiv.org/abs/cs/0205070

[10] Rajpurkar, P., Jia, R., & Liang, P. (2018). Know What You Don’t Know: Unanswerable Questions for SQuAD (pp. 784–789). https://www.aclweb.org/anthology/P18-2124.pdf

[11] Reddy, R., Iyer, B., Arafat, M., Zhang, R., Sil, A., Castelli, V., Florian, R., & Roukos, S. (2020). End-to-End QA on COVID-19: Domain Adaptation with Synthetic Training. https://arxiv.org/pdf/2012.01414.pdf

[12] The BioASQ Challenge. http://www.bioasq.org/

[13] Tjong Kim Sang, E. F., & De Meulder, F. (2003). Introduction to the CoNLL-2003 shared task. Proceedings of the Seventh Conference on Natural Language Learning at HLT-NAACL 2003 -. https://doi.org/10.3115/1119176.1119195

[14] Trischler, A., Wang, T., Yuan, X., Harris, J., Sordoni, A., Bachman, P., & Suleman, K. (2017). NewsQA: A Machine Comprehension Dataset (pp. 191–200). Association for Computational Linguistics. https://www.aclweb.org/anthology/W17-2623.pdf

[15] Xu, Y., Zhong, X., Yepes, A. J. J., & Lau, J. H. (2020). Forget Me Not: Reducing Catastrophic Forgetting for Domain Adaptation in Reading Comprehension. ArXiv:1911.00202 [Cs]. https://arxiv.org/abs/1911.00202

[16] Zellers, R., Holtzman, A., Rashkin, H., Bisk, Y., Farhadi, A., Roesner, F., Choi, Y., & Allen, P. (2019). Defending Against Neural Fake News. https://arxiv.org/pdf/1905.12616.pdf
