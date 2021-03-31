# Domain-Adaptation

This is the Spring 2021 collaboration project between a few NYU Data Science students and the IBM NLP Research group. 

## Using Same-Domain Labelled Data to Improve Pretrained Language Model Performance on Question Answering

Aditya Singhal
David Shimshoni
Alex Sheng
Avi Sil
Sara Rosenthal

### Motivation
Successfully fine-tuning for a task can take thousands of gold-standard data entries, which can take years to collect or can be labor and cost expensive. This sort of quality and quantity isn’t always available for specific tasks in smaller domains. But the benefits of domain-specific finetuning have been clearly shown in papers like Lee et. al 2019 and Gururangan et. al 2020. The latter paper demonstrated that performing domain adaptive pretraining (DAPT) using an unsupervised corpus can improve a language model’s downstream performance on supervised learning tasks in the same domain. We think successful use of sparse, domain-specific, labelled data from different supervised tasks (like NER) for Domain Adaptation could be helpful to improve performance in domains where domain data is unavailable for the required task. This will allow the field to better pool their resources and also help in the use of existing tools in a timely manner, such as the QA system developed to answer medical questions during the COVID-19 crisis [9]. 

### Data Collection
We explore the performance of this compositional adaptation approach in the Movies, News and Biomedical science domains. Specifically, our target domain-specific QA tasks are MoviesQA [12], NewsQA [11] and BioASQ (citation). For fine-tuning a large pre-trained LM on question answering, we will be using the general question answering dataset SQuAD 2.0 [8]. The data for DAPT on RoBERTa-Base [5] will use data from– A) Movies: IMDB [6], Cornell Movie Dialogues [1], Cornell Movie Review Polarity [7], B) News: RealNews Corpus [13] and C) Biomedical: [missing]. The labelled, domain specific data will all be from these different NER datasets: MIT Movie Corpus [4], CoNLL 2003 News NER [10. All these datasets are publicly available. 

### Modeling and Analysis
The RoBERTa-Base model is initialized to pretrained weights, fine-tuned in either sequential or multitask regime on SQuAD 2.0 along with a domain-specific task, and evaluated on the target domain-specific QA task in a zero-shot setting without any training on the target task. One of the baselines will be to take RoBERTa and fine-tune it on SQuAD 2.0. Then we implement DAPT on the previous model using the mentioned domain-specific corpora for a second baseline. These will be trained using HuggingFace’s language modeling and QA scripts. The experimental regimen is to do multitask learning using the Jiant library, with the RoBERTa model on the supervised Named-Entity Recognition task and SQuAD 2.0. All of these model variations will be evaluated on the mentioned domain-specific QA evaluation sets, and the Macro-F1 score will be reported. 

### Collaboration Statement
Alex, David and Adi will handle training and evaluation for the Bio, News and Movies domains respectively. David took charge in research, collaboration and formatting, Adi was instrumental with planning and technical management and Alex helped with literature review and idea formation. This team is mentored by IBM’s Avi Sil and Sara Rosenthal. Xiang Pan is doing similar research but we will produce two different projects for the class. We intend to aggregate our results later on for a publication.

#### References

References
[1] Danescu-Niculescu-Mizil, C., & Lee, L. (2011). Chameleons in imagined conversations: A new approach to understanding coordination of linguistic style in dialogs. https://www.cs.cornell.edu/~cristian/papers/chameleons.pdf

[2] Gururangan, S., Marasović, A., Swayamdipta, S., Lo, K., Beltagy, I., Downey, D., Smith, N., & Allen. (2020). Don’t Stop Pretraining: Adapt Language Models to Domains and Tasks (pp. 8342–8360). https://www.aclweb.org/anthology/2020.acl-main.740.pdf

[3] Lee, J., Yoon, W., Kim, S., Kim, D., Kim, S., So, C. H., & Kang, J. (2019). BioBERT: a pre-trained biomedical language representation model for biomedical text mining. Bioinformatics. https://doi.org/10.1093/bioinformatics/btz682

[4] Liu, J., Pasupat, P., Wang, Y., Cyphers, S., & Glass, J. (2013). Query Understanding Enhanced By Hierarchical Parsing Structures. https://groups.csail.mit.edu/sls/publications/2013/Liu_ASRU_2013.pdf

[5] Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., Levy, O., Lewis, M., Zettlemoyer, L., & Stoyanov, V. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach. ArXiv.org. https://arxiv.org/abs/1907.11692

[6] Maas, A., Daly, R., Pham, P., Huang, D., Ng, A., & Potts, C. (2011). Learning Word Vectors for Sentiment Analysis. https://ai.stanford.edu/~amaas/papers/wvSent_acl2011.pdf

[7] Pang, B., Lee, L., & Vaithyanathan, S. (2019). Thumbs up? Sentiment Classification using Machine Learning Techniques. ArXiv.org. https://arxiv.org/abs/cs/0205070

[8] Rajpurkar, P., Jia, R., & Liang, P. (2018). Know What You Don’t Know: Unanswerable Questions for SQuAD (pp. 784–789). https://www.aclweb.org/anthology/P18-2124.pdf

[9] Reddy, R., Iyer, B., Arafat, M., Zhang, R., Sil, A., Castelli, V., Florian, R., & Roukos, S. (2020). End-to-End QA on COVID-19: Domain Adaptation with Synthetic Training. https://arxiv.org/pdf/2012.01414.pdf

[10] Tjong Kim Sang, E. F., & De Meulder, F. (2003). Introduction to the CoNLL-2003 shared task. Proceedings of the Seventh Conference on Natural Language Learning at HLT-NAACL 2003 -. https://doi.org/10.3115/1119176.1119195

[11] Trischler, A., Wang, T., Yuan, X., Harris, J., Sordoni, A., Bachman, P., & Suleman, K. (2017). NewsQA: A Machine Comprehension Dataset (pp. 191–200). 
Association for Computational Linguistics. https://www.aclweb.org/anthology/W17-2623.pdf

[12] Xu, Y., Zhong, X., Yepes, A. J. J., & Lau, J. H. (2020). Forget Me Not: Reducing Catastrophic Forgetting for Domain Adaptation in Reading Comprehension. ArXiv:1911.00202 [Cs]. https://arxiv.org/abs/1911.00202

[13] Zellers, R., Holtzman, A., Rashkin, H., Bisk, Y., Farhadi, A., Roesner, F., Choi, Y., & Allen, P. (2019). Defending Against Neural Fake News. https://arxiv.org/pdf/1905.12616.pdf


