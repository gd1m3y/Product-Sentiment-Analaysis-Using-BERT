# Sentiment Analaysis Using BERT
## introduction
The Aim of this project is to classify the sentiment of a given product review using BERT.

Sentiment Analysis or Text Classification is the process of determining the sentiment behind the text based on the context.

![image](https://editor.analyticsvidhya.com/uploads/61727sentiment-fig-1-689.jpeg)

This Project Aims to Demonstrate the Text Classification.
## Work Flow
The work Flow of the Project - 

![img](https://github.com/gd1m3y/Text-Summarizer-Using-Elmo/blob/master/extractive.png)

* Preprocessing - Using regular Expressions and many other libraries to remove irregularities in the data such as punctuations,links,numbers which doesnt have any specific effect on the model rather may result in abnormal results.
* Sentence-Tokenization - Tokenization is the process of converting text into tokens so that it can be understood by the model.
* Distill-BERT - we use our model to classify the text into either positive or negative sentences.

The notebook Demonstrates 
## BERT
BERT stands for Bidirectional Encoder Representations from Transformers. It is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context. As a result, the pre-trained BERT model can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of NLP tasks


![image](https://cdn.analyticsvidhya.com/wp-content/uploads/2019/09/bert_encoder.png)

BERT is based on the Transformer architecture.
![image](https://cdn.analyticsvidhya.com/wp-content/uploads/2019/09/bert_emnedding.png)

                        Tokenization in bert

## Data Set
The data set used is boat amazon reviews dataset which is uploaded in the repo.
## Model Used 
The DistilBERT model was proposed in the blog post Smaller, faster, cheaper, lighter: Introducing DistilBERT, a distilled version of BERT, and the paper DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. DistilBERT is a small, fast, cheap and light Transformer model trained by distilling BERT base. It has 40% less parameters than bert-base-uncased, runs 60% faster while preserving over 95% of BERT’s performances as measured on the GLUE language understanding benchmark.
## Results 
### positive word cloud
![image](https://cdn.analyticsvidhya.com/wp-content/uploads/2019/09/bert_emnedding.png)
### positive topics
![image](https://cdn.analyticsvidhya.com/wp-content/uploads/2019/09/bert_emnedding.png)
### negative word cloud 
![image](https://cdn.analyticsvidhya.com/wp-content/uploads/2019/09/bert_emnedding.png)
### negative topics
![image](https://cdn.analyticsvidhya.com/wp-content/uploads/2019/09/bert_emnedding.png)
## Technology Stack

* Spacy - A NLP Library used for variety of tasks Such as Named entity recognition
* Transformer - A Deeplearning Library developed by hugging face .
* Numpy - Basic Mathematical library
* re - for performing string operations
* pandas - Data manipulation library
* matplotlib - visualization libray
* Sklearn - A library consisting of many functions regarding Mathematics and Statistics.
## To-do
* Using different sophisticated models or methodologies to train on embeddings and achieve a better accuracy
* Different Preprocessing steps for a better result
## Contact
Want to contribute ? Contact me at narayanamay123@gmail.com
