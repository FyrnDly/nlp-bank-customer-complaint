🏦 Bank Customer Complaint 🏦
===
## 💣 Problem Statement
> Banks often receive a multitude of customer complaints. Due to the sheer volume, bank customer service teams frequently struggle to categorize these complaints accurately. Consequently, **the complaint resolution process slows down, leading to customer dissatisfaction**.

## 🤖 NLP Implementation
> o address this issue, we can employ Natural Language Processing (NLP) technology. With **NLP, we can create a system that automatically recognizes the content of customer complaints and determines the appropriate product category**. This will make the complaint handling process faster and more efficient.

## 🎯 Target
> The goal of using this NLP system is to **speed up the response time** of the customer service team and **improve the accuracy** in classifying complaints. The target is to create an NLP system with an **accuracy rate of at least 80%**, measured by metrics such as Accuracy, Precision, Recall, and F1-Score.

With this system, banks are expected to respond to customer complaints more quickly and accurately, thereby increasing customer satisfaction and the efficiency of the customer service team. I hope this explanation helps you convey your project more clearly to the readers.

## Papar Information
- Title:  `Model NLP Classification Bank Customer Complaint`
- Authors:  [FyrnDly](https://github.com/FyrnDly)
- Model: **LSTM**
- Deploy: **Streamlit**

## Dataset Preparation
| Dataset | Download |
| ---     | ---   |
| complaints.csv | [download](https://www.kaggle.com/datasets/adhamelkomy/bank-customer-complaint-analysis) |

## Use
- Environment App
  ```
  pip install -r requirements.txt
  ```
- Running Streamlit
  ```
  streamlit run app.py
  ```
- Training Model
  > Open file **/models/model_train.ipnb** on *python notebook* to training model

## Pretrained model
| Description | Model | Vectorizer |
| --- | --- | --- |
| Model Undersampling Training | [model](/models/model_lstm_rus.h5) | [vectorizer](./models/vectorizer.pkl) |
| Model Oversampling Training | [model](/models/model_lstm_ros.h5) | [vectorizer](./models/vectorizer.pkl) |