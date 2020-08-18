# Clustering Bank Transactions using text descriptions

## Introduction

This work is inspired by Ravindra Reddy Tamma's article named [Real-World Machine Learning Case Study: Clustering Transactions Based on Text Descriptions](https://www.analyticsvidhya.com/blog/2020/07/machine-learning-study-clustering-transactions-text-descriptions/). In this repository, I will replicate the method described in the article on a [Kaggle data set provided by Apoorv Patne](https://www.kaggle.com/apoorvwatsky/bank-transaction-data).

## Dataset Analysis

Features included in this dataset are shown as below:

- Account No. - This represents the account number involved in transaction.
- Date - Date of transaction
- Transaction Details - Transaction narrations in bank statements
- Cheque No. - This indicates the cheque number
- Value Date - Date of completion of transaction
- Withdrawal Amount - Indicates the amount withdrawn
- Deposit Amount - Indicates the amount deposited
- Balance Amount - Current balance of account

The feature that mainly used in this repository is **Transaction Details** as we aim to cluster based on texts.

## Methodology

1. Determining the number of topics (categories) using LDA.
