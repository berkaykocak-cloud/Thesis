# Introduction

This study aims to predict the converting sessions of an E-commerce company in the automotive sector. The main challenge that this study addresses is the severe class imbalance.  


* This project offers the implementation of 6 methods against class imbalance combined with 3 different algorithms.
* The methods that are applied are of two familes; Data Level Methods & Algorithm Level Methods.
* The study uses Average Precision and PR AUC as the main performance metrics for comparison. F1 Score, Precision and Recall are also analysed for the best performing algorithm.
* A two step hyperparameter tuning approach is conducted which includes Bayesian Search and Grid Search.
* Bayesian Search is applied to only Random Undersampling and Cost Sensitive Learning due to time constraints.
* Further thresholding is applied to only best performing method-algorithm combination which is cost sensitive learning combined with XGBoost.

# Model Structure

![Model Structure](Model_Structure.png)

# Methods and Algorithms

![Methods Algorithms](Methods_Algorithms.png)

# Examples

# Requirements

* imblearn
* matplotlib
* numpy
* pandas
* seaborn
* sklearn
* skopt
* xgboost

# Further Notes

The experiments are conducted on MacOS 12.3.1 and Python 3.9.7

The data file is not shared along with any preprocessing applied to the dataset due to data privacy concerns. The notebook includes all the methods that are applied and the results that are obtained from these approaches.

For further information you can refer to the [thesis](Thesis_Berkay_Kocak_614468.pdf)
