# CS412 Project: Instagram Influencers Category Classification and Like Count Prediction

## Overview
In this project, the aim was to analyze instagram posts for classifying influencers category. In a seperate task, influencers posts and account information were used to predict the number of likes a post will receive by an influencer. This was achieved by developing seperate ML models:
1. **SVM Classifier** which classified users into categories based on posts processing. For predicting like counts
2. **LightGBM Regressor** was used based on text (posts) processing as well while integrating comment counts into the picture.

For text processing, techiniques such as TF-IDF vectorization and Chi-Squared feature selection were utilized in the data pre-processing stage, which showed boost in the model accuracy. Those point will be discussed in details in the next paragraphs.

## Project Structure
- **data-preperation.py**: Contains the extraction of key data information (username, posts, etc...) from a dataset.
- **data-preprocessing**: Processsing of the large text infromation, with features extracted using TF-idf vectorizer, and chi-squre method used for feature selection.
- **svm-model.py**: Contains a linear SVM model that is tuned and trained on influencer category classifcation task.
- **LightGBM-model.py**: A gradient-boosting machine model used for predicting like counts of a user.
- **.gitignore**: File used to exclude certain files from being tracked by Git.
- **README.md**: This file.

# Results
## 1. Classification Task
## Accuracy
For the classification task, multiple models were experminted on the data, namely Naive Bayes, Random Forest Tree, and SVM. Naive Bayes showed the worst accuracy so far, which is expected due to its assumption that features are independant on each other, which contradicts the core of text analysis. Random forest tree provided acceptable accuracy when tuned moderately. Further tuning may have resulted in better performance, however, tuning the parameters was computationally expensive and time consuming. The best performance in my case was achieved by the SVM model, specifically linear SVM model. Using SVM made sense due to the large size of features (fat matrix), and SVM efficiency on TF-idf based text analysis. Also, the Linear SVM was expected to have a better performance considering the use of log weights in data processing, hence, linear relations between features is assumed. The difference in various models accuracies is shown below.

![image](https://github.com/user-attachments/assets/f4857e8b-733c-4250-b3db-09410a074f85)

## Confusion Matrix
On the other hand, confusion matrix give insight into where the misclassifications happened, which allowed for devolping the model's performance. For instance, it was observed that "Food" class has the best performance with minimal misclassification (96 correct). "Tech" and "Travel" also show good performance with high accuracy.
However, "Gaming" is poorly classified, reflecting issues with rare class representation. This was observed from the low number of training instances having "gaming category". Moreover, "Mom and children" and "Sports" exhibit low accuracy, which is again due to class imbalnce and similarity in key tokens. It may also be due to inadequate feature separation.There was also substantial confusion among other classes.
**Overlaps** observed between:
"Art" and "Fashion"
"Entertainment" and "Sports"

![image](https://github.com/user-attachments/assets/7488f604-a6c2-40ba-9f31-bf888b04d83f)


## 2. Regression Task
This document provides a detailed explanation of the analysis conducted on a regression task aimed at predicting average like counts for a test dataset. The results have been evaluated using various visualizations, including residual distributions, scatter plots of predictions, and density comparisons between predicted and actual values. Below, each aspect of the analysis is described in detail, followed by insights and observations derived from the results.

Residuals Distribution
The first graph shows the distribution of residuals, which represent the difference between actual and predicted values. A large portion of residuals is concentrated near zero, indicating that the model performs well for most test cases. However, the presence of a long tail extending to higher residual values highlights the existence of outliers where the model predictions deviate significantly from the actual data.

This behavior suggests that while the model is generally accurate, it struggles to generalize for certain data points with rare or extreme patterns. These outliers may correspond to specific cases in the dataset where features do not align well with the overall trends captured by the model. Such discrepancies could stem from noisy data, insufficient feature representation, or inherent variability in the underlying dataset.

Predicted Like Counts for the Test Set
The second graph provides a scatter plot illustrating the predicted average like counts for individual users in the test set. The majority of predictions lie in a relatively low range, with occasional spikes indicating significantly higher values. While this pattern aligns with the expected distribution of like counts in social media datasets (where most users have moderate engagement levels and a few have disproportionately high ones), the model appears to overestimate in some cases.

This uneven distribution of predictions highlights a potential issue with overfitting to certain user profiles or data segments. The presence of extreme values suggests that the model may be overly sensitive to features associated with rare cases. While it captures the general trend of the data, its performance for users with exceptionally high or low like counts remains suboptimal.

Distribution of Predicted vs. Actual Like Counts
The density plot comparing predicted and actual like counts provides further insight into the model's performance. The predicted distribution aligns closely with the actual distribution for lower like counts, indicating a good fit for the majority of data points. However, as like counts increase, the two distributions diverge, reflecting the model's difficulty in accurately predicting higher engagement levels.

This discrepancy could be attributed to an imbalance in the dataset, where extreme like counts are underrepresented. Such imbalances can hinder the model's ability to learn and generalize patterns for high-value cases. Additionally, the density comparison suggests that the model underestimates values for higher like counts, likely due to a lack of training examples in this range or insufficient capacity to capture complex, non-linear relationships.

![image](https://github.com/user-attachments/assets/250589fe-6499-45d3-84a5-7fad5e30e772)
![image](https://github.com/user-attachments/assets/601ec51a-f177-48d2-abf0-f662f190bb51)
![image](https://github.com/user-attachments/assets/91cb8f90-8c7f-4566-98b3-2feca44898e7)


## Class Imbalance
Class imbalance was the main issue during expermintation, and it can be heavily observed in the following plot.
![image](https://github.com/user-attachments/assets/ca84b159-ec42-46e4-a380-ecd7b6541579)
## Possible Developments
**Feature Engineering**:
The impact of selecting relative features was observed clearly in both tasks. Hence, refining features mayl lead better distinguishing between overlapping classes.
**Weight Selection**:
One may consider different weigts for better generalization on moderately imbalanced datasets.
**Data Augmentation**:
Increase samples for underrepresented classes (e.g., "gaming") to reduce reliance on high weights.

