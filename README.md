# CS412 Project: Instagram Influencers Category Classification and Like Count Prediction
## Name, Surname, and ID
Mohamed Osama, 29978

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

## Preprocesing
In this preprocessing pipeline, the goal was to prepare the text data by cleaning, transforming, and selecting the most relevant features. I started with text cleaning, where I used a custom preprocess_text function. This function handled Turkish-specific lowercasing with casefold, ensuring accurate treatment of characters like "ı" and "I". I removed URLs, special characters, removed numbers, and eliminated extra whitespaces. These steps ensured the text was clean and consistent, focusing only on the meaningful content.

Next step is the aggreagation of all posts for each user into a single document, joining them with newline characters to form a user-level corpus. This allowed the model to learn a language pattern for each user rather than treating individual posts in isolation. Then, TF-IDF vectorization was applied to convert the corpus into numerical vectors. I used parameters like ngram_range=(1, 3) to capture unigrams, bigrams, and trigrams, and set a maximum feature limit of 15,000 while filtering out terms appearing in fewer than 10 documents. I also integrated Turkish stopwords to exclude common but uninformative words, ensuring that the focus remained on meaningful terms.

To further refine the features, chi-squared feature selection was performed, identifying the top 5,000 features most relevant to the target labels. This reduced dimensionality and improved the interpretability and efficiency of the model. The selected features were analyzed by converting the data into a pandas DataFrame, which allowed me to inspect the most frequent and significant terms.

For the test data, I followed the same cleaning and aggregation steps as the training data to ensure consistency. The test set was transformed using the trained TF-IDF vectorizer, and the same chi-squared selection was applied to align the features across training and testing. This careful, consistent preprocessing pipeline ensured that the text data was clean, representative, and ready for model training, improving both efficiency and performance.

## Machine Learning Models
For the classification tasks, an SVC was used to predict user categories. The dataset was initially split into training and validation sets using train_test_split, with stratified sampling ensuring that the class distribution remained consistent.

A LinearSVC model from sklearn was selected for its suitability with high-dimensional text data such as the TF-IDF vectors. To address class imbalance, custom class weights were calculated using logarithmic scaling. These weights ensured that minority classes were given appropriate emphasis during training, reducing the likelihood of bias toward dominant classes. This logarithmic scaling moderated extreme weights, contributing to a more stable training process.

The LinearSVC model was trained with a maximum iteration limit of 2000 and a fixed random state for reproducibility. Model performance was evaluated using metrics such as accuracy, classification reports, and confusion matrices, which provided insights into the model's effectiveness across different categories. Misclassified instances were further analyzed by examining individual examples along with their associated captions.

For regression tasks, LightGBM was employed to predict user engagement, measured as the average like count per user. The target variable was computed by calculating the mean like count per user, with missing values replaced by zeros. LightGBM, a gradient boosting framework, was configured with hyperparameters such as learning rate, maximum depth, and the Poisson objective, which is well-suited for count-based target variables. Those parameters were result of hypertuning using GridSearchCV. Data was then split into training and validation sets, and predictions were constrained to non-negative integers to ensure validity, as like counts cannot be negative.

Model evaluation for regression was conducted using mean squared error (MSE), providing a measure of how accurately the predicted values matched the true values. Predictions on the test set were then generated and organized into a DataFrame. These classification and regression to some extent addressed challenges such as imbalanced data and high-dimensional feature spaces, presenting a reliable approach to text-based predictive modeling.

# Results
## 1. Classification Task
### Accuracy
For the classification task, multiple models were experminted on the data, namely Naive Bayes, Random Forest Tree, and SVM. Naive Bayes showed the worst accuracy so far, which is expected due to its assumption that features are independant on each other, which contradicts the core of text analysis. Random forest tree provided acceptable accuracy when tuned moderately. Further tuning may have resulted in better performance, however, tuning the parameters was computationally expensive and time consuming. The best performance in my case was achieved by the SVM model, specifically linear SVM model. Using SVM made sense due to the large size of features (fat matrix), and SVM efficiency on TF-idf based text analysis. Also, the Linear SVM was expected to have a better performance considering the use of log weights in data processing, hence, linear relations between features is assumed. The difference in various models accuracies is shown below.

![image](https://github.com/user-attachments/assets/f4857e8b-733c-4250-b3db-09410a074f85)

### Confusion Matrix
On the other hand, confusion matrix give insight into where the misclassifications happened, which allowed for devolping the model's performance. For instance, it was observed that "Food" class has the best performance with minimal misclassification (96 correct). "Tech" and "Travel" also show good performance with high accuracy.
However, "Gaming" is poorly classified, reflecting issues with rare class representation. This was observed from the low number of training instances having "gaming category". Moreover, "Mom and children" and "Sports" exhibit low accuracy, which is again due to class imbalnce and similarity in key tokens. It may also be due to inadequate feature separation.There was also substantial confusion among other classes.
**Overlaps** observed between:
"Art" and "Fashion"
"Entertainment" and "Sports"

![image](https://github.com/user-attachments/assets/7488f604-a6c2-40ba-9f31-bf888b04d83f)


## 2. Regression Task
This document provides a detailed explanation of the analysis conducted on a regression task aimed at predicting average like counts for a test dataset. The results have been evaluated using various visualizations, including residual distributions, scatter plots of predictions, and density comparisons between predicted and actual values. Below, each aspect of the analysis is described in detail, followed by insights and observations derived from the results.

### Residuals Distribution
The first graph shows the distribution of residuals, which represent the difference between actual and predicted values. A large portion of residuals is concentrated near zero, indicating that the model performs well for most test cases. However, the presence of a long tail extending to higher residual values highlights the existence of outliers where the model predictions deviate significantly from the actual data.

This behavior suggests that while the model is generally accurate, it struggles to generalize for certain data points with rare or extreme patterns. These outliers may correspond to specific cases in the dataset where features do not align well with the overall trends captured by the model. Such discrepancies could stem from noisy data, insufficient feature representation, or inherent variability in the underlying dataset.

![image](https://github.com/user-attachments/assets/601ec51a-f177-48d2-abf0-f662f190bb51)

### Predicted Like Counts for the Test Set
The second graph provides a scatter plot illustrating the predicted average like counts for individual users in the test set. The majority of predictions lie in a relatively low range, with occasional spikes indicating significantly higher values. While this pattern aligns with the expected distribution of like counts in social media datasets (where most users have moderate engagement levels and a few have disproportionately high ones), the model appears to overestimate in some cases.

This uneven distribution of predictions highlights a potential issue with overfitting to certain user profiles or data segments. The presence of extreme values suggests that the model may be overly sensitive to features associated with rare cases. While it captures the general trend of the data, its performance for users with exceptionally high or low like counts remains suboptimal.

![image](https://github.com/user-attachments/assets/91cb8f90-8c7f-4566-98b3-2feca44898e7)

###Distribution of Predicted vs. Actual Like Counts
The density plot comparing predicted and actual like counts provides further insight into the model's performance. The predicted distribution aligns closely with the actual distribution for lower like counts, indicating a good fit for the majority of data points. However, as like counts increase, the two distributions diverge, reflecting the model's difficulty in accurately predicting higher engagement levels.

This discrepancy could be attributed to an imbalance in the dataset, where extreme like counts are underrepresented. Such imbalances can hinder the model's ability to learn and generalize patterns for high-value cases. Additionally, the density comparison suggests that the model underestimates values for higher like counts, likely due to a lack of training examples in this range or insufficient capacity to capture complex, non-linear relationships.

![image](https://github.com/user-attachments/assets/250589fe-6499-45d3-84a5-7fad5e30e772)


## Class Imbalance
Class imbalance was the main issue during expermintation, and it can be heavily observed in the following plot, which shows Balanced vs. Log Scaling Weights.
The **Balanced** weights are significantly higher for underrepresented classes like "gaming." This suggests that the balanced strategy attempts to compensate for class imbalances by assigning disproportionately high weights to rare classes. On the other hand, **Log Scaling** weights are relatively uniform compared to the balanced weights. This method smooths the impact of imbalanced classes, avoiding extreme weights. It was extremely effective in my case.

![image](https://github.com/user-attachments/assets/ca84b159-ec42-46e4-a380-ecd7b6541579)

## Possible Developments
1. **Feature Engineering**:
The impact of selecting relative features was observed clearly in both tasks. Hence, refining features mayl lead better distinguishing between overlapping classes.
2. **Weight Selection**:
One may consider different weigts for better generalization on moderately imbalanced datasets.
3. **Data Augmentation**:
Increase samples for underrepresented classes (e.g., "gaming") to reduce reliance on high weights.

