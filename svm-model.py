from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(df_tfidf, y_train, test_size=0.2, stratify=y_train,random_state=42)
#x_train, x_val, y_train, y_val = train_test_split(df_tfidf, y_train, test_size=0.2,random_state=42)

from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

import numpy as np
from collections import Counter


class_counts = Counter(y_train)
total_samples = len(y_train)

unique_classes, class_counts = np.unique(y_train, return_counts=True)
log_class_weights = {cls: 1 / (np.log(cnt) / np.log(20)) for cls, cnt in zip(unique_classes, class_counts)}
print("Class Weights with Log10 scaling:", log_class_weights)


# To compare with 'balanced' class weights
class_weights = compute_class_weight(
    class_weight='balanced', 
    classes=np.unique(y_train), 
    y=y_train
)
class_weight_dict = {cls: weight for cls, weight in zip(np.unique(y_train), class_weights)}
print("Class Weights:", class_weight_dict)


from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix


svm_model = LinearSVC(class_weight=log_class_weights,max_iter=2000,random_state=123)
svm_model.fit(x_train, y_train)

#@title Train Data
y_train_pred = svm_model.predict(x_train)

print("Accuracy:", accuracy_score(y_train, y_train_pred))
print("\nClassification Report:")
print(classification_report(y_train, y_train_pred, zero_division=0))

y_val_pred = svm_model.predict(x_val)

print("Accuracy:", accuracy_score(y_val, y_val_pred))
print("\nClassification Report:")
print(classification_report(y_val, y_val_pred, zero_division=0))
print("Confusion Matrix:\n", confusion_matrix(y_val, y_val_pred))

# Micro-checking misclassified instances to understand possible reasons better

val_usernames = train_usernames[len(x_train):] 
val_corpus = corpus[len(x_train):]

y_val_aligned = y_val[:len(val_usernames)]
y_val_pred_aligned = y_val_pred[:len(val_usernames)]

val_data = pd.DataFrame({
    "Username": val_usernames,
    "True Label": y_val_aligned,
    "Predicted Label": y_val_pred_aligned,
    "Captions": val_corpus
})


misclassified = val_data[val_data["True Label"] != val_data["Predicted Label"]]


print("Misclassified Samples with Captions:")
for _, row in misclassified.iterrows():
    print(f"Username: {row['Username']}")
    print(f"True Label: {row['True Label']}")
    print(f"Predicted Label: {row['Predicted Label']}")
    print(f"Captions: {row['Captions']}\n")
