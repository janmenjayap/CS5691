#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import the required libraries
import os
import pickle
import numpy as np
import pandas as pd
from statistics import mode
from sklearn.svm import SVC
from matplotlib import pyplot as plt
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


# In[ ]:


class MultinomialNaiveBayes:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.class_log_prior_ = None
        self.feature_log_prob_ = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)

        n_classes = self.classes_.shape[0]
        self.class_log_prior_ = np.zeros(n_classes)
        self.feature_log_prob_ = np.zeros((n_classes, n_features))

        for i, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.class_log_prior_[i] = np.log((X_c.shape[0] + self.alpha) / (n_samples + self.alpha * n_classes))
            self.feature_log_prob_[i] = np.log((np.sum(X_c, axis=0) + self.alpha) / (np.sum(X_c) + self.alpha * n_features))

    def predict_class_probabilities(self, X):
        return safe_sparse_dot(X, self.feature_log_prob_.T) + self.class_log_prior_

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_class_probabilities(X), axis=1)]


if __name__=='__main__':
    # In[ ]:


    seed = 87


    # In[ ]:


    os.makedirs('plots', exist_ok=True)
    os.makedirs('models', exist_ok=True)


    # In[ ]:


    # Read the CSV file into a DataFrame
    df = pd.read_csv('dataset.csv', encoding='iso-8859-1')
    data = df.where((pd.notnull(df)), '')

    data.head()


    # In[ ]:


    # Extracting features (X) and target variable (y)
    X = data['text']  # Features (email content)
    y = data['label']  # Target variable (spam or not spam)

    # Display the shape of X and y
    print("Shape of X:", X.shape)
    print("Shape of y:", y.shape)


    # In[ ]:


    data.info()


    # In[ ]:


    data.groupby('label').describe()

    # In[ ]:

    # Extracting counts of each label category
    label_counts = data.groupby('label').size()

    # Plotting the pie chart
    plt.figure(figsize=(8, 6))
    plt.pie(label_counts, labels=label_counts.index, autopct='%1.1f%%', startangle=140)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    # plt.title('Distribution of Labels')

    plt.savefig('plots/data.pdf')
    plt.show()


    # In[ ]:


    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(X, y, test_size=0.2, random_state=seed)

    # Display the shape of X and y
    print("Shape of X_train (Unvectorized):", X_train_raw.shape)
    print("Shape of y_train:", y_train_raw.shape)

    print("\nShape of X_test (Unvectorized):", X_test_raw.shape)
    print("Shape of y_test:", y_test_raw.shape)


    # In[ ]:


    X_train_raw.describe()


    # In[ ]:


    feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)

    X_train = feature_extraction.fit_transform(X_train_raw)
    X_test = feature_extraction.transform(X_test_raw)

    y_train = y_train_raw.astype('int')
    y_test = y_test_raw.astype('int')

    # Display the shape of X and y
    print("Shape of X_train (Vectorized):", X_train.shape)
    print("Shape of y_train:", y_train_raw.shape)

    print("\nShape of X_test (Vectorized):", X_test.shape)
    print("Shape of y_test:", y_test_raw.shape)

    with open('models/feature_extraction.pkl', 'wb') as file:
        pickle.dump(feature_extraction, file)


    # #### Naive Bayes

    # In[ ]:


    model_nb = MultinomialNaiveBayes()
    model_nb.fit(X_train, y_train)

    model_nb_data = {
        'alpha': model_nb.alpha,
        'class_log_prior_': model_nb.class_log_prior_,
        'feature_log_prob_': model_nb.feature_log_prob_
    }

    with open('models/multinomial_nb.pkl', 'wb') as file:
        pickle.dump(model_nb_data, file)

    train_prediction_nb = model_nb.predict(X_train)
    print('\033[1m--------------------------------------------------------\033[0m')
    print("Classification_Report (Training): \n",classification_report(y_train, train_prediction_nb, digits=4))

    train_cm_nb = confusion_matrix(y_train, train_prediction_nb)
    train_disp_nb = ConfusionMatrixDisplay(confusion_matrix=train_cm_nb)
    train_disp_nb.plot(cmap='magma')
    # plt.title('Confusion Matrix (Train)')
    plt.savefig('plots/train_cm_nb.pdf')
    plt.show()

    test_prediction = model_nb.predict(X_test)
    print('\033[1m--------------------------------------------------------\033[0m')
    print("Classification_Report (Test): \n",classification_report(y_test, test_prediction, digits=4))

    test_cm_nb = confusion_matrix(y_test, test_prediction)
    test_disp_nb = ConfusionMatrixDisplay(confusion_matrix=test_cm_nb)
    test_disp_nb.plot(cmap='magma')
    # plt.title('Confusion Matrix (Test)')
    plt.savefig('plots/test_cm_nb.pdf')
    plt.show()


    # #### SVM (Linear Kernel)

    # In[ ]:


    model_svm_linear = SVC(kernel='linear', random_state=seed)
    model_svm_linear.fit(X_train, y_train)

    with open('models/svm_linear.pkl', 'wb') as file:
        pickle.dump(model_svm_linear, file)

    train_prediction_svm_linear = model_svm_linear.predict(X_train)
    print('\033[1m--------------------------------------------------------\033[0m')
    print("Classification_Report (Training): \n",classification_report(y_train, train_prediction_svm_linear, digits=4))

    train_cm_svm_linear = confusion_matrix(y_train, train_prediction_svm_linear)
    train_disp_svm_linear = ConfusionMatrixDisplay(confusion_matrix=train_cm_svm_linear)
    train_disp_svm_linear.plot(cmap='magma')
    # plt.title('Confusion Matrix (Train)')
    plt.savefig('plots/train_cm_svm_linear.pdf')
    plt.show()

    test_prediction_svm_linear = model_svm_linear.predict(X_test)
    print('\033[1m--------------------------------------------------------\033[0m')
    print("Classification_Report (Test): \n",classification_report(y_test, test_prediction_svm_linear, digits=4))

    test_cm_svm_linear = confusion_matrix(y_test, test_prediction_svm_linear)
    test_disp_svm_linear = ConfusionMatrixDisplay(confusion_matrix=test_cm_svm_linear)
    test_disp_svm_linear.plot(cmap='magma')
    # plt.title('Confusion Matrix (Test)')
    plt.savefig('plots/test_cm_svm_linear.pdf')
    plt.show()


    # #### SVM (RBF Kernel)

    # In[ ]:


    model_svm_rbf = SVC(kernel='rbf', random_state=seed)
    model_svm_rbf.fit(X_train, y_train)

    with open('models/svm_rbf.pkl', 'wb') as file:
        pickle.dump(model_svm_rbf, file)

    train_prediction_svm_rbf = model_svm_rbf.predict(X_train)
    print('\033[1m--------------------------------------------------------\033[0m')
    print("Classification_Report (Training): \n",classification_report(y_train, train_prediction_svm_rbf, digits=4))

    train_cm_svm_rbf = confusion_matrix(y_train, train_prediction_svm_rbf)
    train_disp_svm_rbf = ConfusionMatrixDisplay(confusion_matrix=train_cm_svm_rbf)
    train_disp_svm_rbf.plot(cmap='magma')
    # plt.title('Confusion Matrix (Train)')
    plt.savefig('plots/train_cm_svm_rbf.pdf')
    plt.show()

    test_prediction_svm_rbf = model_svm_rbf.predict(X_test)
    print('\033[1m--------------------------------------------------------\033[0m')
    print("Classification_Report (Test): \n",classification_report(y_test, test_prediction_svm_rbf, digits=4))

    test_cm_svm_rbf = confusion_matrix(y_test, test_prediction_svm_rbf)
    test_disp_svm_rbf = ConfusionMatrixDisplay(confusion_matrix=test_cm_svm_rbf)
    test_disp_svm_rbf.plot(cmap='magma')
    # plt.title('Confusion Matrix (Test)')
    plt.savefig('plots/test_cm_svm_rbf.pdf')
    plt.show()