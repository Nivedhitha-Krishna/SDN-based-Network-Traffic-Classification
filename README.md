# SDN-based-Network-Traffic-Classification
Software-Defined Networking based Network  Traffic Classification using Machine Learning 
## Summary of the Code – SDN Traffic Classification using Machine Learning

Our project classifies network traffic into three categories: Browsing, VoIP, and P2P using machine learning. The main aim is to identify the type of network traffic using flow statistics without inspecting packet contents.

The implementation consists of two Python files: `traffic_classifier.py` and `predict_flow.py`.

### 1. traffic_classifier.py

This is the main program responsible for training and evaluating the machine learning models.

1. Loading the dataset: Loads the ISCX VPN-nonVPN 2016 dataset using SciPy.
2. Data preprocessing: Selects Browsing, VoIP, and P2P classes, removes missing or infinite values, keeps numerical features, and removes constant columns.
3. Splitting the dataset: Divides the 6,326 flows into 75% training data and 25% testing data. Stratified splitting maintains the proportion of each class.
4. Training models: Trains two models:
   - Random Forest: Uses 200 decision trees and combines their predictions to classify traffic.
   - SVM: Uses an RBF kernel with \\(C=10\\) to separate the three traffic classes. StandardScaler is used to standardize features for SVM.
5. Model evaluation: Calculates accuracy, precision, recall, and F1-score. It also creates a confusion matrix and performs five-fold cross-validation.
6. Saving the model: Saves the trained Random Forest model using Joblib for later predictions and generates plots to display the results.

### 2. predict_flow\.py

This program uses the saved Random Forest model to classify individual network flows.

It loads the trained model, takes the numerical features of selected flows, predicts the traffic class, and provides a confidence probability for each prediction.

The report tests nine unseen flows, with three flows from each class. All nine are classified correctly. The program also demonstrates why unknown traffic needs special handling: a class the model has not learned may still be assigned to one of its existing classes.

### 3. Important Python Functions and Libraries

| Function or library        | Purpose                                        |
| -------------------------- | ---------------------------------------------- |
| `pandas`                   | Handles and processes dataset records          |
| `train_test_split()`       | Divides data into training and testing sets    |
| `RandomForestClassifier()` | Creates the Random Forest model with 200 trees |
| `SVC()`                    | Creates the SVM classifier                     |
| `StandardScaler()`         | Standardizes features for SVM                  |
| `classification_report()`  | Calculates precision, recall, and F1-score     |
| `StratifiedKFold()`        | Creates five folds for cross-validation        |
| `cross_val_score()`        | Evaluates model accuracy across five folds     |
| `joblib`                   | Saves and loads the trained model              |

### 4. Results

- Random Forest: Achieved 99.2% test accuracy and 99.32% mean cross-validation accuracy.
- SVM: Achieved 97.9% test accuracy and 98.06% mean cross-validation accuracy.

Random Forest performed better than SVM and was selected for individual flow prediction.

### 5. Overall Working

The code first loads and cleans the network traffic dataset. It then selects useful numerical features and trains the Random Forest and SVM models. Both models are evaluated using test data and cross-validation. Finally, the trained Random Forest model is saved and used to predict the traffic class of individual flows.

Limitation: The implementation trains and evaluates the models offline. It does not use a live SDN controller or perform real-time traffic classification. The report explains how the classifier could be integrated with an SDN controller to support Quality of Service (QoS) decisions.
