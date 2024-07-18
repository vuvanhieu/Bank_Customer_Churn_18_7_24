import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, roc_auc_score
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, VotingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from imblearn.combine import SMOTETomek
import logging
from sklearn.metrics import precision_recall_fscore_support, recall_score, accuracy_score
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to save shapes information to a CSV file
def save_shapes_info(shapes_info, result_folder):
    shapes_df = pd.DataFrame.from_dict(shapes_info, orient='index', columns=['Shape'])
    shapes_df.to_csv(os.path.join(result_folder, 'shapes_info.csv'))
    logging.debug(f"Shapes information saved to {result_folder}/shapes_info.csv")

def save_model(model, model_name, result_folder):
    model_path = os.path.join(result_folder, f"{model_name}.pkl")
    joblib.dump(model, model_path)
    logging.debug(f"Model {model_name} saved to {model_path}")

def load_model(model_name, result_folder):
    model_path = os.path.join(result_folder, f"{model_name}.pkl")
    model = joblib.load(model_path)
    logging.debug(f"Model {model_name} loaded from {model_path}")
    return model

def create_model_folder(result_folder, model_name):
    logging.debug(f"Creating model folder for {model_name}.")
    model_folder = os.path.join(result_folder, model_name)
    os.makedirs(model_folder, exist_ok=True)
    logging.debug(f"Created model folder: {model_folder}")
    return model_folder

def save_metrics_to_csv(metrics_list, filename):
    logging.debug("Saving metrics to CSV.")
    df = pd.DataFrame(metrics_list)
    df.to_csv(filename, index=False)
    logging.debug(f"Metrics saved to CSV: {filename}")

def plot_confusion_matrix(cm, model_folder):
    logging.debug("Plotting confusion matrix.")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    plt.tight_layout()
    plt.savefig(os.path.join(model_folder, 'confusion_matrix.png'))
    plt.close(fig)
    logging.debug(f"Confusion matrix plot saved: {model_folder}/confusion_matrix.png")

def plot_precision_recall_curve(precision, recall, model_folder):
    logging.debug("Plotting precision-recall curve.")
    fig, ax = plt.subplots()
    ax.plot(recall, precision, color='blue', lw=2)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    plt.tight_layout()
    plt.savefig(os.path.join(model_folder, 'precision_recall_curve.png'))
    plt.close(fig)
    logging.debug(f"Precision-Recall curve plot saved: {model_folder}/precision_recall_curve.png")

def plot_metrics_bar(accuracy, precision, recall, f1_score, sensitivity, specificity, auc_score, model_folder):
    logging.debug("Plotting model metrics bar chart.")
    fig, ax = plt.subplots()
    metrics = [accuracy, precision, recall, f1_score, sensitivity, specificity, auc_score]
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Sensitivity', 'Specificity', 'AUC Score']
    ax.bar(metric_names, metrics, color=['gold', 'lightgreen', 'lightcoral', 'lightskyblue', 'lightpink', 'lightyellow', 'lightgray'])
    for index, value in enumerate(metrics):
        ax.text(index, value, f'{value:.4f}', ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(os.path.join(model_folder, 'model_metrics.png'))
    plt.close(fig)
    logging.debug(f"Model metrics bar plot saved: {model_folder}/model_metrics.png")

def plot_metric_comparison(metrics_dict, metric_name, result_folder):
    logging.debug(f"Plotting comparison for {metric_name}.")
    fig, ax = plt.subplots()
    model_names = list(metrics_dict.keys())
    metric_values = [metrics[metric_name] for metrics in metrics_dict.values()]
    
    ax.bar(model_names, metric_values, color=['gold', 'lightgreen', 'lightcoral', 'lightskyblue', 'lightpink', 'lightyellow', 'lightgray'])
    for index, value in enumerate(metric_values):
        ax.text(index, value, f'{value:.4f}', ha='center', va='bottom')
    
    ax.set_ylabel(metric_name)
    plt.xticks(rotation=45)  # Rotate x-axis labels 45 degrees
    plt.tight_layout()
    plt.savefig(os.path.join(result_folder, f'{metric_name}_comparison.png'))
    plt.close(fig)
    logging.debug(f"{metric_name} comparison plot saved: {result_folder}/{metric_name}_comparison.png")

def cross_val_metrics(model, X, y, metrics_dict, model_name):
    logging.debug("Starting cross-validation metrics evaluation.")
    
    scores = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1',
        'roc_auc': 'roc_auc'
    }
    
    for name, score in scores.items():
        score_values = cross_val_score(model, X, y, cv=5, scoring=score)
        metrics_dict[model_name][f'{name}_cv_mean'] = score_values.mean()
        metrics_dict[model_name][f'{name}_cv_std'] = score_values.std()
        print(f"[{name}] : {score_values.mean():.5f} (+/- {score_values.std():.5f})")
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    sensitivity_scores = []
    specificity_scores = []
    
    X_reset = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
    y_reset = pd.Series(y) if not isinstance(y, pd.Series) else y
    
    for train_index, test_index in skf.split(X_reset, y_reset):
        X_train, X_test = X_reset.iloc[train_index], X_reset.iloc[test_index]
        y_train, y_test = y_reset.iloc[train_index], y_reset.iloc[test_index]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        
        sensitivity_scores.append(sensitivity)
        specificity_scores.append(specificity)
    
    metrics_dict[model_name]['sensitivity_cv_mean'] = np.mean(sensitivity_scores)
    metrics_dict[model_name]['sensitivity_cv_std'] = np.std(sensitivity_scores)
    metrics_dict[model_name]['specificity_cv_mean'] = np.mean(specificity_scores)
    metrics_dict[model_name]['specificity_cv_std'] = np.std(specificity_scores)
    
    print(f"Sensitivity: {np.mean(sensitivity_scores):.5f} (+/- {np.std(sensitivity_scores):.5f})")
    print(f"Specificity: {np.mean(specificity_scores):.5f} (+/- {np.std(specificity_scores):.5f})")

def evaluate_model(model, testX, testy, result_folder, model_name, metrics_dict):
    logging.debug("Starting model evaluation.")
    
    y_pred = model.predict(testX)
    logging.debug(f"Predictions shape: {y_pred.shape}")
    logging.debug(f"Predictions: {y_pred[:5]}")  # Log first 5 predictions for inspection
    
    y_pred_classes = y_pred
    logging.debug(f"Predicted classes: {y_pred_classes[:5]}")  # Log first 5 predicted classes for inspection
    
    accuracy = np.mean(y_pred_classes == testy)
    logging.debug(f"Accuracy: {accuracy}")
    
    precision, recall, f1_score, _ = precision_recall_fscore_support(testy, y_pred_classes, average='binary')
    logging.debug(f"Precision: {precision}, Recall: {recall}, F1 Score: {f1_score}")
    
    sensitivity = recall
    specificity = recall_score(testy, y_pred_classes, pos_label=0)
    logging.debug(f"Specificity: {specificity}")
    
    auc_score = roc_auc_score(testy, y_pred)
    logging.debug(f"AUC Score: {auc_score}")
    
    plot_roc_curve(testy, y_pred, auc_score, result_folder, model_name)
    
    cm = confusion_matrix(testy, y_pred_classes)
    plot_confusion_matrix(cm, result_folder)
    
    precision_vals, recall_vals, _ = precision_recall_curve(testy, y_pred)
    plot_precision_recall_curve(precision_vals, recall_vals, result_folder)
    
    plot_metrics_bar(accuracy, precision, recall, f1_score, sensitivity, specificity, auc_score, result_folder)
    
    metrics = {
        'Model Name': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1_score,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'AUC Score': auc_score
    }
    metrics_dict[model_name] = metrics
    logging.debug(f"Metrics for model {model_name}: {metrics}")

def plot_roc_curve(y_test, y_pred, auc_score, model_folder, model_name):
    logging.debug("Plotting ROC curve.")
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    
    roc_curve_path = os.path.join(model_folder, f"{model_name}_roc_curve.png")
    plt.savefig(roc_curve_path)
    plt.close()
    
    logging.debug(f"ROC curve saved to {roc_curve_path}")

def create_directories(base_dir):
    logging.debug(f"Creating directories if not exist: {base_dir}")
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    return base_dir

# Updated load_clean_data function to include saving shapes information
def load_clean_data(data_path):
    logging.debug("Loading and cleaning data.")
    bank = pd.read_csv(data_path)
    
    # Drop unnecessary columns
    bank = bank.drop(bank.columns[21:24], axis=1)
    
    # Check for missing data
    print("Shape of dataset:", bank.shape)
    bank.dropna(inplace=True)
    print("Shape after dropping NA:", bank.shape)
    
    # Create dummy variable for Customer Attrition
    bank2 = bank.copy()
    att = pd.get_dummies(bank2['Attrition_Flag'], drop_first=True)
    bank2 = bank2.drop('Attrition_Flag', axis=1)
    bank2 = bank2.join(att)
    
    # Encode categorical variables
    categorical_columns = ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']
    bank2 = pd.get_dummies(bank2, columns=categorical_columns, drop_first=True)
    
    # Check the column names to ensure the correct column is used
    print("Columns after creating dummy variables:", bank2.columns)
    
    # Extract features and target
    X = bank2.drop(['Existing Customer'], axis=1)
    y = bank2['Existing Customer'].astype(int)
    
    # Scale the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Combine X and y into a single DataFrame and save to CSV
    combined_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    combined_df['Churn'] = y
    
    cleaned_data_path = os.path.join(os.path.dirname(data_path), 'cleaned_data')
    os.makedirs(cleaned_data_path, exist_ok=True)
    
    combined_df.to_csv(os.path.join(cleaned_data_path, 'cleaned_data.csv'), index=False)
    
    logging.debug(f"Cleaned data saved to {cleaned_data_path}")
    
    trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.2, random_state=42)
    testX, valX, testy, valy = train_test_split(testX, testy, test_size=0.5, random_state=42)
    
    logging.debug(f"trainX shape: {trainX.shape}, trainy shape: {trainy.shape}")
    logging.debug(f"testX shape: {testX.shape}, testy shape: {testy.shape}")
    logging.debug(f"valX shape: {valX.shape}, valy shape: {valy.shape}")

    shapes_info = {
        'X shape': str(X.shape),
        'y shape': str(y.shape),
        'trainX shape': str(trainX.shape),
        'testX shape': str(testX.shape),
        'trainy shape': str(trainy.shape),
        'testy shape': str(testy.shape),
        'valX shape': str(valX.shape),
        'valy shape': str(valy.shape)
    }
    
    return bank, bank2, X, y, trainX, testX, trainy, testy, valX, valy, shapes_info


# Updated save_all_models function to include saving model parameters to a CSV file
def save_all_models(models, result_folder):
    model_params = {}
    for model_name, model in models.items():
        save_model(model, model_name, result_folder)
        # Save model parameters
        model_params[model_name] = model.get_params()
    
    params_df = pd.DataFrame(model_params).T
    params_df.to_csv(os.path.join(result_folder, 'model_params.csv'))
    logging.debug(f"Model parameters saved to {result_folder}/model_params.csv")
    

def load_all_models(model_names, result_folder):
    models = []
    for model_name in model_names:
        models.append(load_model(model_name, result_folder))
    return models

def stacked_dataset(members, inputX):
    stackX = np.zeros((inputX.shape[0], len(members)))  # Initialize an empty array
    for i, model in enumerate(members):
        yhat = model.predict(inputX)
        if yhat.ndim > 1:
            yhat = yhat[:, 1] if yhat.shape[1] > 1 else yhat.ravel()
        stackX[:, i] = yhat
    return stackX

def build_dnn_model(input_shape):
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(input_shape,)))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def fit_stacked_model(members, inputX, inputy):
    stackedX = stacked_dataset(members, inputX)
    input_shape = stackedX.shape[1]
    model = build_dnn_model(input_shape)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(stackedX, inputy, epochs=150, batch_size=32, verbose=1, validation_split=0.2, callbacks=[early_stopping])
    return model

def stacked_prediction(members, model, inputX):
    stackedX = stacked_dataset(members, inputX)
    yhat = model.predict(stackedX)
    yhat_classes = (yhat > 0.5).astype(int).ravel()
    return yhat_classes

# Main function to include saving shapes information and all models
def main():
    logging.debug("Starting main function.")
    directory_work = os.getcwd()
    experiment_folder = '/data2/cmdir/home/hieuvv/CIT/BankChurners'
    model_name = "scenario_2_KNN_RF_SVM_XGB_to_DNN_BankChurners"
    result_folder = create_directories(os.path.join(experiment_folder, model_name))

    data_path = os.path.join(experiment_folder, 'BankChurners.csv')
    bank, bank2, X, y, trainX, testX, trainy, testy, valX, valy, shapes_info = load_clean_data(data_path)


    logging.debug(f"Initial trainX shape: {trainX.shape}")
    logging.debug(f"Initial testX shape: {testX.shape}")
    logging.debug(f"Initial trainy shape: {trainy.shape}")
    logging.debug(f"Initial testy shape: {testy.shape}")

    metrics_dict = {}
    
    models = {
        'KNN': KNeighborsClassifier(),
        'RF': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(probability=True),
        'XGB': XGBClassifier()
    }

    for model_name, model in models.items():
        model_folder = create_model_folder(result_folder, model_name)
        logging.debug(f"Training model: {model_name}")
        model.fit(trainX, trainy)
        evaluate_model(model, testX, testy, model_folder, model_name, metrics_dict)
        cross_val_metrics(model, X, y, metrics_dict, model_name)
        save_model(model, model_name, result_folder)

    model_names = ['KNN', 'RF', 'SVM', 'XGB']

    members = load_all_models(model_names, result_folder)
    
    stacked_model = fit_stacked_model(members, trainX, trainy)
    
    stacked_model_name = 'scenario_2'
    stacked_model_folder = create_model_folder(result_folder, stacked_model_name)
    
    # Transform testX for stacked model evaluation
    testX_stacked = pd.DataFrame(testX) if not isinstance(testX, pd.DataFrame) else testX
    testX_stacked = stacked_dataset(members, testX_stacked)
    evaluate_model(stacked_model, testX_stacked, testy, stacked_model_folder, stacked_model_name, metrics_dict)
    cross_val_metrics(stacked_model, testX_stacked, testy, metrics_dict, stacked_model_name)
    stacked_model.save(os.path.join(result_folder, f"{stacked_model_name}.h5"))

    save_metrics_to_csv(list(metrics_dict.values()), os.path.join(result_folder, 'final_metrics_scenario_2.csv'))
    save_shapes_info(shapes_info, result_folder)
    save_all_models(models, result_folder)

    # Plot comparison for each metric
    metrics_to_compare = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Sensitivity', 'Specificity', 'AUC Score']
    for metric in metrics_to_compare:
        plot_metric_comparison(metrics_dict, metric, result_folder)

if __name__ == "__main__":
    main()
