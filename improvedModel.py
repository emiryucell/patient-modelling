import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, auc, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, silhouette_score
from sklearn.model_selection import KFold
from sklearn.utils import resample
from matplotlib.artist import setp

def remove_outliers(df, numerical_cols, threshold=1.5):
    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df

def feature_normalization(df, numerical_cols):
    # Exclude specified columns from the list of numerical columns to normalize
    scaler = MinMaxScaler()
    # Normalize only the columns that are not excluded
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    return df

def balance_data_oversampling(data):
    df_majority = data[data.readmitted==0]
    df_minority = data[data.readmitted==1]
    
    # Upsample minority class
    df_minority_upsampled = resample(df_minority, 
                                     replace=True,     
                                     n_samples=len(df_majority),   
                                     random_state=123)
    
    # Combine majority class with upsampled minority class
    df_upsampled = pd.concat([df_majority, df_minority_upsampled])
    
    # Display new class counts
    print(df_upsampled.readmitted.value_counts())
    
    return df_upsampled

def get_model(classifier_type):
    # Allowing for different types of classifiers
    if classifier_type == 'LogisticRegression':
        return LogisticRegression(max_iter=1000, random_state=42)
    elif classifier_type == 'RandomForest':
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=4,
            max_features='sqrt',
            random_state=42
        )
    else:
        raise ValueError("Unsupported classifier type")

def process_data_v2(data, oversampling_option=False):
    print("Shape of the data:\n", data.shape)
    
    # Dropping duplicates
    data.drop_duplicates(subset=['patient_nbr'], keep='last', inplace=True)
    print(data.shape)

    # Dropping near zero variance columns and unnecessary columns
    columns_to_delete = [
        'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'acetohexamide',
        'tolbutamide', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide', 'examide',
        'citoglipton', 'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone',
        'metformin-rosiglitazone', 'metformin-pioglitazone', 'payer_code', 'weight', 'medical_specialty', 'encounter_id', 'max_glu_serum', 'patient_nbr']
    data.drop(columns=columns_to_delete, inplace=True)

    data['gender'].value_counts()
    # Dropping since only 3 rows have unknown
    data.drop(data[data['gender']=='Unknown/Invalid'].index, inplace=True)
    print(data.shape)

    data.replace('?', np.nan, inplace=True)
    data.replace('Unknown/Invalid', np.nan, inplace=True)
    # Fetching all the values of the columns
    data['discharge_disposition_id'].unique()
    
    not_alive_patients = data[data['discharge_disposition_id'].isin([11, 19, 20, 21])].index
    # Dropping the identified rows in place
    data.drop(index=not_alive_patients, inplace=True)
    print(data.shape)

    # Remapping it again to 0 and 1
    data['readmitted'] = data['readmitted'].map({'<30': 1, '>30': 0, 'NO': 0})
    
    # Changing age to the mean value instead of the range
    data.loc[(data[data['age'] =='[0-10)'].index), 'age'] = 5
    data.loc[(data[data['age'] =='[10-20)'].index), 'age'] = 15
    data.loc[(data[data['age'] =='[20-30)'].index), 'age'] = 25
    data.loc[(data[data['age'] =='[30-40)'].index), 'age'] = 35
    data.loc[(data[data['age'] =='[40-50)'].index), 'age'] = 45
    data.loc[(data[data['age'] =='[50-60)'].index), 'age'] = 55
    data.loc[(data[data['age'] =='[60-70)'].index), 'age'] = 65
    data.loc[(data[data['age'] =='[70-80)'].index), 'age'] = 75
    data.loc[(data[data['age'] =='[80-90)'].index), 'age'] = 85
    data.loc[(data[data['age'] =='[90-100)'].index), 'age'] = 95


    data['race'].dropna(inplace=True)
    print(data.shape)

    print(data.columns.values)

    for column in ['diag_1', 'diag_2', 'diag_3']:
        # Ensure the column is of type string for string operations
        data[column] = data[column].astype(str)
        
        # Apply the categorization logic to each column based on Table 2 [Ref:https://www.hindawi.com/journals/bmri/2014/781670/tab2/]
        data[column] = np.select(
            [
                data[column].str.contains('V') | data[column].str.contains('E'),
                data[column].str.contains('250'),
                ((pd.to_numeric(data[column], errors='coerce').between(390, 459)) | (pd.to_numeric(data[column], errors='coerce') == 785)),
                ((pd.to_numeric(data[column], errors='coerce').between(460, 519)) | (pd.to_numeric(data[column], errors='coerce') == 786)),
                ((pd.to_numeric(data[column], errors='coerce').between(520, 579)) | (pd.to_numeric(data[column], errors='coerce') == 787)),
                ((pd.to_numeric(data[column], errors='coerce').between(580, 629)) | (pd.to_numeric(data[column], errors='coerce') == 788)),
                pd.to_numeric(data[column], errors='coerce').between(140, 239),
                pd.to_numeric(data[column], errors='coerce').between(710, 739),
                pd.to_numeric(data[column], errors='coerce').between(800, 999),
            ], 
            [
                'Other', 'Diabetes', 'Circulatory', 'Respiratory', 
                'Digestive', 'Genitourinary', 'Neoplasms', 
                'Musculoskeletal', 'Injury'
            ], 
            default='Other'
        )
    print("Before removing outliers:", data.shape)
    outlier_cols = ['time_in_hospital', 'num_procedures', 'num_medications', 'number_diagnoses', 'num_lab_procedures']
    data = remove_outliers(data, outlier_cols)
    print("After removing outliers:", data.shape)
    normalization_columns = ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications', 'number_diagnoses', 'number_outpatient', 'number_emergency', 'number_inpatient']
    data = feature_normalization(data, normalization_columns)

    # Transform the categorical columns into dummy variables
    cat_cols = ['race', 'gender', 'age', 'admission_type_id' , 'discharge_disposition_id', 'admission_source_id', 'diag_1', 'diag_2', 'diag_3', 'A1Cresult', 'metformin', 'glipizide', 'glyburide', 'pioglitazone', 'rosiglitazone', 'insulin', 'change', 'diabetesMed']
    data = pd.get_dummies(data, columns=cat_cols, drop_first=True)
    if oversampling_option:
        print("Oversampling the data...")
        data = balance_data_oversampling(data)
    return data

def perform_rfe_and_evaluate_with_kfold(X, y, classifier_type, n_features_to_select=5, n_splits=5):
    # Initialize the model
    model = get_model(classifier_type)

    # RFE
    rfe = RFE(estimator=model, n_features_to_select=n_features_to_select)
    rfe = rfe.fit(X, y)
    X_rfe = X[X.columns[rfe.support_]]
    
    print("Selected Features:")
    print(X_rfe.columns)
    
    # K-Fold Cross-Validation setup
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Lists to store results and predictions of each fold
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    roc_aucs = []
    tprs = []
    mean_fpr = np.linspace(0, 1, 100)
    
    fig, ax = plt.subplots()
    
    for i, (train_index, test_index) in enumerate(kf.split(X_rfe)):
        X_train, X_test = X_rfe.iloc[train_index], X_rfe.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions and predict probabilities
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Compute metrics
        accuracies.append(accuracy_score(y_test, y_pred))
        precisions.append(precision_score(y_test, y_pred))
        recalls.append(recall_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred))
        roc_auc = auc(*roc_curve(y_test, y_pred_proba)[:2])
        roc_aucs.append(roc_auc)
        
        # Compute ROC curve and AUC for this fold
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        ax.plot(fpr, tpr, lw=1, alpha=0.3, label=f'ROC fold {i+1} (AUC = {roc_auc:.2f})')
        tprs.append(np.interp(mean_fpr, fpr, tpr))
    
    # Plot the mean ROC curve
    mean_tpr = np.mean(tprs, axis=0)
    mean_auc = auc(mean_fpr, mean_tpr)
    ax.plot(mean_fpr, mean_tpr, color='blue',
            label=f'Mean ROC (AUC = {mean_auc:.2f})', lw=2, alpha=0.8)
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='red', label='Chance', alpha=0.8)
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title="ROC Curve")
    ax.legend(loc="lower right")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.show()
    
    # Print the average of the metrics
    print(f"\nK-Fold CV Metrics (Averages) across {n_splits} folds:")
    print(f"Accuracy: {np.mean(accuracies)}")
    print(f"Precision: {np.mean(precisions)}")
    print(f"Recall: {np.mean(recalls)}")
    print(f"F1 Score: {np.mean(f1_scores)}")
    print(f"ROC AUC: {np.mean(roc_aucs)}")
    return X_rfe.columns, model

def find_optimal_clusters(data, max_k):
    distortions = []
    K = range(2, max_k + 1)

    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42).fit(data)
        distortions.append(kmeans.inertia_)

    # Plotting the Elbow Method
    plt.figure(figsize=(8, 6))  # Adjusted figure size for a single plot
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()


def visualize_clusters(X_pca, cluster_labels):
    # Visualizing the clusters
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=cluster_labels, palette='viridis', alpha=0.7, legend="full")
    plt.title('Clusters Visualized After PCA')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title='Cluster')
    plt.show()



def print_cluster_profiles(data, cluster_labels):
    # Calculate the mean values of each cluster
    clustered_data = data.assign(cluster=cluster_labels)
    profile = clustered_data.groupby('cluster').mean()
    print("Cluster profiles (mean values):")
    print(profile.transpose())



def train_local_classifiers_and_evaluate(X, y, cluster_labels, classifer_type):
    # Train local classifiers for each cluster and evaluate them
    unique_clusters = np.unique(cluster_labels)
    metrics_summary = {}  # Dictionary to store metrics for each cluster

    for cluster in unique_clusters:
        X_cluster = X[cluster_labels == cluster]
        y_cluster = y[cluster_labels == cluster]

        # Split the data of the cluster
        X_train, X_test, y_train, y_test = train_test_split(X_cluster, y_cluster, test_size=0.3, random_state=42)

        # Initialize and train a logistic regression model
        model = get_model(classifer_type)

        # model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else [0] * len(y_pred)  # Use predict_proba if available

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba) if hasattr(model, "predict_proba") else "N/A"

        # Store metrics
        metrics_summary[cluster] = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'ROC AUC': roc_auc
        }

        # Print metrics
        print(f"Metrics for Cluster {cluster}:")
        print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, ROC AUC: {roc_auc}")

    return metrics_summary


def main(oversampling_option=False):
    # Load the dataset
    data_v2 = pd.read_csv('diabetic_data.csv')
    data_v2 = process_data_v2(data_v2, oversampling_option=oversampling_option)
    
    X = data_v2.drop('readmitted', axis=1)
    y = data_v2['readmitted']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    for classifier_type in ['LogisticRegression', 'RandomForest']:
        print(f"\nRunning analysis with {classifier_type}...")
        # Perform RFE and evaluate the model using K-fold CV
        selected_features, model = perform_rfe_and_evaluate_with_kfold(X_train, y_train, classifier_type, n_features_to_select=5, n_splits=5)

        predUnseenData = model.predict(X_test[selected_features])

        print(f"\nThe accuracy score of {classifier_type} is:", accuracy_score(y_test, predUnseenData))

        # Determine the optimal number of clusters
        find_optimal_clusters(X, max_k=10)
        
        # After determining the optimal number of clusters, fit the KMeans model
        while True:
            try:
                optimal_k = int(input("Enter the optimal number of clusters: "))
                break
            except ValueError:
                print("Invalid input. Please enter a valid integer.")
        kmeans = KMeans(n_clusters=optimal_k, random_state=42).fit(X)
        
        # Applying PCA for visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        # Add the cluster labels and PCA components to the DataFrame
        data_v2['cluster'] = kmeans.labels_
        data_v2['PCA1'] = X_pca[:, 0]
        data_v2['PCA2'] = X_pca[:, 1]
        
        visualize_clusters(X_pca, data_v2['cluster'])

        # Evaluate clusters
        print_cluster_profiles(data_v2, data_v2['cluster'])

        # Train local classifiers for each cluster and evaluate them
        print("\nLocal Classifier Evaluation:")
        train_local_classifiers_and_evaluate(X, y, data_v2['cluster'].values, classifier_type)

if __name__ == '__main__':
    # Run the main function with and without oversampling
    main(oversampling_option=False)
    main(oversampling_option=True)