import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve, auc
from sklearn.model_selection import KFold
from sklearn.utils import resample

def process_data(data):
    print("Shape of the data:\n", data.shape)
    
    data.drop('encounter_id', axis=1, inplace=True)
    
    missing_values_before = data.isnull().sum()
    print("\nSummary of missing values before replace:\n", missing_values_before)
    
    data.replace('?', np.nan, inplace=True)

    missing_values_after = data.isnull().sum()
    print("\nSummary of missing values after replace:\n", missing_values_after)

    data['readmitted'] = data['readmitted'].map({'<30': 1, '>30': 0, 'NO': 0})
    
    print("\nData types of each column:\n", data.dtypes)

    missing_percent = data.isnull().mean() * 100
    
    columns_to_drop = missing_percent[missing_percent > 90].index
    data.drop(columns=columns_to_drop, inplace=True)
    
    # Dropping cols such as payer_code and medical_specialty since they don't play a major role in predicting the target variable
    columns_to_delete = [
        'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'acetohexamide',
        'tolbutamide', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide', 'examide',
        'citoglipton', 'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone',
        'metformin-rosiglitazone', 'metformin-pioglitazone', 'payer_code', 'medical_specialty', 'patient_nbr']

    data.drop(columns=columns_to_delete, inplace=True)
    
    data.dropna(axis=0, how='any', inplace=True)
    # TODO: Check if A1CResult plays a major role in predicting the target variable
    # TODO: Check if removed column values can be replaced with mean or mode
    print("\nSummary statistics of numerical columns:\n", data.describe())
    
    return data

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

def data_visualisation(data, categorical_int_cols):
    # distribution of unique classes of the target variable
    ax = sns.barplot(x='readmitted', y='readmitted', estimator=lambda x: len(x) / len(data) * 100, 
                     data=data, hue="readmitted", legend=False)
    
    for container in ax.containers:
        ax.bar_label(container, fmt='%.f%%')

    ax.set_ylabel('Percentage (%)')
    sns.set_theme(rc={"figure.figsize":(10, 7)})
    plt.show()

    # count of number of readmitted cases against age
    value_counts = data.sort_values('age').groupby('age')['readmitted'].value_counts().unstack()    
    fig, ax3 = plt.subplots()

    bars = ax3.bar(value_counts.index, value_counts[1])
    ax3.bar_label(bars)
    
    plt.xlabel("Age Groups")
    plt.ylabel("Readmitted Cases")
    plt.title("Count of number of readmitted cases against age")
    plt.show()

    # count of target variable against the number of medications
    ax2 = sns.countplot(x="num_medications", data=data, hue="readmitted", legend=False)
    
    for container in ax2.containers:
        ax2.bar_label(container)

    ax2.get_legend_handles_labels()
    target_unique_classes = data['readmitted'].value_counts().index
    ax2.legend(labels=target_unique_classes, title="readmitted", loc="upper right")
    
    sns.set_theme(rc={"figure.figsize":(10, 7)})
    plt.xticks(rotation=45, ha='right')
    plt.show()


    # Creating a new DataFrame with only the specified numerical columns
    num_df = data.select_dtypes(include='number')
    num_df.drop(categorical_int_cols, axis=1, inplace=True)

    plot_correlation_matrix(num_df)
    plot_scatter_matrix(num_df)
        
    gender_plot = sns.countplot(x = 'gender', data = data, hue = 'readmitted')
    gender_plot.figure.set_size_inches(12, 12)
    gender_plot.legend(title = 'Readmitted', labels = ['No', 'Yes'])
    gender_plot.axes.set_title('Readmission based on Gender')
    plt.show()


    fig, ax = plt.subplots(figsize=(10,15), ncols=1, nrows=3)  # Adjusted for 1 column and 3 rows
    sns.countplot(x="readmitted", data=data, ax=ax[0])  # Plot 1 in the first row
    sns.countplot(x="race", data=data, ax=ax[1])        # Plot 2 in the second row
    sns.countplot(x="gender", data=data, ax=ax[2])      # Plot 3 in the third row
    plt.tight_layout()  # Adjust the layout to make sure there's no overlap
    plt.show()

def plot_scatter_matrix(num_df):

    pd.plotting.scatter_matrix(num_df, alpha=0.2, figsize=(20, 20), diagonal='kde')
    plt.suptitle('Scatter Matrix for Selected Numerical Features')
    plt.show()

def plot_correlation_matrix(num_df):

    # Calculate the correlation matrix
    corr_matrix= num_df.corr()

    # Drop the 'NaN' correlations
    plt.figure(figsize=(12, 12))

    # Visualisation the correlation matrix
    sns.heatmap(corr_matrix,annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.xticks(rotation=45, ha='right')

    # Add labels for axes
    plt.xlabel('Features')
    plt.ylabel('Features')

    #TODO add labels to  the xticks
    plt.title("Correlation matrix")
    plt.xlabel("Numerical Features")
    plt.ylabel("Numerical Features")
    plt.show()

    # Threshold for high correlation (can be adjusted)
    # TODO Evaluate hıghly corr
    threshold = 0.2
    # Declaration of pairs from matrix
    highly_correlated_pairs = corr_matrix.unstack().sort_values(kind="quicksort", ascending=False)
    # Removing self-correlation and correlations below than the threshold
    highly_correlated_pairs = highly_correlated_pairs[(abs(highly_correlated_pairs) > threshold) & (highly_correlated_pairs < 1)]

    # Listing highly correlated pairs
    print("Highly correlated pairs:\n", highly_correlated_pairs.to_string())



def plot_avg_lab_procedures_by_race(data):
    # Bar Chart of Average Number of Lab Procedures by Race
    average_lab_procedures_by_race = data.groupby('race')['num_lab_procedures'].mean().reset_index()
    sns.barplot(data=average_lab_procedures_by_race, x="race", y='num_lab_procedures')
    plt.figure(figsize=(12, 12))
    plt.title('Average Number of Lab Procedures by Race')
    plt.xticks(rotation=45)
    plt.show()


def evaluate_model_performance(data):
    # Skipping diag_1, diag_2 and diag_3 since they have too many distinct values
    data = data.drop(['diag_1', 'diag_2', 'diag_3'], axis=1)

    # Convert categorical variables to dummy variables
    cat_cols = ['race', 'gender', 'age', 'admission_type_id' , 'discharge_disposition_id', 'admission_source_id', 
                'A1Cresult', 'metformin', 'glipizide', 'glyburide', 'pioglitazone', 'rosiglitazone', 'insulin', 'change', 'diabetesMed']
    data = pd.get_dummies(data, columns=cat_cols, drop_first=True)

    print('The shape of the data after converting categorical variables to dummy variables:', data.shape)
    # print('The columns of the data after converting categorical variables to dummy variables:', data.columns.values)

    # Splitting dataset into features (X) and target (y)
    X = data.drop('readmitted', axis=1)
    y = data['readmitted']

    # Splitting the dataset into training and testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Feature selection with RFE
    model = LogisticRegression(max_iter=1000, solver='liblinear')
    rfe = RFE(estimator=model, n_features_to_select=20)
    rfe.fit(X_train, y_train)

    # Selected features
    selected_features = X.columns[rfe.support_]

    # Fitting model with selected features
    model.fit(X_train[selected_features], y_train)

    # Evaluate the model with K-Fold cross-validation
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train[selected_features], y_train, cv=kf)

    # Model predictions
    y_pred = model.predict(X_test[selected_features])

    # Performance metrics
    overall_metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "ROC AUC": roc_auc_score(y_test, y_pred)
    }

    # confusion matrix
    confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)

    fig, ax = plt.subplots(figsize=(4,4))
    ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.tight_layout()

    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(20)
    for item in (ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(15)
    plt.show()

    # Calculate precision and recall. Also the Area Under the Curve (AUC) for precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
    auc_score = auc(recall, precision)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'Precision-Recall Curve (AUC = {auc_score:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.show()

    # ROC curve
    # Predict probabilities & get probability of positive class
    y_pred_proba = model.predict_proba(X_test[selected_features])[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba, pos_label=0)

    fig = plt.figure(figsize=(7,7))
    plt.plot(fpr, tpr, lw=2, label='Logistic Regression')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random guessing')
    plt.plot([0, 0, 1], [0, 1, 1], linestyle='-.', alpha=0.5, color='red', label='Perfect')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.legend(loc=4, prop={'size': 18})

    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(20)
    for item in (ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(15)
    plt.show()

    # Output the results
    return {
        "Selected Features": selected_features.tolist(),
        "Cross-Validation Score": cv_scores.mean(),
        "Overall Metrics": overall_metrics
    }


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

def main():
    data = pd.read_csv('diabetic_data.csv')
    data = process_data(data)
    print('The shape of the data after processing:', data.shape)
    target_var = ['readmitted']
    print('The numerical cols are:', data.select_dtypes(include='number').columns.values)
   
    # Removing outliers
    # Not removing outliers from these columns since they are categorical types
    categorical_int_cols = ['admission_type_id', 'discharge_disposition_id', 'admission_source_id']
   
    # Not removing outliers from these columns since values are inside 3 standard deviations
    non_outlier_cols = ['number_outpatient', 'number_emergency', 'number_inpatient', 'time_in_hospital', 'num_procedures']
    final_non_outlier_cols = target_var + categorical_int_cols + non_outlier_cols
    print("\nColumns that are not outliers:\n", final_non_outlier_cols)
    
    # Exclude the non-outlier columns
    numerical_cols = data.select_dtypes(include='number').columns
    numerical_cols = numerical_cols.drop(final_non_outlier_cols)

    print('\n Dropping outliers from: ', numerical_cols.values)
    data = remove_outliers(data, numerical_cols, threshold=1.5)

    # We are not performing normalisation on any of the columns since the range of the values for every feature has insignificant difference
    # feature_normalization(data, numerical_cols)
    print("\nFinal shape of the data:\n", data.shape)

    # Data Visualisation
    data_visualisation(data, categorical_int_cols)
    
    # Evaluate model performance
    results = evaluate_model_performance(data)
    print("\n", results, "\n")

    # Perform oversampling to balance the data
    data_balanced = balance_data_oversampling(data)    
    print('\nThe shape of the balanced data:', data_balanced.shape)
    
    # Evaluate model performance after balancing the data
    results = evaluate_model_performance(data_balanced)
    print("\n", results, "\n")

    # data.to_csv('processed_data.csv', index=False)


if __name__ == '__main__':
    main()
