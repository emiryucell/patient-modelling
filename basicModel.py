import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.artist import setp
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve, auc
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.utils import resample


def process_data(data):
    print("Shape of the data:\n", data.shape)
    # Dropping specific column from data (encounter_id)
    data.drop('encounter_id', axis=1, inplace=True)

    # Checking the missing values
    missing_values_before = data.isnull().sum()
    print("\nSummary of missing values before replace:\n", missing_values_before)

    # Replacing "?" to NaN to identify missing values
    data.replace('?', np.nan, inplace=True)

    # Checking the missing values after replacing
    missing_values_after = data.isnull().sum()
    print("\nSummary of missing values after replace:\n", missing_values_after)
    # Mapping readmitted values to "0" and "1"
    data['readmitted'] = data['readmitted'].map({'<30': 1, '>30': 0, 'NO': 0})

    print("\nData types of each column:\n", data.dtypes)

    # Checking the columns' missing value percentage
    missing_percent = data.isnull().mean() * 100

    # Dropping the columns which have more than 90% percentage missing value
    columns_to_drop = missing_percent[missing_percent > 90].index
    data.drop(columns=columns_to_drop, inplace=True)

    # Dropping cols such as payer_code and medical_specialty since they don't play a major role in predicting the target variable
    columns_to_delete = [
        'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'acetohexamide',
        'tolbutamide', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide', 'examide',
        'citoglipton', 'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone',
        'metformin-rosiglitazone', 'metformin-pioglitazone', 'payer_code', 'medical_specialty', 'patient_nbr']

    data.drop(columns=columns_to_delete, inplace=True)

    # Dropping NULL values
    data.dropna(axis=0, how='any', inplace=True)

    print("\nSummary statistics of numerical columns:\n", data.select_dtypes(include='number').describe())

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


def data_visualisation(data, categorical_int_cols):
    # Specifying order and labels for readmitted graph
    readmission_order = [0, 1]
    readmission_legend_labels = ['Non-Readmitted', 'Readmitted']
    # Distribution of the target variable
    ax = sns.barplot(x='readmitted', y='readmitted', estimator=lambda x: len(x) / len(data) * 100,
                     data=data, hue='readmitted', order=readmission_order, hue_order=readmission_order)
    # Giving percentage to the bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.f%%')

    # Specifying name of the graph and axis
    ax.set_ylabel('Percentage (%)')
    ax.set_xlabel('Readmission')
    ax.set_title('Distribution of Readmission')
    # Showing the graph
    plt.show()

    # distribution of a 'readmitted' column based on different ages
    value_counts = data.sort_values('age').groupby('age')['readmitted'].value_counts().unstack()
    fig, ax3 = plt.subplots(figsize=(12, 8))

    # Creating a graph for number of readmitted cases against age
    bars = ax3.bar(value_counts.index, value_counts[1])
    ax3.bar_label(bars)

    plt.xlabel("Age Groups")
    plt.ylabel("Readmitted Cases")
    plt.title("Number of Readmitted Cases Against Age")
    plt.show()

    # Creating a graph for number of target variable against the number of medications
    plt.figure(figsize=(16, 8))
    ax2 = sns.countplot(x="num_medications", data=data, hue="readmitted", hue_order=readmission_order, legend=False)
    for container in ax2.containers:
        ax2.bar_label(container)
    ax2.legend(labels=readmission_legend_labels, loc="upper right")
    plt.xlabel("Number of Medications")
    plt.ylabel("Number of People Readmitted / Non-Readmitted")
    plt.title("Number of Medications Vs Readmitted Cases")
    plt.show()

    # Creating a new DataFrame with only the specified numerical columns
    num_df = data.select_dtypes(include='number')
    # Removing some numerical cols which are actually categorical
    num_df.drop(categorical_int_cols, axis=1, inplace=True)

    # Plotting correlation matrix
    plot_correlation_matrix(num_df)
    # Plotting scatter matrix
    plot_scatter_matrix(num_df)

    # Plotting Additional Graphs

    # Plotting a graph of Readmission Based on the gender
    plt.figure(figsize=(12, 8))
    gender_plot = sns.countplot(x='gender', data=data, hue='readmitted', hue_order=readmission_order)
    gender_plot.legend(labels=readmission_legend_labels, loc="upper right")
    gender_plot.axes.set_title('Readmission based on Gender')
    plt.xlabel("Genders")
    plt.ylabel("Number of People Readmitted / Non-Readmitted")
    plt.show()
    # Plotting Race and Gender Graph
    legend_for_race = ["Caucasian", "AfricanAmerican", "Asian", "Hispanic", "Other"]
    legend_for_gender = ["Male", "Female"]
    fig, ax = plt.subplots(figsize=(12, 8), ncols=2, nrows=1)
    race = sns.countplot(x="race", data=data, ax=ax[0], hue="race", order=legend_for_race, hue_order=legend_for_race)
    plt.subplots_adjust(bottom=0.25, top=0.9, left=0.15, right=0.85)
    race.legend(labels=legend_for_race, loc="upper right")
    gender = sns.countplot(x="gender", data=data, ax=ax[1], hue="gender")
    gender.legend(labels=legend_for_gender, loc="upper right")
    ax[0].tick_params(axis="x", rotation=45)
    ax[1].tick_params(axis="x", rotation=45)
    gender.set_xlabel("Gender", fontsize=10, weight='bold')
    race.set_xlabel("Race", fontsize=10, weight='bold')
    ax[1].xaxis.set_label_coords(0.5, -0.2)
    plt.subplots_adjust(wspace=1)

    plt.show()


def plot_scatter_matrix(num_df):
    # Visualisation the correlation matrix
    axes = pd.plotting.scatter_matrix(num_df, alpha=0.2, figsize=(18, 10), diagonal='kde')

    # Adjustments for correlation matrix
    for ax in axes.flatten():
        ax.yaxis.label.set_rotation(0)
        ax.yaxis.label.set_ha('right')
        setp(ax.get_xticklabels(), rotation=0)

    plt.suptitle('Scatter Matrix for Selected Numerical Features')
    plt.show()


def plot_correlation_matrix(num_df):
    # Calculate the correlation matrix
    corr_df = num_df.drop(columns=['readmitted'])
    corr_matrix = corr_df.corr()

    # Adjustments for fitting
    fig, ax = plt.subplots(figsize=(15, 8))
    plt.subplots_adjust(bottom=0.25, top=0.9, left=0.15, right=1)

    # Visualisation the correlation matrix
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
    plt.xticks(rotation=45, ha='right')

    # Add labels for axes
    plt.xlabel("Numerical Features", fontsize=10, weight='bold', labelpad=10)
    plt.ylabel("Numerical Features", fontsize=10, weight='bold', labelpad=10)

    plt.title("Correlation Matrix of Numerical Features", pad=10)

    plt.show()

    # Threshold for high correlation (can be adjusted)
    threshold = 0.25
    # Declaration of highly correlated pairs from matrix
    highly_correlated_pairs = corr_matrix.unstack().sort_values(kind="quicksort", ascending=False)
    # Removing self-correlation and correlations below than the threshold
    highly_correlated_pairs = highly_correlated_pairs[
        (abs(highly_correlated_pairs) > threshold) & (highly_correlated_pairs < 1)]

    # Listing highly correlated pairs
    print("\nHighly Correlated Pairs:\n")
    for (idx1, idx2), value in highly_correlated_pairs.items():
        print(f"{idx1} <-> {idx2}: {value}")


def evaluate_model_performance(data):
    # Dropping diag_1, diag_2 and diag_3 since they have too many distinct values
    data = data.drop(['diag_1', 'diag_2', 'diag_3'], axis=1)
    # Convert categorical variables to dummy variables
    cat_cols = ['race', 'gender', 'age', 'admission_type_id', 'discharge_disposition_id', 'admission_source_id', 'A1Cresult', 'metformin', 'glipizide', 'glyburide', 'pioglitazone', 'rosiglitazone',
                'insulin', 'change', 'diabetesMed']
    data = pd.get_dummies(data, columns=cat_cols, drop_first=True)

    print('The shape of the data after converting categorical variables to dummy variables:', data.shape)

    # Splitting dataset into features (X) and target (y)
    X = data.drop('readmitted', axis=1)
    y = data['readmitted']

    # Splitting the dataset into training and testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Feature selection with RFE
    model = LogisticRegression(max_iter=1000, solver='liblinear')
    rfe = RFE(estimator=model, n_features_to_select=5)
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
 
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)   
    auc_score = auc(recall, precision)

    # Plot the mean ROC curve
    ax.plot(fpr, tpr, color='blue',
            label=f'Mean ROC (AUC = {auc_score:.2f})', lw=2, alpha=0.8)
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='red', label='Chance', alpha=0.8)
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title="ROC Curve")
    ax.legend(loc="lower right")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.show()

    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(20)
    for item in (ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(15)
    plt.show()

    # Calculate precision and recall. Also the Area Under the Curve (AUC) for precision-recall curve
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker='.', label=f'Precision-Recall Curve (AUC = {auc_score:.2f})')
    # Adding baseline -- no skill line (precision = no. of positives / total no. of samples)
    plt.plot([0, 1], [len(y_test[y_test==1]) / len(y_test), len(y_test[y_test==1]) / len(y_test)], linestyle='--', label='No Skill')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.show()

    # Output the results
    return {
        "Selected Features": selected_features.tolist(),
        "Cross-Validation Score": cv_scores.mean(),
        "Overall Metrics": overall_metrics
    }


def balance_data_oversampling(data):
    # re-distributing target variable
    df_majority = data[data.readmitted == 0]
    df_minority = data[data.readmitted == 1]

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
    # Reading Dataset
    data = pd.read_csv('diabetic_data.csv')

    # Data Processing
    data = process_data(data)
    print('The shape of the data after processing:', data.shape)
    # Deciding target variable
    target_var = ['readmitted']
    print('The numerical cols are:', data.select_dtypes(include='number').columns.values)

    # Not removing outliers from these columns since they are categorical types
    categorical_int_cols = ['admission_type_id', 'discharge_disposition_id', 'admission_source_id']

    # Not removing outliers from these columns since values are inside 3 standard deviations
    non_outlier_cols = ['number_outpatient', 'number_emergency', 'number_inpatient', 'time_in_hospital',
                        'num_procedures']

    # Deciding columns which won't processed for outlier removal
    final_non_outlier_cols = target_var + categorical_int_cols + non_outlier_cols

    print("\nColumns that are not outliers:\n", final_non_outlier_cols)

    # Specifying numerical columns
    numerical_cols = data.select_dtypes(include='number').columns

    # Boxplot for non-outlier columns
    plt.figure(figsize=(20, 10))
    for i, col in enumerate(final_non_outlier_cols, 1):
        plt.subplot(3, 3, i)
        data.boxplot(col)
        plt.title(col)
    plt.tight_layout()
    plt.show()

    # Exclude the non-outlier columns
    numerical_cols = numerical_cols.drop(final_non_outlier_cols)

    # Removing outliers
    print('\n Dropping outliers from: ', numerical_cols.values)
    data = remove_outliers(data, numerical_cols, threshold=1.5)

    # We are not performing normalisation on any of the columns since the range of the values for every feature has insignificant difference

    print("\nFinal shape of the data:\n", data.shape)
    print(data.columns.values)

    # Data Visualisation
    data_visualisation(data, categorical_int_cols)

    # Evaluate model performance
    results = evaluate_model_performance(data)
    print(results)

    # Perform oversampling to balance the data
    data_balanced = balance_data_oversampling(data)
    print('The shape of the balanced data:', data_balanced.shape)

    # Evaluate model performance after balancing the data
    results = evaluate_model_performance(data_balanced)
    print(results)

    # data.to_csv('processed_data.csv', index=False)

    # Further analysis of improved model is performed in the next file. Please refer to 'improvedModel.py'


if __name__ == '__main__':
    main()
