import umap
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from umap import UMAP
import plotly.express as px
import numpy as np
from scipy.stats import zscore

def preprocess_data(df, max_unique_values=10):
    # Handle missing values
    numeric_cols = df.select_dtypes(include=['number']).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())  # Fill missing values with median for numeric columns

    # Handle non-numeric values
    for col in df.select_dtypes(include=['object']).columns:
        try:
            df[col] = df[col].astype(float)
        except ValueError:
            pass  # If conversion to float fails, keep the column as it is (assume it's string data)

    # Drop unnecessary columns with a high number of unique categorical values
    unnecessary_columns = []
    for col in df.columns:
        if df[col].dtype == 'object':
            unique_values = df[col].nunique()
            if unique_values > max_unique_values:
                unnecessary_columns.append(col)
    df = df.drop(columns=unnecessary_columns)

    # Encode categorical variables
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if len(df[col].unique()) <= 2:  # Binary categorical variable
            df[col] = pd.factorize(df[col])[0]
        else:  # Categorical variable with more than 2 categories
            df = pd.get_dummies(df, columns=[col], drop_first=True)

    return df

def evaluate_classifier1(model, X, y):
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    return accuracy

def drop_by_importance_threshold(df, feature_importance, threshold):
    # Determine columns to drop based on feature importance
    columns_to_drop = feature_importance[feature_importance['Importance'] < threshold]['Feature'].tolist()

    # Drop columns that are not necessary
    df = df.drop(columns=columns_to_drop)

    return df

def classification_main(df, importance_threshold, accuracy_threshold):

    # Preprocess data
    df_ini = preprocess_data(df)
    df=remove_outliers_zscore(df_ini, threshold=3)
    # Separate features and target for classification
    X_cls = df.iloc[:, :-1]  # Features (all columns except the last one)
    y_cls = df.iloc[:, -1]   # Target (last column) for classification

    # Apply feature scaling for classification
    scaler_cls = StandardScaler()
    X_scaled_cls = scaler_cls.fit_transform(X_cls)

    # Train a Random Forest Classifier to get feature importance
    rf_cls = RandomForestClassifier()
    rf_cls.fit(X_scaled_cls, y_cls)

    # Calculate feature importance for classification
    feature_importance_cls = pd.DataFrame({'Feature': X_cls.columns, 'Importance': rf_cls.feature_importances_})
    feature_importance_cls = feature_importance_cls.sort_values(by='Importance', ascending=False)

    # Drop columns based on importance threshold for classification
    df_cls = drop_by_importance_threshold(df, feature_importance_cls, importance_threshold)

    X_cls = df_cls.iloc[:, :-1]
    y_cls = df_cls.iloc[:, -1]
    scaler1_cls = StandardScaler()
    X_scaled1_cls = scaler1_cls.fit_transform(X_cls)

    # Split data into training and testing sets for classification
    X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X_scaled1_cls, y_cls, test_size=0.2, random_state=42)
    prev_accuracy = 0
    while True:
        # Train and evaluate Random Forest Classifier
        rf_cls.fit(X_train_cls, y_train_cls)
        rf_accuracy = evaluate_classifier1(rf_cls, X_test_cls, y_test_cls)

        # Check if accuracy change is less than the threshold
        if abs(prev_accuracy - rf_accuracy) > accuracy_threshold:
            break

        # Drop the column with the least importance
        least_important_feature = feature_importance_cls.iloc[-1]['Feature']
        df_cls = df_cls.drop(columns=[least_important_feature])
        X_cls = df_cls.iloc[:, :-1]
        y_cls = df_cls.iloc[:, -1]
        scaler2_cls = StandardScaler()
        X_scaled2_cls = scaler1_cls.fit_transform(X_cls)
        X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X_scaled2_cls, y_cls, test_size=0.2, random_state=42)

        # Update feature importance and previous accuracy
        rf_cls.fit(X_train_cls, y_train_cls)
        feature_importance_cls = feature_importance_cls[feature_importance_cls['Feature'] != least_important_feature]
        prev_accuracy = rf_accuracy
    create_and_save_graphs(df_ini, df_cls, output_dir='graphs')
    name,accu=find_best_model(df_cls)
    return name,accu

def remove_outliers_zscore(df, threshold=3):
    # Calculate z-scores for each column in the DataFrame
    z_scores = np.abs((df - df.mean()) / df.std())

    # Create a mask to identify rows where more than 20% of columns have z-score greater than threshold
    outlier_mask = (z_scores > threshold).sum(axis=1) / len(df.columns) > 0.2

    # Remove rows that meet the outlier criteria
    df_cleaned = df[~outlier_mask]

    return df_cleaned

def evaluate_classifier(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def select_models(df):
    # Separate features and target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Dataset Properties
    num_samples = len(df)
    num_features = X.shape[1]
    high_dimensional = num_features > 100
    correlation_matrix = X.corr().abs()
    sparsity = (X == 0).mean().mean()
    class_distribution = y.value_counts(normalize=True)
    imbalanced = class_distribution.min() < 0.1

    # Determine number of categorical and continuous columns
    num_categorical = len(X.select_dtypes(include=['object', 'category']).columns)
    num_continuous = len(X.select_dtypes(include=['number']).columns)

    # Initialize the full model list
    initial_models = [
        LogisticRegression(),
        GaussianNB(),
        KNeighborsClassifier(),
        RandomForestClassifier(),
        SVC(probability=True),
        XGBClassifier()
    ]

    # Step-by-step rejection criteria
    surviving_models = initial_models.copy()
    removed_by_criteria = {model: 0 for model in initial_models}

    # Step 1: Size of the Dataset
    if num_samples >= 1000:
        for model in initial_models:
            if isinstance(model, (LogisticRegression, GaussianNB, KNeighborsClassifier)) and model in surviving_models:
                surviving_models.remove(model)
                removed_by_criteria[model] += 1

    # Step 2: Dimensionality
    if high_dimensional:
        for model in initial_models:
            if isinstance(model, (LogisticRegression, GaussianNB, KNeighborsClassifier)) and model in surviving_models:
                surviving_models.remove(model)
                removed_by_criteria[model] += 1

    # Step 3: Linearity
    if correlation_matrix.mean().mean() <= 0.5:  # Example threshold for correlation
        for model in initial_models:
            if isinstance(model, LogisticRegression) and model in surviving_models:
                surviving_models.remove(model)
                removed_by_criteria[model] += 1

    # Step 4: Sparsity
    if sparsity > 0.7:  # Example threshold for sparsity
        for model in initial_models:
            if isinstance(model, (RandomForestClassifier, XGBClassifier)) and model in surviving_models:
                surviving_models.remove(model)
                removed_by_criteria[model] += 1

    # Step 5: Imbalance
    if imbalanced:
        for model in initial_models:
            if isinstance(model, (LogisticRegression, GaussianNB, KNeighborsClassifier)) and model in surviving_models:
                surviving_models.remove(model)
                removed_by_criteria[model] += 1

    # Step 6: Number of Categorical Columns
    if num_categorical > 10:  # Example threshold for a high number of categorical features
        for model in initial_models:
            if isinstance(model, (LogisticRegression, SVC)) and model in surviving_models:
                surviving_models.remove(model)
                removed_by_criteria[model] += 1

    # Step 7: Number of Continuous Columns
    if num_continuous > 10:  # Example threshold for a high number of continuous features
        for model in initial_models:
            if isinstance(model, KNeighborsClassifier) and model in surviving_models:
                surviving_models.remove(model)
                removed_by_criteria[model] += 1

    # Check if any models are left
    if not surviving_models:
        # Select models that survived the least number of rejection rounds
        max_survival = max(removed_by_criteria.values())
        surviving_models = [model for model, count in removed_by_criteria.items() if count == max_survival]

    return surviving_models

def find_best_model(df):
    # Separate features and target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Select models
    models = select_models(df)

    # Evaluate models
    model_accuracies = {}
    for model in models:
        accuracy = evaluate_classifier(model, X_train_scaled, X_test_scaled, y_train, y_test)
        model_accuracies[type(model).__name__] = accuracy

    # Find the best performing model
    best_model_name = max(model_accuracies, key=model_accuracies.get)
    best_accuracy = model_accuracies[best_model_name]

    # Print all accuracies
    for model_name, accuracy in model_accuracies.items():
        print(f"{model_name}: {accuracy:.4f}")

    print(f"\nBest Performing Model: {best_model_name} with Accuracy: {best_accuracy:.4f}")
    return best_model_name, best_accuracy



def create_and_save_graphs(df_initial, df_final, output_dir='graphs'):
    # Create the graphs directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Identify outliers using Z-score
    z_scores = np.abs(zscore(df_initial.iloc[:, :-1]))
    outliers = (z_scores > 3).any(axis=1)
    
    # Limit the number of outliers to 20%
    max_outliers = int(len(df_initial) * 0.2)
    if outliers.sum() > max_outliers:
        outlier_scores = z_scores[outliers].sum(axis=1)
        top_outliers = np.argsort(outlier_scores)[-max_outliers:]
        outliers[:] = False
        outliers[top_outliers] = True

    df_no_outliers = df_initial[~outliers]

    # Class distribution pie charts before and after preprocessing
    class_counts_initial = df_initial.iloc[:, -1].value_counts()
    class_counts_final = df_final.iloc[:, -1].value_counts()

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    class_counts_initial.plot.pie(autopct='%1.1f%%', title='Class Distribution Before Preprocessing')
    plt.subplot(1, 3, 2)
    class_counts_final.plot.pie(autopct='%1.1f%%', title='Class Distribution After Preprocessing')
    plt.savefig(os.path.join(output_dir, 'class_distribution_piecharts.png'))
    plt.close()

    # Pie charts for each class showing the percentage of outliers
    for cls in class_counts_initial.index:
        class_mask = (df_initial.iloc[:, -1] == cls)
        outlier_mask = outliers[class_mask]
        outlier_percentage = outlier_mask.mean()
        plt.figure()
        plt.pie([outlier_percentage, 1 - outlier_percentage], labels=['Outliers', 'Non-outliers'], autopct='%1.1f%%')
        plt.title(f'Outlier Percentage for Class {cls}')
        plt.savefig(os.path.join(output_dir, f'outlier_percentage_class_{cls}.png'))
        plt.close()

    # UMAP scatter plot for each class with different colors for outliers
    if df_final.shape[1] > 2:
        umap = UMAP(n_components=2)
        embedding = umap.fit_transform(df_final.iloc[:, :-1])
        df_umap = pd.DataFrame(embedding, columns=['UMAP1', 'UMAP2'])
        df_umap['Class'] = df_final.iloc[:, -1]

        plt.figure()
        sns.scatterplot(data=df_umap, x='UMAP1', y='UMAP2', hue='Class', palette='tab10')
        plt.title('UMAP Scatter Plot after')
        plt.savefig(os.path.join(output_dir, 'umap_scatter_plot_after.png'))
        plt.close()

    if df_initial.shape[1] > 2:
        umap = UMAP(n_components=2)
        embedding = umap.fit_transform(df_initial.iloc[:, :-1])
        df_umap = pd.DataFrame(embedding, columns=['UMAP1', 'UMAP2'])
        df_umap['Class'] = df_initial.iloc[:, -1]

        plt.figure()
        sns.scatterplot(data=df_umap, x='UMAP1', y='UMAP2', hue='Class', palette='tab10')
        plt.title('UMAP Scatter Plot before')
        plt.savefig(os.path.join(output_dir, 'umap_scatter_plot_before.png'))
        plt.close()

    
    # Pair plots before and after preprocessing
    if df_initial.shape[1] > 8:
        umap = UMAP(n_components=8)
        df_reduced_initial = pd.DataFrame(umap.fit_transform(df_initial.iloc[:, :-1]), columns=[f'Feature_{i}' for i in range(8)])
        df_reduced_initial['Class'] = df_initial.iloc[:, -1]
        df_reduced_final = pd.DataFrame(umap.fit_transform(df_final.iloc[:, :-1]), columns=[f'Feature_{i}' for i in range(8)])
        df_reduced_final['Class'] = df_final.iloc[:, -1]
        sns.pairplot(df_reduced_initial, hue='Class')
        plt.savefig(os.path.join(output_dir, 'pairplot_before.png'))
        plt.close()
        sns.pairplot(df_reduced_final, hue='Class')
        plt.savefig(os.path.join(output_dir, 'pairplot_after.png'))
        plt.close()
    else:
        sns.pairplot(df_initial, hue=df_initial.columns[-1])
        plt.savefig(os.path.join(output_dir, 'pairplot_before.png'))
        plt.close()
        sns.pairplot(df_final, hue=df_final.columns[-1])
        plt.savefig(os.path.join(output_dir, 'pairplot_after.png'))
        plt.close()

    # Movable 3D plots before and after preprocessing
    if df_initial.shape[1] >= 3:
        if df_initial.shape[1] > 3:
            umap = UMAP(n_components=3)
            df_reduced_initial = pd.DataFrame(umap.fit_transform(df_initial.iloc[:, :-1]), columns=['UMAP1', 'UMAP2', 'UMAP3'])
            df_reduced_initial['Class'] = df_initial.iloc[:, -1]
            df_reduced_final = pd.DataFrame(umap.fit_transform(df_final.iloc[:, :-1]), columns=['UMAP1', 'UMAP2', 'UMAP3'])
            df_reduced_final['Class'] = df_final.iloc[:, -1]
        else:
            df_reduced_initial = df_initial.copy()
            df_reduced_final = df_final.copy()

        fig = px.scatter_3d(df_reduced_initial, x='UMAP1', y='UMAP2', z='UMAP3', color='Class')
        fig.write_html(os.path.join(output_dir, '3d_plot_before.html'))
        fig = px.scatter_3d(df_reduced_final, x='UMAP1', y='UMAP2', z='UMAP3', color='Class')
        fig.write_html(os.path.join(output_dir, '3d_plot_after.html'))

if __name__ == "__main__":
    df = pd.read_csv('data.csv')
    importance_threshold = 0.01
    accuracy_threshold = 0.001
    name,accu=classification_main(df, importance_threshold, accuracy_threshold)

