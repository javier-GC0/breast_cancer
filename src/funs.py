import os
import json
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import cross_validate


def config_kaggle():
    with open("./config/kaggle.json", "r") as f:
        config = json.load(f)

    if config["username"] == "XXXXXXXXX" and config["key"] == "XXXXXXXXX": # If kaggle.json has not been changed
        raise ValueError("Change the config parameters on kaggle.json")
    
    os.environ["KAGGLE_USERNAME"] = config["username"]
    os.environ["KAGGLE_KEY"] = config["key"]

    print("Kaggle configuration done!")


def info_df(df, types=False):
    print(f"DataFrame Information:")
    print(f"- Rows: {df.shape[0]}")
    print(f"- Columns: {df.shape[1]}")
    
    feat = {"Categorical features": len(df.select_dtypes(include=["object", "category"]).columns), 
            "Numerical features": len(df.select_dtypes(include=["number"]).columns), 
            "Datetime features": len(df.select_dtypes(include=["datetime"]).columns),
            "Boolean features": len(df.select_dtypes(include=["bool"]).columns)}
    
    if types:
        for name, num in feat.items():
            if num > 0:
                print(f"- {name}: {num}")

def find_duplicates(df):
    duplicate_rows = df[df.duplicated()]

    if not duplicate_rows.empty:
        return duplicate_rows.index.tolist()
    else:
        print("No duplicate rows found.")


def primary_key(df):
    for column in df.columns:
        unique_values = df[column].nunique()
        rows = len(df)
        
        is_unique = unique_values == rows
        no_nulls = df[column].notnull().all()
        
        if is_unique and no_nulls: # Condition of primary key (unique and non null values)
            print(f"The primary key is '{column}'.")
            return
    
    print("No primary key found in the DataFrame.")


def categ_feats(df):
    categ_feats = list(df.select_dtypes(include=["object", "category"]).columns)
    for var in categ_feats:
        print(f"Values of {var}: {df[var].unique()}")


def check_null_values(df):
    null_counts = df.isnull().sum()  
    total_nulls = null_counts.sum() 
    
    if total_nulls == 0:
        print("The DataFrame has no null values in any column.")
    else:
        print("Null values detected:")
        for col, count in null_counts.items():
            if count > 0:
                print(f"- Column '{col}': {count} null values")


def boxplots(df, nrows, ncols, vars):
    total_plots = nrows * ncols

    if len(vars) > total_plots:
        raise ValueError(f"The grid of {nrows}x{ncols} ({total_plots} subplots) is not enough to plot {len(vars)} variables.")
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 5, nrows * 4))
    axes = axes.flatten()  
    
    for i, col in enumerate(vars):
        sns.boxplot(data=df, x=col, ax=axes[i], hue="diagnosis", palette="Set1")
        axes[i].set_title(f"Boxplot of {col.capitalize()}")
        axes[i].set_xlabel(col.capitalize())

    for j in range(len(vars), len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.show()



def distributions(df, nrows, ncols, vars):
    total_plots = nrows * ncols

    if len(vars) > total_plots:
        raise ValueError(f"The grid of {nrows}x{ncols} ({total_plots} subplots) is not enough to plot {len(vars)} variables.")
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 5, nrows * 4))
    axes = axes.flatten()  
    
    for i, col in enumerate(vars):
        sns.histplot(data=df, x=col, hue="diagnosis", kde=True, bins=30, ax=axes[i], palette="Set1")
        axes[i].set_title(f"Distribution of {col.capitalize()}")
        axes[i].set_xlabel(col.capitalize())
        axes[i].set_ylabel("Frequency")

    for j in range(len(vars), len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.show()


def categ_plots(df, column):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    sns.countplot(data=df, x=column, hue=column, palette="Set1", ax=axes[0])
    axes[0].set_xlabel(column.capitalize())
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"Countplot {column}")

    counts = df[column].value_counts()
    axes[1].pie(counts, labels=counts.index, autopct="%1.1f%%", colors=sns.color_palette("Set1", n_colors=2)[::-1])
    axes[1].set_title(f"Pie Chart {column}")

    plt.tight_layout()
    plt.show()


def correlation_heatmap(df, target=False):
    title = "Correlation Heatmap"
    if target:
        corr_matrix = df.corr()[[target]]
        plt.figure(figsize=(5, 8))
        title += f" ({target})"
    else:
        corr_matrix = df.corr()
        plt.figure(figsize=(20, 16))

    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title(title)
    plt.show()


def plot_confusion_matrices(conf_matrices):
    num_models = len(conf_matrices)
    plt.figure(figsize=(12, 6))
    
    for i, (model_name, cm) in enumerate(conf_matrices.items(), 1):
        plt.subplot(2, num_models // 2 + 1, i)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, 
                    xticklabels=["Benign", "Malignant"], yticklabels=["Benign", "Malignant"])
        plt.title(model_name)
        plt.xlabel("Predicted")
        plt.ylabel("True")
    
    plt.tight_layout()
    plt.show()


def train_and_evaluate(models, X_train, y_train, X_test, y_test):
    results = {}
    conf_matrices = {}
    for name, model in models.items():
        model.fit(X_train, y_train) # Train
        y_pred = model.predict(X_test) # Predict
        results[name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
        }
        conf_matrices[name] = confusion_matrix(y_test, y_pred)
    
    plot_confusion_matrices(conf_matrices)

    return results
    


def evaluate_with_cv(models, X, y, cv_folds=5):
    results = {}
    for name, model in models.items():
        cv_results = cross_validate(model, X, y, cv=cv_folds, scoring=["accuracy", "f1_macro"], return_train_score=False)
        
        results[name] = {
            "mean_accuracy": cv_results["test_accuracy"].mean(),
            "mean_f1": cv_results["test_f1_macro"].mean(),
            "std_accuracy": cv_results["test_accuracy"].std(),
            "std_f1": cv_results["test_f1_macro"].std()
        }

    return results


def plot_model_metrics(models_metrics_dict, metrics):
    num_metrics = len(metrics)
    fig, axes = plt.subplots(1, len(metrics), figsize=(10 * num_metrics, 6))  
    
    if num_metrics == 1:
        axes = [axes]  
    
    for i, metric in enumerate(metrics):
        sorted_models = sorted(models_metrics_dict.items(), key=lambda x: x[1][metric], reverse=False)
        model_names = [model[0] for model in sorted_models]
        metric_values = [model[1][metric] for model in sorted_models]
        bars = axes[i].barh(model_names, metric_values)

        for bar in bars:
            axes[i].text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                         f'{bar.get_width():.4f}', va="center", ha="left", fontsize=10)

        axes[i].set_xlabel(f"{metric.capitalize()} Score")
        axes[i].set_title(f"Model Comparison by {metric.capitalize()}")
        axes[i].set_xlim(0, 1)  # For accuracy/f1 range (0 to 1)
    
    plt.tight_layout()
    plt.show()