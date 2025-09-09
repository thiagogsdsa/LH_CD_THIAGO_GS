######################################################
# ===================== IMPORTS =====================
# Imports padrão
import os
import sys
from pathlib import Path
import pickle
import json

# Ciência de dados e matemática
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, chi2_contingency, boxcox

# Pré-processamento e Machine Learning
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LassoCV, RidgeCV

# Visualização
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
from scipy.stats import gaussian_kde
import numpy as np
from IPython.display import display, Image
# Utilitários
from unidecode import unidecode


######################################################
# ===================== Variables =====================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

######################################################
# ===================== UTILS =====================

def p_print(d):
    print(json.dumps(d, indent=4))

######################################################
# ===================== DATAFRAME =====================


def detect_outliers(df_numeric, factor=1.5):
    """
    Detects outliers in numeric columns of a DataFrame using the IQR method.

    Args:
        df_numeric (pd.DataFrame): Numeric columns of a DataFrame.
        factor (float): Multiplicative factor for IQR.

    Returns:
        pd.DataFrame: Boolean DataFrame of outliers
        pd.Series: Counts of outliers per column
        pd.Series: Percent of outliers per column
    """
    Q1 = df_numeric.quantile(0.25)
    Q3 = df_numeric.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - factor * IQR
    upper = Q3 + factor * IQR

    outlier_mask = (df_numeric < lower) | (df_numeric > upper)
    outliers_count = outlier_mask.sum()
    outliers_percent = (outliers_count / len(df_numeric) * 100).round(2)

    return outlier_mask, outliers_count, outliers_percent


def summarize_dataframe(
    dataframe, head=True, head_size=5, sample_size=5, outlier_factor=1.5
):
    """
    Provides a detailed overview of a pandas DataFrame including outliers and categorical stats.
    """

    # ======= DESCRIPTIVE INFORMATION =======
    df_info = pd.DataFrame({
        "Num NAs": dataframe.isna().sum(),
        "Percent NAs": (dataframe.isna().mean() * 100).round(2),
        "Num unique": dataframe.nunique(),
        "Data Type": dataframe.dtypes,
    })

    # ======= NUMERIC STATISTICS =======
    df_numeric = dataframe.select_dtypes(include="number")
    df_stats = None
    outlier_mask = None
    if not df_numeric.empty:
        outlier_mask, outliers_count, outliers_percent = detect_outliers(
            df_numeric, factor=outlier_factor
        )

        df_stats = pd.DataFrame({
            "attribute": df_numeric.columns,
            "mean": df_numeric.mean().values,
            "median": df_numeric.median().values,
            "std": df_numeric.std().values,
            "iqr": (df_numeric.quantile(0.75) - df_numeric.quantile(0.25)).values,
            "min": df_numeric.min().values,
            "max": df_numeric.max().values,
            "range": (df_numeric.max() - df_numeric.min()).values,
            "skew": df_numeric.skew().values,
            "kurtosis": df_numeric.kurtosis().values,
            "outliers_count": outliers_count.values,
            "outliers_percent": outliers_percent.values,
        })

    # ======= CATEGORICAL STATISTICS =======
    df_categorical = dataframe.select_dtypes(exclude="number")
    cat_stats = None
    if not df_categorical.empty:
        cat_stats = pd.DataFrame({
            "attribute": df_categorical.columns,
            "top": df_categorical.mode().iloc[0].values,
            "top_freq": [df_categorical[col].value_counts().iloc[0] for col in df_categorical.columns],
            "top_percent": [
                (df_categorical[col].value_counts(normalize=True).iloc[0]*100).round(2)
                for col in df_categorical.columns
            ]
        })

    # ======= SAMPLE/HEAD =======
    df_sample = dataframe.head(head_size) if head else dataframe.sample(sample_size)

    return {
        "info": df_info,
        "numeric_stats": df_stats,
        "categorical_stats": cat_stats,
        "head_sample": df_sample,
        "outliers": outlier_mask,
    }

######################################################
# ===================== Plots =====================

# def plot_distribution(df, col_name, nbins=30):
#     """
#     Plots a histogram with KDE for a numeric column.
#     """
#     if col_name not in df.columns:
#         raise ValueError(f"Column '{col_name}' does not exist in DataFrame.")

#     series = df[col_name].dropna().astype(float)
    
#     # Histogram
#     hist = go.Histogram(
#         x=series,
#         nbinsx=nbins,
#         name="Histogram",
#         opacity=0.7
#     )
    
#     # KDE
#     kde = gaussian_kde(series)
#     x_range = np.linspace(series.min(), series.max(), 200)
    
#     bin_width = (series.max() - series.min()) / nbins
#     kde_scaled = kde(x_range) * len(series) * bin_width  # escala coerente com o histograma
    
#     kde_trace = go.Scatter(
#         x=x_range,
#         y=kde_scaled,
#         mode="lines",
#         name="KDE",
#         line=dict(color="red")
#     )
    
#     fig = go.Figure([hist, kde_trace])
#     fig.update_layout(
#         title=f"Distribution of '{col_name}'",
#         width=800, height=400,
#         xaxis_title=col_name,
#         yaxis_title="Count"
#     )
#     fig.show()

# def plot_categorical_association(df, cat_cols=None, alpha=0.05, top_k=5):
#     """
#     Plots a heatmap of chi-squared p-values among categorical variables.
#     Only uses top_k categories of each variable to improve readability.
#     Cells are 1 if p < alpha (significant association), 0 otherwise.
    
#     Parameters:
#     - df: pandas DataFrame
#     - cat_cols: list of categorical column names (default: all object columns)
#     - alpha: significance level
#     - top_k: maximum number of categories per variable
#     """
#     if cat_cols is None:
#         cat_cols = df.select_dtypes(exclude=['number']).columns.tolist()
    
#     n = len(cat_cols)
#     significance_matrix = np.zeros((n, n))
    
#     for i, col1 in enumerate(cat_cols):
#         for j, col2 in enumerate(cat_cols):
#             if i == j:
#                 significance_matrix[i, j] = 1
#             else:
#                 # pegar top_k categorias de cada coluna
#                 top_categories_col1 = df[col1].value_counts().nlargest(top_k).index
#                 top_categories_col2 = df[col2].value_counts().nlargest(top_k).index
                
#                 # filtrar df para essas categorias
#                 df_filtered = df[df[col1].isin(top_categories_col1) & df[col2].isin(top_categories_col2)]
                
#                 # criar crosstab
#                 contingency = pd.crosstab(df_filtered[col1], df_filtered[col2])
                
#                 if contingency.shape[0] > 1 and contingency.shape[1] > 1:
#                     chi2, p, _, _ = chi2_contingency(contingency)
#                     significance_matrix[i, j] = int(p < alpha)
#                 else:
#                     significance_matrix[i, j] = 0  # Not enough variation
    
#     fig = go.Figure(
#         go.Heatmap(
#             z=significance_matrix,
#             x=cat_cols,
#             y=cat_cols,
#             colorscale=[[0, 'white'], [1, 'red']],
#             zmin=0, zmax=1,
#             colorbar=dict(title=f"Significant at alpha={alpha}")
#         )
#     )
#     fig.update_layout(title=f"Categorical Association Heatmap (Top {top_k} categories)", width=800, height=600)
#     fig.show()

# def plot_boxplot_stats(df, col_name):

#     if col_name not in df.columns:
#         raise ValueError(f"Column '{col_name}' does not exist in the DataFrame.")

#     series = df[col_name].dropna().astype(float)

#     # Quantiles
#     q = series.quantile([0.25, 0.75])
#     iqr = q[0.75] - q[0.25]
#     lower_bound = q[0.25] - 1.5 * iqr
#     upper_bound = q[0.75] + 1.5 * iqr

#     # Outliers
#     outliers = series[(series < lower_bound) | (series > upper_bound)]
#     num_outliers = len(outliers)
#     perc_outliers = (num_outliers / len(series)) * 100

#     # Stats
#     print(f"Column: {col_name}")
#     print(f"Number of outliers: {num_outliers:,}")
#     print(f"Percentage of outliers: {perc_outliers:.2f}%")

#     # Determine axis range to better show box body
#     y_min = max(series.min(), q[0.25] - 1.5 * iqr)
#     y_max = min(series.max(), q[0.75] + 1.5 * iqr)

#     # Boxplot
#     fig = go.Figure()
#     fig.add_trace(
#         go.Box(
#             y=series,
#             name=col_name,
#             boxpoints="suspectedoutliers",  # show only points outside 1.5*IQR
#             marker_color="orange",
#             boxmean="sd",
#             line=dict(width=2),
#         )
#     )

#     fig.update_layout(
#         title=f"Boxplot of '{col_name}'",
#         width=800,
#         height=400,
#         yaxis=dict(range=[y_min, y_max]),  # zoom to show the box clearly
#     )

#     fig.show()

# def plot_top_frequencies(df, col_name, top_k=10):
#     """
#     Plots a bar chart of the top_k most frequent values in a column.
#     """
#     if col_name not in df.columns:
#         raise ValueError(f"Column '{col_name}' does not exist in DataFrame.")
    
#     title=f"Top {top_k} Frequencies of '{col_name}'"
#     freq = df[col_name].value_counts().nlargest(top_k)

#     # Count frequencies
#     if not top_k:
#         freq = df[col_name].value_counts()
#         title=f" Distribution of '{col_name}'"
    
#     # Bar chart
#     bar = go.Bar(
#         x=freq.index.astype(str),
#         y=freq.values,
#         name="Frequency",
#         marker_color='blue'
#     )
    
#     fig = go.Figure([bar])
#     fig.update_layout(
#         title= title,
#         xaxis_title=col_name,
#         yaxis_title="Count",
#         width=800, height=400
#     )
#     fig.show()

# def plot_numeric_correlation(df, numeric_cols=None):
#     """
#     Plots a correlation heatmap for numeric columns.
#     """
#     if numeric_cols is None:
#         numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    
#     corr = df[numeric_cols].corr()
    
#     fig = go.Figure(
#         go.Heatmap(
#             z=corr.values,
#             x=corr.columns,
#             y=corr.columns,
#             colorscale='RdBu',
#             zmin=-1, zmax=1,
#             colorbar_title="Correlation"
#         )
#     )
#     fig.update_layout(title="Correlation Heatmap", width=800, height=600)
#     fig.show()



def plot_distribution(df, col_name, nbins=30):
    """
    Plots a histogram with KDE for a numeric column.
    Displays a static image for GitHub while keeping interactive plot locally.
    """
    if col_name not in df.columns:
        raise ValueError(f"Column '{col_name}' does not exist in DataFrame.")

    series = df[col_name].dropna().astype(float)
    
    # Histogram
    hist = go.Histogram(
        x=series,
        nbinsx=nbins,
        name="Histogram",
        opacity=0.7
    )
    
    # KDE
    kde = gaussian_kde(series)
    x_range = np.linspace(series.min(), series.max(), 200)
    
    bin_width = (series.max() - series.min()) / nbins
    kde_scaled = kde(x_range) * len(series) * bin_width
    
    kde_trace = go.Scatter(
        x=x_range,
        y=kde_scaled,
        mode="lines",
        name="KDE",
        line=dict(color="red")
    )
    
    fig = go.Figure([hist, kde_trace])
    fig.update_layout(
        title=f"Distribution of '{col_name}'",
        width=800, height=400,
        xaxis_title=col_name,
        yaxis_title="Count"
    )
    
    # # Render interactive plot locally
    # fig.show()
    
    # Render static PNG inline para GitHub
    img_bytes = fig.to_image(format="png")
    display(Image(img_bytes))



def plot_boxplot_stats(df, col_name):
    """
    Plots a boxplot with outlier stats for a numeric column using Matplotlib.
    """
    if col_name not in df.columns:
        raise ValueError(f"Column '{col_name}' does not exist in the DataFrame.")

    series = df[col_name].dropna().astype(float)

    # Quantiles
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Outliers
    outliers = series[(series < lower_bound) | (series > upper_bound)]
    num_outliers = len(outliers)
    perc_outliers = (num_outliers / len(series)) * 100

    print(f"Column: {col_name}")
    print(f"Number of outliers: {num_outliers:,}")
    print(f"Percentage of outliers: {perc_outliers:.2f}%")

    # Boxplot
    plt.figure(figsize=(8,4))
    plt.boxplot(series, vert=True, patch_artist=True,
                showfliers=True, 
                boxprops=dict(facecolor='orange', color='black'),
                medianprops=dict(color='red'),
                flierprops=dict(markerfacecolor='red', marker='o', markersize=5, alpha=0.6))
    plt.title(f"Boxplot of '{col_name}'")
    plt.ylabel(col_name)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.show()



def plot_top_frequencies(df, col_name, top_k=10):
    """
    Plots a bar chart of the top_k most frequent values in a column using Matplotlib.
    """
    if col_name not in df.columns:
        raise ValueError(f"Column '{col_name}' does not exist in DataFrame.")
    
    # Frequências
    if top_k:
        freq = df[col_name].value_counts().nlargest(top_k)
        title = f"Top {top_k} Frequencies of '{col_name}'"
    else:
        freq = df[col_name].value_counts()
        title = f"Distribution of '{col_name}'"
    
    # Gráfico de barras
    plt.figure(figsize=(8,4))
    plt.bar(freq.index.astype(str), freq.values, color='blue', alpha=0.7)
    plt.title(title)
    plt.xlabel(col_name)
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_numeric_correlation(df, numeric_cols=None):
    """
    Plots a correlation heatmap for numeric columns using Matplotlib.
    Works on GitHub and locally.
    """
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    corr = df[numeric_cols].corr()
    n = len(corr.columns)
    
    plt.figure(figsize=(10,8))
    im = plt.imshow(corr, cmap='RdBu', vmin=-1, vmax=1)
    
    # Colorbar
    plt.colorbar(im, fraction=0.046, pad=0.04, label='Correlation')
    
    # Labels
    plt.xticks(ticks=np.arange(n), labels=corr.columns, rotation=45, ha='right')
    plt.yticks(ticks=np.arange(n), labels=corr.columns)
    
    # Mostrar valores dentro das células
    for i in range(n):
        for j in range(n):
            plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha='center', va='center', color='black', fontsize=8)
    
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()

def plot_categorical_association(df, cat_cols=None, alpha=0.05, top_k=5):
    """
    Plots a heatmap of chi-squared significance among categorical variables using Matplotlib.
    Cells are 1 if p < alpha (significant association), 0 otherwise.
    Only uses top_k categories per variable.
    """
    if cat_cols is None:
        cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    
    n = len(cat_cols)
    significance_matrix = np.zeros((n, n))
    
    for i, col1 in enumerate(cat_cols):
        for j, col2 in enumerate(cat_cols):
            if i == j:
                significance_matrix[i, j] = 1
            else:
                top_categories_col1 = df[col1].value_counts().nlargest(top_k).index
                top_categories_col2 = df[col2].value_counts().nlargest(top_k).index
                
                df_filtered = df[df[col1].isin(top_categories_col1) & df[col2].isin(top_categories_col2)]
                contingency = pd.crosstab(df_filtered[col1], df_filtered[col2])
                
                if contingency.shape[0] > 1 and contingency.shape[1] > 1:
                    chi2, p, _, _ = chi2_contingency(contingency)
                    significance_matrix[i, j] = int(p < alpha)
                else:
                    significance_matrix[i, j] = 0
    
    # Plot usando Matplotlib
    plt.figure(figsize=(8,6))
    plt.imshow(significance_matrix, cmap='Reds', vmin=0, vmax=1)
    plt.colorbar(label=f'Significant at alpha={alpha}')
    
    plt.xticks(ticks=np.arange(n), labels=cat_cols, rotation=45, ha='right')
    plt.yticks(ticks=np.arange(n), labels=cat_cols)
    
    # mostrar valores dentro das células
    for i in range(n):
        for j in range(n):
            plt.text(j, i, int(significance_matrix[i,j]), ha='center', va='center', color='black')
    
    plt.title(f"Categorical Association Heatmap (Top {top_k} categories)")
    plt.tight_layout()
    plt.show()

######################################################
# ===================== CLasses =====================

######################################################
# ===================== CLasses =====================

# ---- Transformer para sklearn ----

class FunctionPipelineTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, functions):
        self.functions = functions
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_transformed = X.copy()
        for func in self.functions:
            X_transformed = func(X_transformed)
        return X_transformed



