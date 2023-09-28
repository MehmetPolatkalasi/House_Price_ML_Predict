import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

pd.set_option('display.max_columns', 20)
pd.set_option('display.width', None)

df = pd.read_csv("House_Price_Train.csv")

#######################################################################
# 1. Exploratory Data Analysis
#######################################################################

def check_df(dataframe, head=5):
    print("#################### Shape ######################")
    print(dataframe.shape, end="\n\n")
    print("#################### Types ######################")
    print(dataframe.dtypes, end="\n\n")
    print("#################### Head #######################")
    print(dataframe.head(head), end="\n\n")
    print("#################### Tail #######################")
    print(dataframe.tail(head), end="\n\n")
    print("#################### NA #########################")
    print(dataframe.isnull().sum(), end="\n\n")
    print("#################### Quantiles ##################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1], numeric_only=True).T, end="\n\n")

check_df(df)

df.drop("Id", axis=1, inplace=True)

for col in df.columns:
    if "Year" in col:
        df[col] = pd.to_datetime(df[col])
        df[col] = df[col].dt.strftime('%Y-%m-%d')
        df[col] = pd.to_datetime(df[col])

df['dateSold'] = df['YrSold'].astype(str) + '-' + df['MoSold'].astype(str) + '-01'
df['dateSold'] = pd.to_datetime(df['dateSold'])
df['dateSold'] = df['dateSold'].dt.strftime('%Y-%m-%d')
df['dateSold'] = pd.to_datetime(df['dateSold'])

df["GarageYrBlt"] = pd.to_datetime(df["GarageYrBlt"])
df['GarageYrBlt'] = df['GarageYrBlt'].dt.strftime('%Y-%m-%d')
df['GarageYrBlt'] = pd.to_datetime(df['GarageYrBlt'])

df.drop("YrSold", axis=1, inplace=True)
df.drop("MoSold", axis=1, inplace=True)


def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.

    Parameters
    ----------
    dataframe: dataframe
        değişken isimleri alınmak istenen dataframe'dir.
    cat_th: int, float
        numerik fakat kategorik olan değişkenler için sınıf eşik değeri
    car_th: int, float
        kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    -------
    cat_cols: list
        Kategorik değişken listesi
    num_cols: list
        Numerik değişken listesi
    cat_but_car: list
        Kategorik görünümlü kardinal değişken listesi

    Notes
    ------
    cat_cols + num_cols + cat_but_car = toplam değişken sayısı
    num_but_cat cat_cols'un içerisinde.


    """
    cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]
    num_but_cat = [col for col in df.columns if df[col].nunique() < cat_th and df[col].dtypes in ["int", "float", "int64", "float64"]]
    cat_but_car = [col for col in df.columns if
                   df[col].nunique() > car_th and str(df[col].dtypes) in ["category", "object"]]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in df.columns if df[col].dtypes in ["int", "float", "int64", "float64"]]
    num_cols = [col for col in num_cols if col not in cat_cols]

    print(f"Observation: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")

    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)


def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##################################################")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

for col in cat_cols:
    cat_summary(df, col)


def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)
    print("##################################################")

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

for col in num_cols:
    num_summary(df, col, plot=True)


def correlation_matrix(df, cols):
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    fig = sns.heatmap(df[cols].corr(), annot=True, linewidths=0.5, annot_kws={"size": 12}, linecolor="w", cmap="RdBu")
    plt.show(block=True)

correlation_matrix(df, num_cols)


def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")

for col in cat_cols:
    target_summary_with_cat(df, "SalePrice", col)


#######################################################################
# 2. Data Preprocessing & Feature Engineering
#######################################################################

df.columns = [col.upper() for col in df.columns]


def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns

missing_values_table(df)


df.drop(["POOLQC", "MISCFEATURE", "ALLEY", "FENCE", "FIREPLACEQU", "LOTFRONTAGE"], axis=1, inplace=True)

df.dropna(inplace=True)





