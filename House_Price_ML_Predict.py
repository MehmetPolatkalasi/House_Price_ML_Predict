import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingClassifier, AdaBoostRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

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

df["MOSOLD_CAT"] = None
df.loc[df["MOSOLD"] == 12, "MOSOLD_CAT"] = "winter"
df.loc[df["MOSOLD"] < 3, "MOSOLD_CAT"] = "winter"
df.loc[(df["MOSOLD"] >= 3) & (df["MOSOLD"] <= 5), "MOSOLD_CAT"] = "spring"
df.loc[(df["MOSOLD"] > 5) & (df["MOSOLD"] <= 8), "MOSOLD_CAT"] = "summer"
df.loc[(df["MOSOLD"] > 8) & (df["MOSOLD"] <= 11), "MOSOLD_CAT"] = "autumn"

cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=5)

for col in cat_cols:
    cat_summary(df, col)

cat_cols = [col for col in cat_cols if "SALEPRICE" not in col]


def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = one_hot_encoder(df, cat_cols, drop_first=True)
check_df(df)
df.columns = [col.upper() for col in df.columns]

cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=5)
cat_cols = [col for col in cat_cols if "SALEPRICE" not in col]


def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

for col in num_cols:
    print(col, check_outlier(df, col))

def replace_with_threshoulds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

for col in num_cols:
    replace_with_threshoulds(df, col)


X_scaled = StandardScaler().fit_transform(df[num_cols])
df[num_cols] = pd.DataFrame(X_scaled, columns=df[num_cols].columns)

check_df(df)
missing_values_table(df)
df.dropna(inplace=True)


y = df["SALEPRICE"]
X = df.drop(["SALEPRICE", "NEIGHBORHOOD"], axis=1)


#######################################################################
# 3. Base Model
#######################################################################
def base_models(X, y, scoring="neg_mean_squared_error"):
    print("Base Models....")
    regressors = [("LR", LogisticRegression()),
                  ("KNN", KNeighborsRegressor()),
                  ("SVC", SVC()),
                  ("CART", DecisionTreeRegressor()),
                  ("RF", RandomForestRegressor()),
                  ("Adaboost", AdaBoostRegressor()),
                  ("GBM", GradientBoostingRegressor()),
                  ("XGBoost", XGBRegressor()),
                  ("LightGBM", LGBMRegressor(verbose=-1)),
                  # ("CatBoost", CatBoostRegressor(verbose=False))
                  ]

    for name, regressor in regressors:
        cv_results = cross_validate(regressor, X, y, cv=3, scoring=scoring)
        print(f"{scoring}: {round(cv_results['test_score'].mean(), 4)} ({name}) ")

base_models(X, y)
df.dtypes






