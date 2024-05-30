# %%
# Libraries for data manipulation
import pandas as pd
import numpy as np

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go

# preprocessing libraries
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
# import iterative imputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Modeling libraries
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
#for classification tasks
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, RandomForestRegressor
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import KBinsDiscretizer
# Scoring libraries
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_absolute_error, mean_squared_error, r2_score, f1_score, precision_score, recall_score

# ignore warnings   
import warnings
warnings.filterwarnings('ignore')

# %% [markdown]
# ## 1- Load Data

# %%
# Load the data from the github repository
url = 'https://raw.githubusercontent.com/AliBadran716/Public-Datasets/main/UCI%20Heart%20Disease%20Data/heart_disease_uci.csv'
df = pd.read_csv(url, delimiter=',')
# Print the first 5 rows of the dataframe 
df.head()

# %%
# Print the last 5 rows of the dataframe
df.tail()

# %% [markdown]
# ## 2- Exploratory Data Analysis

# %%
# Print the descriptive statistics of the dataframe
df.describe()

# %%
# Print Column Information
df.info()

# %%
# Shape of the dataframe
df.shape

# %% [markdown]
# ### 2.1- Age

# %%
# Find the minimum and maximum values of the 'id' column in the dataframe
df['id'].min(), df['id'].max()

# %%
# age column
print(f"Minimum age : {df['age'].min()}")
print(f"Maximum age: {df['age'].max()}")
print("\nAge column description:")
df['age'].describe()

# %%
# Plot mean, median, mode of age
plt.figure(figsize=(10,6))
sns.histplot(df['age'], bins=30, kde=True)
plt.axvline(df['age'].mean(), color='red', label='Mean')
plt.axvline(df['age'].median(), color='blue', label='Median')
plt.axvline(df['age'].mode()[0], color='green', label='Mode')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')
plt.legend()
plt.show()
# Print the mean, median, mode of age
print('Mean:', df['age'].mean())
print('Median:', df['age'].median())
print('Mode:', df['age'].mode()[0])

# %%
# Plot age distribution with color based on sex using plotly
fig = px.histogram(df, x='age', color='sex', marginal='box', title='Age Distribution with Color Based on sex')
fig.show()
# Print the Count of each age and the sex of the person at this age
age_sex_counts = df.groupby(['age', 'sex']).size().reset_index(name='count')
print(age_sex_counts)

# %% [markdown]
# #### Conclusion
# - The minimum age to have a heart disease starts from 28 years old.
# - Most of the people get heart disease at the age of 53-54 years.
# - Most of the males and females get are with heart disease at the age of 54-55 years.
# - Male percentage in the data: 78.91%
# - Female Percentage in the data: 21.09%
# - Males are 274.23% more than females in the data.

# %% [markdown]
# ### 2.2- Sex

# %%
# Print the count of each sex in the dataframe
df['sex'].value_counts()

# %%
# calculate male and female count percentage
male_count = df['sex'].value_counts()[0]
female_count = df['sex'].value_counts()[1]
total_count = male_count + female_count

# calculate percentages
male_percentage = (male_count / total_count) * 100
female_percentage = (female_count / total_count) * 100

print(f"Male percentage in the data: {male_percentage:.2f}%")
print(f"Female Percentage in the data: {female_percentage:.2f}%")

# difference
difference_percentage = ((male_count - female_count) / female_count) * 100
print(f"Males are {difference_percentage:.2f}% more than females in the data.")

# %%
# Plot the count of females and maeles in the dataframe
fig = px.pie(df, names='sex', color='sex')
fig.show()

# %% [markdown]
# ### 2.3- Dataset Column

# %%
# find the unique values in dataset column
print(df['dataset'].unique())
print("\n")
print(df['dataset'].value_counts())

# %%
# plot the countplot of dataset column
sns.countplot(data=df, x='dataset', hue = 'sex')

# plots using plotly
fig = px.bar(df, x='dataset', color='sex', title='Location distribution with Color Based on sex')
fig.show()

#values count of dataset column grouped by sex
print(df.groupby('sex')['dataset'].value_counts())

# %%
# Plot dataset location distribution with color based on age using plotly
fig = px.scatter(df, x='dataset', y='age', color='age', title='Location Distribution with Age')
fig.update_traces(marker=dict(size=8, opacity=0.6), selector=dict(mode='markers'))
fig.show()
# Print the Count of each age and the sex of the person at this age
print(df.groupby('dataset')['age'].value_counts())


# %%
# make a plot of age column using plotly and coloring this by dataset column
fig = px.histogram(data_frame=df, x='age', color='dataset')
fig.show()

# print the mean median and mode of age column grouped by dataset column
print(f"Mean of Data Set: {df.groupby('dataset')['age'].mean()}")
print("-------------------------------------")
print(f"Median of Data Set: {df.groupby('dataset')['age'].median()}")
print("-------------------------------------")
print(f"Mode of Data Set: {df.groupby('dataset')['age'].agg(pd.Series.mode)}")
print("-------------------------------------")

# %% [markdown]
# #### Conclusion
# 
# - We have highest number of people from Cleveland (304) and lowest from Switzerland (123).
# - The highest number of females in this dataset are from Cleveland (97) and lowest from VA Long Beach (6).
# - The highest number of males in this dataset are from Hungary (212) and lowest from Switzerland (113).

# %% [markdown]
# ##### The Mean Age according to the dataset is :
# 
# - Cleveland 54.351974
# 
# - Hungary 47.894198
# 
# - Switzerland 55.317073
# 
# - VA Long Beach 59.350000
# 
# ##### The Median Age according to the dataset is :
# 
# - Cleveland 55.5
# 
# - Hungary 49.0
# 
# - Switzerland 56.0
# 
# - VA Long Beach 60.0
# 
# ##### The Mode Age according to the dataset is :
# 
# - Cleveland 58
# 
# - Hungary 54
# 
# - Switzerland 61
# 
# - VA Long Beach [62, 63]
# 
# 

# %% [markdown]
# ### 2.4- Chest Pain

# %% [markdown]
# #### Values Description
# 1- Asymptomatic: No chest pain or discomfort.
# 
# 2- Non-Anginal: Chest pain not typical of heart-related issues; requires further investigation.
# 
# 3- Atypical Angina: Chest pain with characteristics different from typical heart-related chest pain.
# 
# 4- Typical Angina: Classic chest pain indicating potential insufficient blood supply to the heart.

# %%
# Values count
df['cp'].value_counts()

# %%
# Plot the count of cp column
cp_num = pd.crosstab(df.cp,df.num)
cp_num

# %%
# use crosstab to plot the count of chest pain type and heart disease
pd.crosstab(df.cp,df.num).plot(kind="bar",figsize=(10,6), 
                               color = ['salmon','blue','pink','lightblue','green'])
plt.title('Heart disease frequency for Chest Pain type')
plt.xlabel('Chest Pain type')
plt.ylabel('Amount')
plt.legend(['No Disease', 'Stage 1', 'Stage 2', 'Stage 3', 'Stage 4'])

# %%
# Plot dataset location distribution with color based on sex using plotly
fig = px.histogram(df, x='cp', color='sex', marginal='box', title='Chest Pain Type distribution with Color Based on sex')
fig.show()
# Print the Count of each age and the sex of the person at this age
cp_sex_counts = df.groupby(['sex', 'cp']).size().reset_index(name='count')
print(cp_sex_counts)

# %%
# count plot of cp column with dataset(countires) column
sns.countplot(df, x='cp', hue='dataset')
# Print the Count of cp and the dataset of the person at this cp
cp_dataset_counts = df.groupby(['cp', 'dataset']).size().reset_index(name='count')
print(cp_dataset_counts)

# %%
# draw the plot of age column grouped by cp column using plotly
fig = px.histogram(data_frame=df, x='age', color='cp', title='Age Distribution with Color Based on Chest Pain Type')
fig.show()
# Print the Count of each age and the cp of the person at this age
print(df.groupby('cp')['age'].value_counts())

# %%
# draw the plot of cp column grouped by num column using plotly
fig = px.histogram(data_frame=df, x='cp', color='num', title='Chest Pain Type Distribution with Color Based on Heart Disease')
fig.show()

# %% [markdown]
# #### Conclusion
# - A total of 104 individuals are identified as having neither chest pain nor heart disease.
# 
# - Only 23 individuals are found to have no chest pain while experiencing critical heart disease.
# 
# - A group of 83 individuals is observed to be free from chest pain while having severe heart disease.
# 
# - In the dataset, 197 individuals are noted for having no chest pain and exhibiting mild heart disease.
# 
# - Among the individuals, 89 have no chest pain while presenting with moderate heart disease.

# %% [markdown]
# ### 2.5- Resting Blood Pressure

# %%
# find the value counts of trestbps column
df['trestbps'].describe()

# %%
# Plotting Resting Blood Pressure vs Disease with a Boxplot
fig = go.Figure()
fig.add_trace(go.Box(y=df['trestbps'].values , name='BP at Rest for all', marker_color = 'green',boxmean=True))
fig.add_trace(go.Box(y=df[df['num']== 0]['trestbps'].values, name ='No Disease', marker_color = 'blue', boxmean = True))
fig.add_trace(go.Box(y=df[df['num'] !=0]['trestbps'].values, name ='Heart Disease', marker_color = 'red', boxmean = True))
fig.update_layout(title = 'BP Distribution (at rest)', yaxis_title = 'Blood Pressure (mm/Hg)', title_x = 0.5)
fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
fig.show()

# %%
# Plot mean, median, mode of trestbps
plt.figure(figsize=(10,6))
sns.histplot(df['trestbps'], bins=30, kde=True)
plt.axvline(df['trestbps'].mean(), color='red', label='Mean')
plt.axvline(df['trestbps'].median(), color='blue', label='Median')
plt.axvline(df['trestbps'].mode()[0], color='green', label='Mode')
plt.title('Resting Blood Pressure Distribution')
plt.xlabel('Resting Blood Pressure')
plt.ylabel('Count')
plt.legend()
plt.show()
# Print the mean, median, mode of trestbps
print('Mean:', df['trestbps'].mean())
print('Median:', df['trestbps'].median())
print('Mode:', df['trestbps'].mode()[0])

# %%
# Scatter plot of trestbps column by age column and coloring with dataset column
fig = px.scatter(data_frame=df, x='age', y='trestbps', color='dataset', symbol='sex', symbol_map={'male': 'square', 'female': 'circle'})
fig.update_layout(title='Resting Blood Pressure vs Age vs Sex', xaxis_title='Age', yaxis_title='Resting Blood Pressure')
fig.show()

# %% [markdown]
# ### 2.6- Cholestrol column

# %%
# Print the statistical description of the chol column
df['chol'].describe()

# %%
# Print the distribution and statistical properties of the contingency table created by the cross-tabulation of cholesterol levels and the target variable. 
cross = pd.crosstab(df['chol'], df['num']).describe()
cross

# %%
# Plot mean, median, mode of chol
plt.figure(figsize=(10,6))
sns.histplot(df['chol'], bins=30, kde=True)
plt.axvline(df['chol'].mean(), color='red', label='Mean')
plt.axvline(df['chol'].median(), color='blue', label='Median')
plt.axvline(df['chol'].mode()[0], color='green', label='Mode')
plt.title('Cholesterol Distribution')
plt.xlabel('Cholesterol')
plt.ylabel('Count')
plt.legend()
plt.show()
# Print the mean, median, mode of chol
print('Mean:', df['chol'].mean())
print('Median:', df['chol'].median())
print('Mode:', df['chol'].mode()[0])

# %%
# Scatter plot of chol column by age column and coloring with dataset column and sex
fig = px.scatter(data_frame=df, x='age', y='chol', color='dataset', symbol='sex', symbol_map={'male': 'square', 'female': 'circle'})
fig.update_layout(title='Cholesterol vs Age by Dataset and Sex', xaxis_title='Age', yaxis_title='Cholesterol')
fig.show()

# %%
# Boxplot of cholesterol levels by heart disease
sns.boxplot(y=df['chol'], hue=df['num'])

# %%
# draw the plot of chol column grouped by num column using plotly
fig = go.Figure()
fig.add_trace(go.Violin(y=df['chol'].values , name='All Patient', marker_color = 'green'))
fig.add_trace(go.Violin(y=df[df['num']== 0]['chol'].values, name ='No Disease', marker_color = 'blue'))
fig.add_trace(go.Violin(y=df[df['num'] == 4]['chol'].values, name ='Heart Disease', marker_color = 'red'))
fig.update_layout(title = 'Cholesterol Level Distribution', yaxis_title = 'Cholesterol Level', title_x = 0.5 )
fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
fig.show()

# %% [markdown]
# ### 2.7- Fasting Blood Sugar Column

# %%
# Print the value counts of fbs column
df['fbs'].value_counts()

# %%
# Plot the count of fbs in the dataframe
fig = px.pie(df, names='fbs', color='fbs', title='Fasting Blood Sugar Distribution')
fig.show()

# %%
# Plot Fasting Blood Sugar value counts distribution with cp, sex, dataset and age independent plots
for col in ['cp', 'sex', 'dataset', 'age']:
  plt.figure(figsize=(10,6))
  sns.countplot(data=df, x='fbs', hue=col)
  plt.title(f'Fasting Blood Sugar Value Counts by {col}')
  plt.xlabel('Fasting Blood Sugar')
  plt.ylabel('Count')
  plt.xticks(rotation=0)  # Rotate x-axis labels if needed
  plt.show()
  # Print the Count of each age and the col
  fbs_col_counts = df.groupby(['fbs', col]).size().reset_index(name='count')
  print(fbs_col_counts)

# %% [markdown]
# ### 2.8- ECG observation at resting condition Column

# %%
# Print the value counts of restecg column
df['restecg'].value_counts()

# %%
# Plot Resting Electrocardiographic Results value counts distribution
plt.figure(figsize=(10,6))
df['restecg'].value_counts().plot(kind='bar', color='skyblue')
plt.title('Resting Electrocardiographic Results Value Counts')
plt.xlabel('Resting Electrocardiographic Results')
plt.ylabel('Count')
plt.xticks(rotation=0)  # Rotate x-axis labels if needed
plt.show()

# %%
# Plot restecg value counts distribution with fbs, cp, sex, dataset and age independent plots
for col in ['fbs','cp', 'sex', 'dataset', 'age']:
  plt.figure(figsize=(10,6))
  sns.countplot(data=df, x='restecg', hue=col)
  plt.title(f'Resting Electrocardiographic Value Counts by {col}')
  plt.xlabel('Resting Electrocardiographic')
  plt.ylabel('Count')
  plt.xticks(rotation=0)  # Rotate x-axis labels if needed
  plt.show()
  # Print the Count of each age and the col
  restecg_col_counts = df.groupby(['restecg', col]).size().reset_index(name='count')
  print(restecg_col_counts)

# %% [markdown]
# ### 2.9- Thalesmia Column

# %%
# Print the value counts of thalach column
df['thal'].value_counts()

# %%
# Group by thal by sex
df.groupby('thal')['sex'].value_counts()

# %%
# Groupby Thal by Dataset(Countries)
df.groupby('thal')['dataset'].value_counts()

# %%
# count plot of cp column by dataset column
sns.countplot(df, x='thal', hue='sex')

# %%
# draw the plot of thal column grouped by age column using plotly
fig = px.histogram(data_frame=df, x='age', color='thal')
fig.show()

# %%
# draw the plot of thal column grouped by age column using plotly
fig = px.histogram(data_frame=df, x='thal', color='dataset')
fig.show()

# %%
# Plot or groupby to check the people who have thal does the have cp or not
df.groupby('thal')['cp'].value_counts()
#Visualize
sns.countplot(df, x='thal', hue='cp')

# %%
# Now Check People with Thal Survive or Not 
df.groupby('thal')['num'].value_counts()
# Plot to Visualize
sns.countplot(df, x='thal', hue='num' , palette='viridis')

# %% [markdown]
# #### Conclusion
# - Among the individuals, 110 males and 86 females are classified as normal.
# - A total of 42 males and 4 females exhibit a fixed defect.
# - In the dataset, 171 males and 21 females are identified with a reversible defect. The higher ratio of males compared to females is attributed to the dataset's male predominance.
# - Both individuals with thalassemia and those with normal thalassemia experience chest pain.
# - Individuals with normal thalassemia often exhibit a higher ratio of being free from heart disease, although some may still experience heart-related conditions.
# - Those with thalassemia generally have an increased likelihood of heart disease, yet some individuals with thalassemia do not develop such health issues.

# %% [markdown]
# ### 2.8- Target Column

# %% [markdown]
# Values Categories
# 
# 0 = no heart disease
# 
# 1 = mild heart disease
# 
# 2 = moderate heart disease
# 
# 3 = severe heart disease
# 
# 4 = critical heart disease
# 

# %%
df['num'].value_counts()

# %%
# Groupby num with sex 
df.groupby('num')['sex'].value_counts()
# Plot to Visualize
sns.countplot(df, x='num', hue='sex')

# %%
# groupby num by age 
df.groupby('num')['age'].value_counts()
# Plot to Visualize
sns.histplot(df, x='age', hue='num')

# %%
# Make Histplot using Plotly 
px.histogram(data_frame=df, x='age', color='num')

# %% [markdown]
# ### Conclusion
# - Men exhibit a higher ratio of being disease-free, while females show a lower ratio in the dataset.
# 
# - Conversely, based on the dataset, men are more affected by diseases compared to women. </h3>

# %% [markdown]
# ## 3-Dealing with outliers, Missing values and Duplicates

# %%
# define categorical and numeric columns
categorical_cols = ['thal', 'ca', 'slope', 'exang', 'restecg','fbs', 'cp', 'sex', 'num']
bool_cols = ['fbs', 'exang']
numeric_cols = ['oldpeak', 'thalch', 'chol', 'trestbps', 'age']

# %% [markdown]
# #### Outliers
# 
# 

# %%
plt.figure(figsize=(20, 20))

colors = ['red', 'green', 'blue', 'orange', 'purple']

for i, col in enumerate(numeric_cols):
    plt.subplot(3, 2, i+1)
    sns.boxplot(x=df[col], color=colors[i])
    plt.title(col)
plt.show()

# %%
fig = px.box(data_frame=df, y='age')
fig.show()

fig = px.box(data_frame=df, y='trestbps')
fig.show()

fig = px.box(data_frame=df, y='chol')
fig.show()

fig = px.box(data_frame=df, y='thalch')
fig.show()

fig = px.box(data_frame=df, y='oldpeak')
fig.show()

# %%
# remove the row from data with 0 trestbps as it is obviously an error in the data
df = df[df['trestbps'] != 0]

# %% [markdown]
# - The rest of outliers in this dataset gives us a lot of information about the data, so we wont remove any of them.

# %% [markdown]
# #### Missing Values
# - This dataset have some columns with higher missing values ratios, so we use advanced methods to impute them.
#     - We use iterative imputer to impute the missing values with Random forest classifier and Random Forest Regressor.

# %%
df.isnull().sum()[df.isnull().sum() > 0].sort_values(ascending=False)
missing_data_cols = df.isnull().sum()[df.isnull().sum() > 0].index.tolist()
missing_data_cols

# %%
#Function to impute missing values in categorical columns
def impute_categorical_missing_data(selected_col):
    df_null = df[df[selected_col].isnull()] # select rows with null values in the column
    df_not_null = df[df[selected_col].notnull()] # select rows with not null values in the column

    X = df_not_null.drop(selected_col, axis=1) # select all rows except the column we want to impute
    y = df_not_null[selected_col] # select the column we want to impute
    
    other_missing_cols = [col for col in missing_data_cols if col != selected_col] # select all columns except the column we want to impute
    
    label_encoder = LabelEncoder() # instantiate label encoder

    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype == 'category': # check if the column is categorical not numeric
            X[col] = label_encoder.fit_transform(X[col]) # encode the column to allow for imputation

    if selected_col in bool_cols:
        y = label_encoder.fit_transform(y) # encode the column to allow for imputation
        
    iterative_imputer = IterativeImputer(estimator=RandomForestRegressor(random_state=42), add_indicator=True) # instantiate iterative imputer

    for col in other_missing_cols:
        if X[col].isnull().sum() > 0: # check if the column has missing values
            col_with_missing_values = X[col].values.reshape(-1, 1) 
            imputed_values = iterative_imputer.fit_transform(col_with_missing_values) # impute missing values
            X[col] = imputed_values[:, 0] # replace missing values with imputed values
        else:
            pass
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # split data into training and testing sets

    rf_classifier = RandomForestClassifier() # instantiate random forest classifier for categorical columns

    rf_classifier.fit(X_train, y_train) # fit the model

    y_pred = rf_classifier.predict(X_test) # make predictions

    acc_score = accuracy_score(y_test, y_pred) # calculate accuracy

    print("The feature '"+ selected_col+ "' has been imputed with", round((acc_score * 100), 2), "accuracy\n") 

    X = df_null.drop(selected_col, axis=1) # select all rows except the column we want to impute

    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype == 'category': # check if the column is categorical not numeric
            X[col] = label_encoder.fit_transform(X[col]) # encode the column to allow for imputation
 
    for col in other_missing_cols:
        if X[col].isnull().sum() > 0: # check if the column has missing values
            col_with_missing_values = X[col].values.reshape(-1, 1)
            imputed_values = iterative_imputer.fit_transform(col_with_missing_values) # impute missing values
            X[col] = imputed_values[:, 0] # replace missing values with imputed values
        else:
            pass
                
    if len(df_null) > 0: # check if there are any rows with null values
        df_null[selected_col] = rf_classifier.predict(X) # predict missing values using the random forest classifier
        if selected_col in bool_cols: # check if the column is boolean
            df_null[selected_col] = df_null[selected_col].map({0: False, 1: True}) # map 0 and 1 to False and True
        else:
            pass 
    else:
        pass # if there are no rows with null values, do nothing

    df_combined = pd.concat([df_not_null, df_null]) # combine the imputed and non-imputed dataframes
    
    return df_combined[selected_col]

# %%
#Function to impute missing values in numeric columns
def impute_numerical_missing_data(selected_col):
    df_null = df[df[selected_col].isnull()] # select rows with null values in the column
    df_not_null = df[df[selected_col].notnull()] # select rows with not null values in the column

    X = df_not_null.drop(selected_col, axis=1) # select all rows except the column we want to impute
    y = df_not_null[selected_col] # select the column we want to impute
    
    other_missing_cols = [col for col in missing_data_cols if col != selected_col] # select all columns except the column we want to impute
    
    label_encoder = LabelEncoder() # instantiate label encoder

    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype == 'category': # check if the column is categorical not numeric
            X[col] = label_encoder.fit_transform(X[col]) # encode the column to allow for imputation
    
    iterative_imputer = IterativeImputer(estimator=RandomForestRegressor(random_state=42), add_indicator=True) # instantiate iterative imputer

    for col in other_missing_cols:
        if X[col].isnull().sum() > 0: # check if the column has missing values
            col_with_missing_values = X[col].values.reshape(-1, 1) 
            imputed_values = iterative_imputer.fit_transform(col_with_missing_values) # impute missing values
            X[col] = imputed_values[:, 0] # replace missing values with imputed values
        else:
            pass # if the column has no missing values, do nothing
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # split data into training and testing sets

    rf_regressor = RandomForestRegressor() # instantiate random forest regressor for numeric columns

    rf_regressor.fit(X_train, y_train) # fit the model

    y_pred = rf_regressor.predict(X_test) # make predictions

    print("MAE =", mean_absolute_error(y_test, y_pred), "\n") # calculate mean absolute error
    print("RMSE =", mean_squared_error(y_test, y_pred, squared=False), "\n") # calculate root mean squared error
    print("R2 =", r2_score(y_test, y_pred), "\n") # calculate R-squared

    X = df_null.drop(selected_col, axis=1) # select all rows except the column we want to impute

    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype == 'category': # check if the column is categorical not numeric
            X[col] = label_encoder.fit_transform(X[col]) # encode the column to allow for imputation

    for col in other_missing_cols:
        if X[col].isnull().sum() > 0: # check if the column has missing values
            col_with_missing_values = X[col].values.reshape(-1, 1) 
            imputed_values = iterative_imputer.fit_transform(col_with_missing_values) # impute missing values
            X[col] = imputed_values[:, 0] # replace missing values with imputed values
        else:
            pass # if the column has no missing values, do nothing
                
    if len(df_null) > 0:  # check if there are any rows with null values
        df_null[selected_col] = rf_regressor.predict(X) # predict missing values using the random forest regressor
    else:
        pass # if there are no rows with null values, do nothing

    df_combined = pd.concat([df_not_null, df_null]) # combine the imputed and non-imputed dataframes
    
    return df_combined[selected_col]

# %%
df.isnull().sum()[df.isnull().sum() > 0].sort_values(ascending=False) 

# %%
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis', yticklabels=False)
plt.show()

# %%
# impute missing values
for col in missing_data_cols:
    print("Missing Values", col, ":", str(round((df[col].isnull().sum() / len(df)) * 100, 2))+"%")
    if col in categorical_cols:
        df[col] = impute_categorical_missing_data(col)
    elif col in numeric_cols:
        df[col] = impute_numerical_missing_data(col)
    else:
        pass

# %%
#check for missing values after imputing
df.isnull().sum()

# %%
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis', yticklabels=False)
plt.show()

# %% [markdown]
# #### Checking for Duplicates

# %%
df.duplicated().sum()

# %%
#Rename attributes for consistency 
#Removing any spaces in the names an encoding boolean values
data = df.copy()


data['thal'].replace({'fixed defect':'fixed_defect' , 'reversable defect': 'reversable_defect' }, inplace =True)
data['cp'].replace({'typical angina':'typical_angina', 'atypical angina': 'atypical_angina' }, inplace =True)
data['restecg'].replace({'normal': 'normal' , 'st-t abnormality': 'ST-T_wave_abnormality' , 'lv hypertrophy': 'left_ventricular_hypertrophy' }, inplace =True)

# Genrating New Dataset with Less Columns Which Are Necessary .
data_1 = data[['age','sex','cp','dataset', 'trestbps', 'chol', 'fbs','restecg' , 'thalch', 'exang', 'oldpeak', 'slope', 'ca', 'thal']].copy()
# Some Changes in Target Variable | Only Two Categories (0,1) . 0 for No-Disease , 1 for Disease
data_1['target'] = ((data['num'] > 0)*1).copy()
# Encoding Sex 
data_1['sex'] = (data['sex'] == 'Male')*1
# Encoding Fbs and exang
data_1['fbs'] = (data['fbs'])*1
data_1['exang'] = (data['exang'])*1

# Renaming Columns Names

data_1.columns = ['age', 'sex', 'chest_pain_type','country' ,'resting_blood_pressure', 
              'cholesterol', 'fasting_blood_sugar','Restecg',
              'max_heart_rate_achieved', 'exercise_induced_angina', 
              'st_depression', 'st_slope_type', 'num_major_vessels', 
              'thalassemia_type', 'target']
# Load Data Sample 
data_1.head()

# %%
# correlation matrix
plt.figure(figsize=(15, 10))
sns.heatmap(data[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")    
plt.show()

# %% [markdown]
# ## 4-Modeling

# %%
def logistic_regression(data):
  """
  Perform logistic regression on the given dataset.

  Parameters:
  - data: pandas DataFrame
    The input dataset containing the features and target variable.

  Returns:
  - None

  This function performs logistic regression by encoding categorical variables, scaling numerical variables,
  splitting the dataset into training and testing sets, and fitting a logistic regression model.
  It then prints the accuracy of the model on the training and test sets.
  """
  X= data_1.drop('target', axis=1)
  y = data_1['target']
  label_ecoders = {}  # Dictionary to store label_ecoders
  for col in X.columns:
    if X[col].dtype == 'object' or X[col].dtype == 'category':
      le = LabelEncoder()
      le.fit(X[col])
      X[col] = le.transform(X[col])
      label_ecoders[col] = le
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

  scaler = StandardScaler()   

  X_train = scaler.fit_transform(X_train)
  X_test = scaler.transform(X_test)
  
# Logistic Regression model with cross-validation
  log_reg = LogisticRegression(max_iter=10000)
  param_grid = {
      'C': [0.1, 1, 10, 100],
      'solver': ['liblinear', 'saga']
  }

  grid = GridSearchCV(log_reg, param_grid, refit=True, cv=5)
  grid.fit(X_train, y_train)

  model = grid.best_estimator_
  best_params = grid.best_params_

  y_pred_train = model.predict(X_train)
  y_pred_test = model.predict(X_test)

  # Training set metrics
  print("\nBest Parameters:", best_params)
  print("\nTraining Accuracy:", accuracy_score(y_train, y_pred_train))
  print("\nTraining Precision:", precision_score(y_train, y_pred_train, average='macro'))
  print("\nTraining Recall:", recall_score(y_train, y_pred_train, average='macro'))
  print("\nTraining F1 Score:", f1_score(y_train, y_pred_train, average='macro'))

  # Test set metrics
  print("\nTest Accuracy:", accuracy_score(y_test, y_pred_test))
  print("\nTest Precision:", precision_score(y_test, y_pred_test, average='macro'))
  print("\nTest Recall:", recall_score(y_test, y_pred_test, average='macro'))
  print("\nTest F1 Score:", f1_score(y_test, y_pred_test, average='macro'))
  # Plot confusion matrix
  cm = confusion_matrix(y_test, y_pred_test)
  plt.figure(figsize=(8, 6))
  sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
  plt.xlabel('Predicted')
  plt.ylabel('Actual')
  plt.title('Confusion Matrix')
  plt.show()

  #decode labels
  for col, le in label_ecoders.items():
    X[col] = le.inverse_transform(X[col])


# %%
def naive_gaussian_model(data):
  X= data_1.drop('target', axis=1)
  y = data_1['target']
  label_ecoders = {}  # Dictionary to store label_ecoders
  for col in X.columns:
    if X[col].dtype == 'object' or X[col].dtype == 'category':
      le = LabelEncoder()
      le.fit(X[col])
      X[col] = le.transform(X[col])
      label_ecoders[col] = le
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

  scaler = StandardScaler()   

  X_train = scaler.fit_transform(X_train)
  X_test = scaler.transform(X_test)
  
# Naive Bayes model
  model = GaussianNB()
  model.fit(X_train, y_train)

  y_pred_train = model.predict(X_train)
  y_pred_test = model.predict(X_test)

  # Training set metrics
  print("\nTraining Accuracy:", accuracy_score(y_train, y_pred_train))
  print("\nTraining Precision:", precision_score(y_train, y_pred_train, average='macro'))
  print("\nTraining Recall:", recall_score(y_train, y_pred_train, average='macro'))
  print("\nTraining F1 Score:", f1_score(y_train, y_pred_train, average='macro'))

  # Test set metrics
  print("\nTest Accuracy:", accuracy_score(y_test, y_pred_test))
  print("\nTest Precision:", precision_score(y_test, y_pred_test, average='macro'))
  print("\nTest Recall:", recall_score(y_test, y_pred_test, average='macro'))
  print("\nTest F1 Score:", f1_score(y_test, y_pred_test, average='macro'))
 # Plot confusion matrix
  cm = confusion_matrix(y_test, y_pred_test)
  plt.figure(figsize=(8, 6))
  sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
  plt.xlabel('Predicted')
  plt.ylabel('Actual')
  plt.title('Confusion Matrix')
  plt.show() 

# %%
def KNN_model(data):
  X= data_1.drop('target', axis=1)
  y = data_1['target']
  label_ecoders = {}  # Dictionary to store label_ecoders
  for col in X.columns:
    if X[col].dtype == 'object' or X[col].dtype == 'category':
      le = LabelEncoder()
      le.fit(X[col])
      X[col] = le.transform(X[col])
      label_ecoders[col] = le
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

  scaler = StandardScaler()   

  X_train = scaler.fit_transform(X_train)
  X_test = scaler.transform(X_test)
# KNN model with cross-validation
  knn = KNeighborsClassifier()
  param_grid = {
      'n_neighbors': [3, 5, 7, 9],
      'weights': ['uniform', 'distance'],
      'metric': ['euclidean', 'manhattan']
  }

  grid = GridSearchCV(knn, param_grid, refit=True, cv=5)
  grid.fit(X_train, y_train)

  model = grid.best_estimator_
  best_params = grid.best_params_

  y_pred_train = model.predict(X_train)
  y_pred_test = model.predict(X_test)

  # Training set metrics
  print("\nBest Parameters:", best_params)
  print("\nTraining Accuracy:", accuracy_score(y_train, y_pred_train))
  print("\nTraining Precision:", precision_score(y_train, y_pred_train, average='macro'))
  print("\nTraining Recall:", recall_score(y_train, y_pred_train, average='macro'))
  print("\nTraining F1 Score:", f1_score(y_train, y_pred_train, average='macro'))

  # Test set metrics
  print("\nTest Accuracy:", accuracy_score(y_test, y_pred_test))
  print("\nTest Precision:", precision_score(y_test, y_pred_test, average='macro'))
  print("\nTest Recall:", recall_score(y_test, y_pred_test, average='macro'))
  print("\nTest F1 Score:", f1_score(y_test, y_pred_test, average='macro'))
# Plot confusion matrix
  cm = confusion_matrix(y_test, y_pred_test)
  plt.figure(figsize=(8, 6))
  sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
  plt.xlabel('Predicted')
  plt.ylabel('Actual')
  plt.title('Confusion Matrix')
  plt.show()

  #decode labels
  for col, le in label_ecoders.items():
    X[col] = le.inverse_transform(X[col])

# %%
def SVM_model(data):
  X= data_1.drop('target', axis=1)
  y = data_1['target']
  label_ecoders = {}  # Dictionary to store label_ecoders
  for col in X.columns:
    if X[col].dtype == 'object' or X[col].dtype == 'category':
      le = LabelEncoder()
      le.fit(X[col])
      X[col] = le.transform(X[col])
      label_ecoders[col] = le
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

  scaler = StandardScaler()   

  X_train = scaler.fit_transform(X_train)
  X_test = scaler.transform(X_test)
# SVM model with cross-validation
  svm = SVC()
  param_grid = {
      'C': [0.1, 1, 10, 100],
      'gamma': [1, 0.1, 0.01, 0.001],
      'kernel': ['rbf', 'linear']
  }

  grid = GridSearchCV(svm, param_grid, refit=True, cv=5)
  grid.fit(X_train, y_train)

  model = grid.best_estimator_
  best_params = grid.best_params_

  y_pred_train = model.predict(X_train)
  y_pred_test = model.predict(X_test)

  # Training set metrics
  print("\nBest Parameters:", best_params)
  print("\nTraining Accuracy:", accuracy_score(y_train, y_pred_train))
  print("\nTraining Precision:", precision_score(y_train, y_pred_train, average='macro'))
  print("\nTraining Recall:", recall_score(y_train, y_pred_train, average='macro'))
  print("\nTraining F1 Score:", f1_score(y_train, y_pred_train, average='macro'))

  # Test set metrics
  print("\nTest Accuracy:", accuracy_score(y_test, y_pred_test))
  print("\nTest Precision:", precision_score(y_test, y_pred_test, average='macro'))
  print("\nTest Recall:", recall_score(y_test, y_pred_test, average='macro'))
  print("\nTest F1 Score:", f1_score(y_test, y_pred_test, average='macro'))

  # Plot confusion matrix
  cm = confusion_matrix(y_test, y_pred_test)
  plt.figure(figsize=(8, 6))
  sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
  plt.xlabel('Predicted')
  plt.ylabel('Actual')
  plt.title('Confusion Matrix')
  plt.show()
  #decode labels
  for col, le in label_ecoders.items():
    X[col] = le.inverse_transform(X[col])

# %%
# Random Forest
def train_rf(data,target):
    X = data.drop('target',axis=1)
    y = data['target']
    label_ecoders = {}
    
    #label encoder for categorical columns
    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype == 'category':
            le = LabelEncoder()
            le.fit(X[col])
            X[col] = le.transform(X[col])
            label_ecoders[col] = le
            
    #split the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    #scaling data
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    #define model
    rf_model = RandomForestClassifier(random_state=42, class_weight='balanced')

    #hyperparameter tuning
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    #grid search
    grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='accuracy' )
    grid_search.fit(X_train, y_train)


    #define best model
    best_rf_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # fit best model
    best_rf_model.fit(X_train, y_train)

    #make predictions
    y_pred_train = best_rf_model.predict(X_train)
    y_pred_test = best_rf_model.predict(X_test)

      # Training set metrics
    print("\nBest Parameters:", best_params)
    print("\nTraining Accuracy:", accuracy_score(y_train, y_pred_train))
    print("\nTraining Precision:", precision_score(y_train, y_pred_train, average='macro'))
    print("\nTraining Recall:", recall_score(y_train, y_pred_train, average='macro'))
    print("\nTraining F1 Score:", f1_score(y_train, y_pred_train, average='macro'))

    # Test set metrics
    print("\nTest Accuracy:", accuracy_score(y_test, y_pred_test))
    print("\nTest Precision:", precision_score(y_test, y_pred_test, average='macro'))
    print("\nTest Recall:", recall_score(y_test, y_pred_test, average='macro'))
    print("\nTest F1 Score:", f1_score(y_test, y_pred_test, average='macro'))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred_test)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    #decode labels
    for col, le in label_ecoders.items():
        X[col] = le.inverse_transform(X[col])


# %%
def train_xgb(data,target):
    X = data.drop('target',axis=1)
    y = data['target']
    label_ecoders = {}
    
    #label encoder for categorical columns
    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype == 'category':
            le = LabelEncoder()
            le.fit(X[col])
            X[col] = le.transform(X[col])
            label_ecoders[col] = le
            
    #split the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    #scaling data
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    #define model
    xgb_model = XGBClassifier(random_state=0,)

    #hyperparameter tuning
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.2, 0.1, 0.01, 0.001],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'gamma': [0, 1, 2]

    }

    #grid search
    grid_search = GridSearchCV(xgb_model, param_grid, cv=5, scoring='accuracy' )
    grid_search.fit(X_train, y_train)


    #define best model
    best_xgb_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # fit best model
    best_xgb_model.fit(X_train, y_train)

    #make predictions
    y_pred_train = best_xgb_model.predict(X_train)
    y_pred_test = best_xgb_model.predict(X_test)

    # Training set metrics
    print("\nBest Parameters:", best_params)
    print("\nTraining Accuracy:", accuracy_score(y_train, y_pred_train))
    print("\nTraining Precision:", precision_score(y_train, y_pred_train, average='macro'))
    print("\nTraining Recall:", recall_score(y_train, y_pred_train, average='macro'))
    print("\nTraining F1 Score:", f1_score(y_train, y_pred_train, average='macro'))

    # Test set metrics
    print("\nTest Accuracy:", accuracy_score(y_test, y_pred_test))
    print("\nTest Precision:", precision_score(y_test, y_pred_test, average='macro'))
    print("\nTest Recall:", recall_score(y_test, y_pred_test, average='macro'))
    print("\nTest F1 Score:", f1_score(y_test, y_pred_test, average='macro'))
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred_test)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    #decode labels
    for col, le in label_ecoders.items():
        X[col] = le.inverse_transform(X[col])

# %%
# Models testing
print('Logistic Regression Model')
logistic_regression(data)

# %%
print('Naive Gaussian Model')
naive_gaussian_model(data)

# %%
print('KNN Model')
KNN_model(data)


# %%
print('SVM Model')
SVM_model(data)


# %%
print('Random Forest Model')
train_rf(data_1,'target')


# %%
print('XGBoost Model')
train_xgb(data_1,'target')


