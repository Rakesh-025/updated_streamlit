import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

#### Basic EDA of Data
data=pd.read_excel(r"C:\Model Deployment in ML\360diditmg Project\pro_71.xlsx")
data.shape
data.columns
data.describe()
data.info()
data.isnull().sum()
data.head()

### Identify duplicates records in the data ###
duplicate = data.duplicated()
duplicate
sum(duplicate)

# Removing Duplicates
data = data.drop_duplicates()
# check for count of NA'sin each column
data.isna().sum()

# Create an imputer object that fills 'Nan' values
# Mean and Median imputer are used for numeric data 
# Mode is used for discrete data 

# for Mean, Meadian, Mode imputation we can use Simple Imputer or data.fillna()
from sklearn.impute import SimpleImputer
# Median Imputer for numerical data like Maintenance_cost, marketing cost, debentures,duration_of_coaching_in_hours salary of the trainer,profit margin, competitor price
median_imputer = SimpleImputer(missing_values=np.nan, strategy='median')
data["Maintenance_cost"] = pd.DataFrame(median_imputer.fit_transform(data[["Maintenance_cost"]]))
data["Marketing_cost"] = pd.DataFrame(median_imputer.fit_transform(data[["Marketing_cost"]]))
data["Debentures"] = pd.DataFrame(median_imputer.fit_transform(data[["Debentures"]]))
data["Salary_of_the_trainer"] = pd.DataFrame(median_imputer.fit_transform(data[["Salary_of_the_trainer"]]))
data["Profit_Margin"] = pd.DataFrame(median_imputer.fit_transform(data[["Profit_Margin"]]))
data["Competitor_Price"] = pd.DataFrame(median_imputer.fit_transform(data[["Competitor_Price"]]))
data["Duration_of_coaching_in_Hours"] = pd.DataFrame(median_imputer.fit_transform(data[["Duration_of_coaching_in_Hours"]]))
 
data.isna().sum()

# Mode Imputer for column mode_of_class, Name_of_course, Placement_Gurante/Assistance, location, level of course, Trainer_qualification,,level of maintenenace, level of marketing,certificated issued or not
mode_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
data["Mode_of_Class"] = pd.DataFrame(mode_imputer.fit_transform(data[["Mode_of_Class"]]))
data["Name_of_Course"] = pd.DataFrame(mode_imputer.fit_transform(data[["Name_of_Course"]]))
data["Placement_Gurante/Assistance"] = pd.DataFrame(mode_imputer.fit_transform(data[["Placement_Gurante/Assistance"]]))
data["Location"] = pd.DataFrame(mode_imputer.fit_transform(data[["Location"]]))
data["Level_of_Course"] = pd.DataFrame(mode_imputer.fit_transform(data[["Level_of_Course"]]))
data["Trainer_Qualification"] = pd.DataFrame(mode_imputer.fit_transform(data[["Trainer_Qualification"]]))
data["Level_of_Maintenance"] = pd.DataFrame(mode_imputer.fit_transform(data[["Level_of_Maintenance"]]))
data["Level_of_Marketing"] = pd.DataFrame(mode_imputer.fit_transform(data[["Level_of_Marketing"]]))
data["Certificate_issued_or_not"] = pd.DataFrame(mode_imputer.fit_transform(data[["Certificate_issued_or_not"]]))

data.isna().sum()

#### Visualization of variables #####

sns.countplot(data.Competitor_Price)
sns.jointplot(x='Competitor_Price' , y='Duration_of_coaching_in_Hours', data=data)
sns.jointplot(x='Competitor_Price' , y='Maintenance_cost', data=data)
sns.jointplot(x='Competitor_Price' , y='Debentures', data=data)
sns.relplot(x='Competitor_Price' , y='Salary_of_the_trainer', data=data)
sns.relplot(x='Competitor_Price' , y='Profit_Margin', data=data)
sns.relplot(x='Competitor_Price' , y='Trainer_Qualification', data=data)
sns.relplot(x='Competitor_Price' , y='Location', data=data)
sns.jointplot(x='Competitor_Price' , y='Location', data=data)

#outlier analysis##
#### Of Duration_of_coaching_in_Hours 
IQR = data["Duration_of_coaching_in_Hours"].quantile(0.75)-data["Duration_of_coaching_in_Hours"].quantile(0.25)
upper_limit=data["Duration_of_coaching_in_Hours"].quantile(0.75)+1.5*IQR
lower_limit=data["Duration_of_coaching_in_Hours"].quantile(0.25)-1.5*IQR

############### Winsorization ###############
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['Duration_of_coaching_in_Hours'])

data['Duration_of_coaching_in_Hours'] = winsor.fit_transform(data[['Duration_of_coaching_in_Hours']])
sns.boxplot(data["Duration_of_coaching_in_Hours"])

##Maintenance_cost##
IQR = data["Maintenance_cost"].quantile(0.75)-data["Maintenance_cost"].quantile(0.25)
upper_limit=data["Maintenance_cost"].quantile(0.75)+1.5*IQR
lower_limit=data["Maintenance_cost"].quantile(0.25)-1.5*IQR
############### Winsorization ###############
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['Maintenance_cost'])

data['Maintenance_cost'] = winsor.fit_transform(data[['Maintenance_cost']])
sns.boxplot(data['Maintenance_cost'])

##Marketing_cost##
IQR = data["Marketing_cost"].quantile(0.75)-data["Marketing_cost"].quantile(0.25)
upper_limit=data["Marketing_cost"].quantile(0.75)+1.5*IQR
lower_limit=data["Marketing_cost"].quantile(0.25)-1.5*IQR
############### Winsorization ###############
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['Marketing_cost'])

data['Marketing_cost']  = winsor.fit_transform(data[['Marketing_cost']])
sns.boxplot(data['Marketing_cost'])

##Debentures
IQR = data["Debentures"].quantile(0.75)-data["Debentures"].quantile(0.25)
upper_limit=data["Debentures"].quantile(0.75)+1.5*IQR
lower_limit=data["Debentures"].quantile(0.25)-1.5*IQR
############### Winsorization ###############
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['Debentures'])

data["Debentures"] = winsor.fit_transform(data[['Debentures']])
sns.boxplot(data["Debentures"])

##Salary_of_the_trainer
IQR = data["Salary_of_the_trainer"].quantile(0.75)-data["Salary_of_the_trainer"].quantile(0.25)
upper_limit=data["Salary_of_the_trainer"].quantile(0.75)+1.5*IQR
lower_limit=data["Salary_of_the_trainer"].quantile(0.25)-1.5*IQR
############### Winsorization ###############
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['Salary_of_the_trainer'])

data["Salary_of_the_trainer"] = winsor.fit_transform(data[['Salary_of_the_trainer']])
sns.boxplot(data["Salary_of_the_trainer"])

##Profit_Margin
IQR = data["Profit_Margin"].quantile(0.75)-data["Profit_Margin"].quantile(0.25)
upper_limit=data["Profit_Margin"].quantile(0.75)+1.5*IQR
lower_limit=data["Profit_Margin"].quantile(0.25)-1.5*IQR
############### Winsorization ###############
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['Profit_Margin'])

data["Profit_Margin"] = winsor.fit_transform(data[['Profit_Margin']])
sns.boxplot(data["Profit_Margin"])

# Exploratory data analysis by sweetviz
#data.describe()
#import sweetviz
#eda_report = sweetviz.analyze(data)
#eda_report.show_html('EDA_report.html')


# Measures of Central Tendency / First moment business decision
data.Maintenance_cost.mean()
data.Duration_of_coaching_in_Hours.mean() # '.' is used to refer to the variables within object
data.Marketing_cost.mean()
data.Debentures.mean()
data.Salary_of_the_trainer.mean()
data.Profit_Margin.mean()

data.Certificate_issued_or_not.mode()
data. Level_of_Marketing.mode()
data.Trainer_Qualification.mode()
data.Level_of_Maintenance.mode()

# Measures of Dispersion / Second, third and fourth moment business decision
data.Maintenance_cost.var()   #variance
data.Maintenance_cost.std()   # std deviation
data.Maintenance_cost.skew()   #skewness
data.Maintenance_cost.kurt()    # kurtosis

data.Duration_of_coaching_in_Hours.var() 
data.Duration_of_coaching_in_Hours.std()
data.Duration_of_coaching_in_Hours.skew() 
data.Duration_of_coaching_in_Hours.kurt() 

data.Debentures.var()
data.Debentures.std()
data.Debentures.skew()
data.Debentures.kurt()

data.Marketing_cost.var()
data.Marketing_cost.std()
data.Marketing_cost.skew()
data.Marketing_cost.kurt()

data.Salary_of_the_trainer.var()
data.Salary_of_the_trainer.std()
data.Salary_of_the_trainer.skew()
data.Salary_of_the_trainer.kurt()

data.Profit_Margin.var()
data.Profit_Margin.std()
data.Profit_Margin.skew()
data.Profit_Margin.kurt()
range = max(data.Profit_Margin) - min(data.Profit_Margin) # range
range

######### Label Encoder ############
from sklearn.preprocessing import LabelEncoder
# creating instance of labelencoder
labelencoder = LabelEncoder()

# Data Split into Input and Output variables
X = data.iloc[:, 0:16]

y = data['Competitor_Price']
#y = data.iloc[:, 16:] # Alternative approach

data.columns

X['Mode_of_Class']= labelencoder.fit_transform(X['Mode_of_Class'])
X['Name_of_Course'] = labelencoder.fit_transform(X['Name_of_Course'])
X['Placement_Gurante/Assistance'] = labelencoder.fit_transform(X['Placement_Gurante/Assistance'])
X['Location'] = labelencoder.fit_transform(X['Location'])
X['Level_of_Course'] = labelencoder.fit_transform(X['Level_of_Course'])
X['Trainer_Qualification'] = labelencoder.fit_transform(X['Trainer_Qualification'])
X['Level_of_Maintenance'] = labelencoder.fit_transform(X['Level_of_Maintenance'])
X['Level_of_Marketing'] = labelencoder.fit_transform(X['Level_of_Marketing'])
X['Certificate_issued_or_not'] = labelencoder.fit_transform(X['Certificate_issued_or_not'])


### we have to convert y to data frame so that we can use concatenate function
# concatenate X and y
df_new = pd.concat([X, y], axis =1)

## rename column name
df_new.columns
df_new = df_new.rename(columns={0:'Competitor_Price'})
df_new.info()

df= df_new[['Maintenance_cost','Marketing_cost','Profit_Margin','Duration_of_coaching_in_Hours','Debentures']]
y = df_new.iloc[:, -1]

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#Fitting model with trainig data
regressor.fit(df,y)

# Saving model to disk
pickle.dump(regressor, open('model_Prediction.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model_Prediction.pkl','rb'))
print(model.predict([[5500, 6000, 10000,150,2000]]))

