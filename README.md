## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
## STEP 1:
Read the given Data.
## STEP 2:
Clean the Data Set using Data Cleaning Process.
## STEP 3:
Apply Feature Encoding for the feature in the data set.
## STEP 4:
Apply Feature Transformation for the feature in the data set.
## STEP 5:
Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation

• Reciprocal Transformation

• Square Root Transformation

• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method

• Yeojohnson method

# CODING AND OUTPUT:
```py
import pandas as pd
import numpy as np
import seaborn as sns
df=pd.read_csv('/content/Encoding Data.csv')
df
```
![alt text](output_1.png)
```py
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
![alt text](output_2.png)
```py
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
![alt text](output_3.png)
```py
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![output](output_4.png)
```py
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[['nom_0']]))
```
![alt text](output-5.png)
```py
df2=pd.concat([df2,enc],axis=1)
df2
```
![alt text](output-6.png)
```py
pd.get_dummies(df2,columns=["nom_0"])
```
![alt text](output-7.png)
```py
from category_encoders import BinaryEncoder
df=pd.read_csv('/data.csv')
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb1=df.copy()
dfb
```
![alt text](output-8.png)
```py
from category_encoders import TargetEncoder
te=TargetEncoder()
cc=df.copy()
new=te.fit_transform(X=cc["City"],y=cc["Target"])
cc=pd.concat([cc,new],axis=1)
cc
```
![alt text](output-9.png)
```py
import pandas as pd
import numpy as np
from scipy import stats
df=pd.read_csv("/content/Data_to_Transform.csv")
df
```
![alt text](output-10.png)
```py
df.skew()
```
![alt text](output-11.png)
```py
np.log(df['Highly Negative Skew'])
```
![alt text](output-12.png)
```py
np.reciprocal(df['Moderate Positive Skew'])
```
![alt text](output-13.png)
```py
np.sqrt(df['Highly Positive Skew'])
```
![alt text](output-14.png)
```py
df["Highly Positive Skew_Boxcox"],parameters=stats.boxcox(df['Highly Positive Skew'])
df
```
![alt text](output-15.png)
```py
df.skew()
```
![alt text](output-16.png)
```py
df["Moderate Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df['Moderate Negative Skew'])
df
```
![alt text](output-17.png)
```py
df.skew()
```
![alt text](output-18.png)
```py
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df['Moderate Negative Skew_1']=qt.fit_transform(df[['Moderate Negative Skew']])
df
```
![alt text](output-19.png)
```py
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats
sm.qqplot(df['Moderate Negative Skew'],line='45')
plt.show()
```
![alt text](output-20.png)
```py
sm.qqplot(df['Highly Negative Skew'],line='45')
plt.show()
```
![alt text](output-21.png)
```py
sm.qqplot(df['Moderate Negative Skew_1'],line='45')
plt.show()
```
![alt text](output-22.png)

# RESULT:
Thus,the Feature Encoding and Transformation process has been performed on the given data.

       
