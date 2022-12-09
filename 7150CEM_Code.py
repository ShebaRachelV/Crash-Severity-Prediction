# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 17:08:41 2022

@author: Sheba
"""
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 13:38:44 2022

@author: Sheba
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 10:35:49 2022

@author: Sheba
"""


import pandas as pd

# to increase the output width
pd.options.display.width= None
pd.options.display.max_columns= None
pd.set_option('display.max_rows', 40)
pd.set_option('display.max_columns', 40)

#to load the dataset
df1 = pd.read_csv("Young_Driver_Crashes.csv")
df2 = pd.read_csv("Fatal_Crashes.csv")
df3 = pd.read_csv("Older_Driver_Crashes.csv")

df4= df3[(df3['MostSevereInjury'] == 'K')|(df3['MostSevereInjury'] == 'C')|\
         (df3['MostSevereInjury'] == 'B') | (df3['MostSevereInjury'] == 'A')]
    
df5 = pd.read_csv("Distracted_Driver_Crashes.csv")
df6= df5[(df5['MostSevereInjury'] == 'K')|(df5['MostSevereInjury'] == 'C')|\
         (df5['MostSevereInjury'] == 'B') | (df5['MostSevereInjury'] == 'A')]

df7 = pd.read_csv("Car_Seat_Crashes.csv")
df8= df7[(df7['MostSevereInjury'] == 'K')|(df7['MostSevereInjury'] == 'C')|\
         (df7['MostSevereInjury'] == 'B') | (df7['MostSevereInjury'] == 'A')]

df9 = pd.read_csv("Wrong_Way_Crashes.csv")
df10= df9[(df9['MostSevereInjury'] == 'K')|(df9['MostSevereInjury'] == 'C')|\
         (df9['MostSevereInjury'] == 'B') | (df9['MostSevereInjury'] == 'A')]
    
df11 = pd.read_csv("Unlicensed_Driver_Crashes.csv")
df12= df11[(df11['MostSevereInjury'] == 'K')|(df11['MostSevereInjury'] == 'C')|\
         (df11['MostSevereInjury'] == 'B') | (df11['MostSevereInjury'] == 'A')]
    
df13 = pd.read_csv("Aggressive_Driving_Crashes.csv")
df14= df13[(df13['MostSevereInjury'] == 'K')|(df13['MostSevereInjury'] == 'C')|\
         (df13['MostSevereInjury'] == 'B') | (df13['MostSevereInjury'] == 'A')]
    
df15 = pd.read_csv("Roadway_Departure_Crashes.csv")
df16= df15[(df15['MostSevereInjury'] == 'K')|(df15['MostSevereInjury'] == 'C')|\
         (df15['MostSevereInjury'] == 'B') | (df15['MostSevereInjury'] == 'A')]

df17 = pd.read_csv("Motor_Coach.csv")
df18= df17[(df17['MostSevereInjury'] == 'K')|(df17['MostSevereInjury'] == 'C')|\
         (df17['MostSevereInjury'] == 'B') | (df17['MostSevereInjury'] == 'A')]

df19 = pd.read_csv("FMCSA_Crashes.csv")
df20= df19[(df19['MostSevereInjury'] == 'K')|(df19['MostSevereInjury'] == 'C')|\
         (df19['MostSevereInjury'] == 'B') | (df19['MostSevereInjury'] == 'A')]  

df21 = pd.read_csv("DUI_Crashes.csv")
df22= df21[(df21['MostSevereInjury'] == 'K')|(df21['MostSevereInjury'] == 'C')|\
         (df21['MostSevereInjury'] == 'B') | (df21['MostSevereInjury'] == 'A')]  

df23 = pd.read_csv("Intersection_Crashes.csv")
df24= df23[(df23['MostSevereInjury'] == 'K')|(df23['MostSevereInjury'] == 'C')|\
         (df23['MostSevereInjury'] == 'B') | (df23['MostSevereInjury'] == 'A')]  

df25 = pd.read_csv("Railroad_Crashes.csv")
df26= df25[(df25['MostSevereInjury'] == 'K')|(df25['MostSevereInjury'] == 'A')|\
         (df25['MostSevereInjury'] == 'B')]  

df27 = pd.read_csv("Motorcycle_Crashes.csv")
df28= df27[(df27['MostSevereInjury'] == 'K')|(df27['MostSevereInjury'] == 'A')|\
         (df27['MostSevereInjury'] == 'B')]  
    
df29 = pd.read_csv("Work_Zone_Crashes.csv")
df30= df29[(df29['MostSevereInjury'] == 'K')|(df29['MostSevereInjury'] == 'A')|\
         (df29['MostSevereInjury'] == 'B')]  

df31 = pd.read_csv("Pedestrian_Crashes.csv")
df32= df31[(df31['MostSevereInjury'] == 'K')|(df31['MostSevereInjury'] == 'A')|\
         (df31['MostSevereInjury'] == 'B')]  
    
df33 = pd.read_csv("Speeding_Crashes.csv")
df34= df33[(df33['MostSevereInjury'] == 'K')|(df33['MostSevereInjury'] == 'A')|\
         (df33['MostSevereInjury'] == 'B')]  
   
df35 = pd.read_csv("ATV_Crashes.csv")
df36= df35[(df35['MostSevereInjury'] == 'K')|(df35['MostSevereInjury'] == 'A')|\
         (df35['MostSevereInjury'] == 'B')]    

df37 = pd.read_csv("Fixed_Object_Crashes.csv")
df38= df37[(df37['MostSevereInjury'] == 'K')|(df37['MostSevereInjury'] == 'A')|\
         (df37['MostSevereInjury'] == 'B')]        
    
df39 = pd.read_csv("Non_Motorist.csv")
df40= df39[(df39['MostSevereInjury'] == 'K')|(df39['MostSevereInjury'] == 'A')|\
         (df39['MostSevereInjury'] == 'B')]     
    
df41 = pd.read_csv("Transit_Bus_Crashes.csv")
df42= df41[(df41['MostSevereInjury'] == 'K')|(df41['MostSevereInjury'] == 'A')|\
         (df41['MostSevereInjury'] == 'B')]     

df43 = pd.read_csv("Driver_And_Pedestrian_65_And_Over.csv")
df44= df43[(df43['MostSevereInjury'] == 'K')|(df43['MostSevereInjury'] == 'A')|\
         (df43['MostSevereInjury'] == 'B')] 

df45 = pd.read_csv("Bike_Crashes.csv")
df46= df45[(df45['MostSevereInjury'] == 'K')|(df45['MostSevereInjury'] == 'A')|\
         (df45['MostSevereInjury'] == 'B')] 
  
    
crash = pd.concat([df1,df2,df4,df6,df8,df10,df12,df14,df16,df18,df20,df22,df24,df26,df28,\
                   df30,df32,df34,df36,df38,df40,df42,df44,df46])

crash.shape
crash.dtypes

crash = crash.drop_duplicates()
crash = crash.reset_index(drop=True)
crash = crash.sort_values(by = 'CrashID')

class_count_0, class_count_1,class_count_2,class_count_3,class_count_4 = crash['MostSevereInjury'].value_counts(ascending=True)
print(class_count_0)
print(class_count_1)
print(class_count_2)
print(class_count_3)
print(class_count_4)

# show the resultant Dataframe

crash.shape
#delete the columns with more than 50% null values and those columns that are 
#irrelevant
crash = crash.drop(columns=['NameOfIntersectingRoadway','RouteClass',\
                            'RouteClassDesc','DriverPedestrian','LawEnforcementAgencyName'])
 
ctabres=pd.crosstab(index=crash['LightConditionDesc'],columns=crash['LightCondition'])
print(ctabres)
 
# importing the required function
from scipy.stats import chi2_contingency
 
# Performing Chi-sq test
ChiSqResult = chi2_contingency(ctabres)
 
print('The P-Value of the ChiSq Test is:', ChiSqResult[1])
#Unnecessary columns
crash = crash.drop(columns=['CrashID','CrashDate','X','Y','NameOfRoadway','DayofWeekNumeric',\
                            'CrashSpecificLocation','CrashSeverity','WeatherCondition1',\
                              'FirstHarmfulEvent','SchoolBusRelated','MostSevereInjury',\
                                'MannerOfCrashCollisionImpact','TrafficwayClassType',\
                                    'TypeOfIntersection','LightCondition',\
                                        'TrafficSurfaceCondition','TrafficwayDescription',\
                                            'TrafficwayDescriptionText',\
                                            'IsCrashRelatedToAWorkZone'])
    

    
print(crash.shape)

print(crash.isnull().sum())

crash = crash.dropna()

print(crash['CrashTownName'].unique())
print(crash['CrashTownName'].nunique())

crash['CrashTownName'] = crash['CrashTownName'].replace \
    (['EastHaven', 'WestHartford','NorthHaven','EastHartford',\
      'NewLondon','NewHaven','EastWindsor','NewBritain','NewCanaan',\
          'RockyHill','SouthWindsor','NewFairfield','NorthCanaan',\
              'NorthBranford','WindsorLocks','OldSaybrook','OldLyme',\
                  'EastHampton','NorthStonington','NewHartford','NewMilford',\
                      'EastLyme','EastHaddam','BeaconFalls','EastGranby',\
                          'WestHaven','DeepRiver'],\
     ['East Haven','West Hartford','North Haven','East Hartford',\
      'New London','New Haven','East Windsor','New Britain','New Canaan',\
          'Rocky Hill','South Windsor','New Fairfield','North Canaan',\
              'North Branford','Windsor Locks','Old Saybrook','Old Lyme',\
                'East Hampton','North Stonington','New Hartford', 'New Milford',\
                    'East Lyme','East Haddam','Beacon Falls','East Granby',\
                        'West Haven','Deep River']
         )
 


crash.drop(crash[(crash['MostSevereInjuryDesc'] == 'No Apparent Injury (O)') &\
                 ((crash['MannerCollisionImpactDesc'] == 'Unknown')|\
                     (crash['LightConditionDesc'] == 'Unknown')|\
                         (crash['TrafficSurfaceConditionDesc'] == 'Unknown')|\
                             (crash['FirstHarmfulEventDesc'] == 'Unknown')|\
                                 (crash['TypeOfIntersectionDesc'] == 'Unknown')|\
                                     (crash['SchoolBusRelatedDesc'] == 'Unknown')|\
                                         (crash['CrashSpecificLocationDesc'] == 'Unknown')|\
                                             (crash['TrafficwayClassTypeDesc'] == 'Unknown')|\
                                                 (crash['IsCrashRelatedToAWorkZoneDesc'] == 'Unknown'))].index, inplace=True)

crash.drop(crash[(crash['MostSevereInjuryDesc'] == 'No Apparent Injury (O)') &\
                 ((crash['MannerCollisionImpactDesc'] == 'Not Applicable')|\
                     (crash['LightConditionDesc'] == 'Not Applicable')|\
                         (crash['TrafficSurfaceConditionDesc'] == 'Not Applicable')|\
                             (crash['FirstHarmfulEventDesc'] == 'Not Applicable')|\
                                 (crash['TypeOfIntersectionDesc'] == 'Not Applicable')|\
                                     (crash['SchoolBusRelatedDesc'] == 'Not Applicable')|\
                                         (crash['CrashSpecificLocationDesc'] == 'Not Applicable')|\
                                             (crash['TrafficwayClassTypeDesc'] == 'Not Applicable')|\
                                                 (crash['IsCrashRelatedToAWorkZoneDesc'] == 'Not Applicable'))].index, inplace=True)
    
class_count_0, class_count_1,class_count_2,class_count_3,class_count_4 = crash['MostSevereInjuryDesc'].value_counts(ascending=True)
print(class_count_0)
print(class_count_1)
print(class_count_2)
print(class_count_3)
print(class_count_4)

crash.shape
crash.to_csv('accidentrecord.csv')

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
crash['CrashTimeHour_cat'] = le.fit_transform(crash['CrashTimeHour'])
crash['CrashDateMonth_cat'] = le.fit_transform(crash['CrashDateMonth'])
crash['CrashDateYear_cat'] = le.fit_transform(crash['CrashDateYear'])
crash['DayofWeek_cat'] = le.fit_transform(crash['DayofWeek'])
crash['CrashTownName_cat'] = le.fit_transform(crash['CrashTownName'])
crash['CrashSpecificLocationDesc_cat'] = le.fit_transform(crash['CrashSpecificLocationDesc'])
crash['MostSevereInjuryDesc_cat'] = le.fit_transform(crash['MostSevereInjuryDesc'])
crash['WeatherConditionDesc_cat'] = le.fit_transform(crash['WeatherConditionDesc'])
crash['FirstHarmfulEventDesc_cat'] = le.fit_transform(crash['FirstHarmfulEventDesc'])
crash['SchoolBusRelatedDesc_cat'] = le.fit_transform(crash['SchoolBusRelatedDesc'])
crash['MannerCollisionImpactDesc_cat'] = le.fit_transform(crash['MannerCollisionImpactDesc'])
crash['TrafficwayClassTypeDesc_cat'] = le.fit_transform(crash['TrafficwayClassTypeDesc'])
crash['TypeOfIntersectionDesc_cat'] = le.fit_transform(crash['TypeOfIntersectionDesc'])
crash['LightConditionDesc_cat'] = le.fit_transform(crash['LightConditionDesc'])
crash['TrafficSurfaceConditionDesc_cat'] = le.fit_transform(crash['TrafficSurfaceConditionDesc'])
crash['IsCrashRelatedToAWorkZoneDesc_cat'] = le.fit_transform(crash['IsCrashRelatedToAWorkZoneDesc'])

crash.head(10)

from sklearn.preprocessing import LabelEncoder
# creating instance of labelencoder
labelencoder = LabelEncoder()
# Assigning numerical values and storing in another column
crash['CrashSeverityDesc_Cat'] = labelencoder.fit_transform(crash['CrashSeverityDesc'])

crash=crash.drop(columns =['CrashTimeHour','CrashDateMonth','CrashDateYear','DayofWeek',\
        'CrashTownName','CrashSpecificLocationDesc','MostSevereInjuryDesc',\
          'CrashSeverityDesc','WeatherConditionDesc','FirstHarmfulEventDesc',\
              'SchoolBusRelatedDesc','MannerCollisionImpactDesc',\
                  'TrafficwayClassTypeDesc','TypeOfIntersectionDesc','LightConditionDesc',\
                      'TrafficSurfaceConditionDesc','IsCrashRelatedToAWorkZoneDesc' ])

print(crash.isnull().sum())

cor=crash.corr()
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(20,15))
sns.heatmap(cor,annot=True)

crash=crash.drop(columns =['MostSevereInjuryDesc_cat'])

print(crash.isnull().sum())

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

X1 = crash.drop(columns = ['CrashSeverityDesc_Cat'])

y1 = crash['CrashSeverityDesc_Cat']

chi_scores = chi2(X1,y1)
chi_scores

selector = SelectKBest(chi2, k=10)
new_data = selector.fit(X1,y1)

# Get columns to keep and create new dataframe with those only
cols = selector.get_support(indices=True)
print(cols)
features_df_new = X1.iloc[:,cols]

new_data.scores_

chi_values = pd.Series(chi_scores[0],index = X1.columns)
chi_values.sort_values(ascending = False , inplace = True)
chi_values.plot.bar()

print(crash.isnull().sum())

cat_cols = ['CrashTimeHour_cat','CrashDateYear_cat','CrashDateMonth_cat',\
        'CrashSpecificLocationDesc_cat','CrashTownName_cat',\
          'FirstHarmfulEventDesc_cat',\
              'MannerCollisionImpactDesc_cat',\
                  'TypeOfIntersectionDesc_cat','LightConditionDesc_cat',\
                      'TrafficSurfaceConditionDesc_cat','WeatherConditionDesc_cat',\
                          'SchoolBusRelatedDesc_cat','TrafficwayClassTypeDesc_cat',\
                              'IsCrashRelatedToAWorkZoneDesc_cat']
    
cat_cols_encoded = []
for col in cat_cols:
  cat_cols_encoded += [f"{col[:]}_{cat}" for cat in list(crash[col].unique())]

cat_cols_encoded

from sklearn.preprocessing import OneHotEncoder
onehot_encoder = OneHotEncoder(sparse=False,handle_unknown='ignore')
encoded_cols = onehot_encoder.fit_transform(crash[cat_cols])
df_enc = pd.DataFrame(encoded_cols,columns = cat_cols_encoded )
df_enc.shape
crash.shape

print(crash.isnull().sum())

crash = crash.drop(columns=['CrashTimeHour_cat','CrashDateMonth_cat','CrashDateYear_cat','DayofWeek_cat',\
        'CrashTownName_cat','CrashSpecificLocationDesc_cat',\
          'FirstHarmfulEventDesc_cat',\
              'SchoolBusRelatedDesc_cat','MannerCollisionImpactDesc_cat',\
                  'TrafficwayClassTypeDesc_cat','TypeOfIntersectionDesc_cat','LightConditionDesc_cat',\
                      'TrafficSurfaceConditionDesc_cat','IsCrashRelatedToAWorkZoneDesc_cat',\
                          'WeatherConditionDesc_cat'])
    
print(crash.isnull().sum())
crash.shape

crash = pd.concat([crash,df_enc],axis=1, join='inner',ignore_index=False)

print(crash.isnull().sum())

from sklearn.model_selection import train_test_split

X = crash.drop(columns = ['CrashSeverityDesc_Cat'])
y = crash['CrashSeverityDesc_Cat']

import pandas as pd
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_data = scaler.fit_transform(X)
dataset =pd.DataFrame(scaled_data,columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(
             dataset, y, test_size = 0.25,random_state=0)

from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.metrics import classification_report_imbalanced
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier


from collections import Counter
counter1 = Counter(y_train)
print(counter1)

strategy = {0:119958,1:129794,2:119958}
over = SMOTE(sampling_strategy=strategy)

strategy1 = {0:119958,1:119958,2:119958}
under = RandomUnderSampler(sampling_strategy=strategy1)

X_resampled, y_resampled = over.fit_resample(X_train, y_train)
X_resampled1, y_resampled1 = under.fit_resample(X_resampled, y_resampled)

counter = Counter(y_resampled)
print(counter)
counter2 = Counter(y_resampled1)
print(counter2)

# Create a pipeline
model = RandomForestClassifier(n_estimators=100,class_weight="balanced") 
model = DecisionTreeClassifier(class_weight="balanced")
model = XGBClassifier()

model.fit(X_resampled1,y_resampled1)
model.fit(X_train,y_train)

print(classification_report_imbalanced(y_test, model.predict(X_test)))

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,  model.predict(X_test))
