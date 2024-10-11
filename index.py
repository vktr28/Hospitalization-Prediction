import pandas as pd
import numpy as np
import matplotlib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import warnings
warnings.simplefilter("ignore")
pd.set_option('display.max_columns', None)
data = pd.read_csv("Hospitalization.csv")
df = pd.DataFrame(data)
lab = LabelBinarizer().fit(df["Gender"])
transformed = lab.transform(df["Gender"]) # M - 1, F - 0

df["Gender_Binary"] = transformed

deseases = np.char.lower(np.unique([x for x in df["Deseases"]])) # Getting all the unique elements from the 'Deceases' column

deseases_ordered = pd.get_dummies(df["Deseases"], dtype=int) # Placing each decease in separate columns

le = LabelBinarizer().fit(df["Hospitalized_Recently"])
hospitalized = pd.DataFrame(le.transform(df["Hospitalized_Recently"])) # Converting Yes/No into 1/0
df["Hospitalized_binary"] = hospitalized
df = df.assign(**deseases_ordered)
df = df.drop(columns=["Deseases", "Hospitalized_Recently", "Patient_id", "Gender"])

Y = df["Hospitalized_binary"]
X = df.drop(columns=["Hospitalized_binary"])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=473) ### Model shows the best correlation coefficient - 0.88 which is very good
regression = LogisticRegression().fit(X_train,Y_train)
while True:
    try:
       age = float(input("How old are you?: "))

       gender = input("What's your gender? (M/F): ")
       function = lambda x: 1 if x.lower() == "m" or x.lower() == "male" or x.lower() == "man" else 0

       bin_gender = function(gender)
       comorbidities = int(input("How many other deseases do you have?: "))
       days = int(input("How long have you been in hospital before? (days): "))
       apps = int(input("How many appointments have you had ?: "))
       treatment = int(input("How much treatment have you been prescribed lately?: "))
       desease = input("What's your desease?: ")
       des_num = np.zeros(len(deseases), dtype=int)
       index = np.where(desease.lower() == deseases)[0][0]
       des_num[index] = 1
       des_num = des_num.reshape(1,-1)

       to_predict1 = np.array([age, comorbidities, days, apps, treatment, bin_gender]).reshape(1,-1)
       to_predict = np.concatenate((to_predict1, des_num), axis=1)
       prediction = regression.predict(to_predict)
       print("Yes, you are highly likely to be hospitalized" if prediction == 1 else "No, you are unlikely to be hospitalized")
    except:
        print("Something has gone wrong! Check you data and try again!")
    e = input("\nPass enter if you want to continue or other key if else: ")
    if e != "":
       break

