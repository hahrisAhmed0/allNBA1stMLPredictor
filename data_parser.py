import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


#****READ FROM CSV FILE
data = pd.read_csv('all_seasons.csv')
data2 = pd.read_csv('All-NBA_Team_4.csv')
#print(data2.head)

#****CLEAN RESPECTIVE DATA

del data[data.columns[0]]
for i in range(5):
   del data2[data2.columns.values[-1]]
data2 = data2.iloc[41:]


def clean(x):
  x = x.replace("*", "").replace("(", "").replace(")","").replace("^", "").replace("†","").replace("ć","c").replace("č","c").replace("§","").replace("–","-")
  for i in "0123456789":
    x=x.replace(i,"")
  return str(x)

def clean2(x):
    x = x.replace("*", "").replace("(", "").replace(")", "").replace("^", "").replace("†","").replace("ć","c").replace("č","c").replace("§","").replace("–","-")
    return(str(x))

data2['First team'] = data2['First team'].apply(clean)


#****COMBINE ALL NBA TEAM INTO DICT + FURTHER CLEANING

allNBADict = data2.groupby(['Season'])['First team'].apply(list).to_dict()


for key,val in allNBADict.items():
  newT = [ x if x[-1].isalpha() else x[:-1]  for x in val]
  allNBADict[key] = newT


tempL = list(allNBADict.keys())
for key in tempL:
  new_key = clean2(key)
  allNBADict[new_key] = allNBADict.pop(key)

allNBADict["2006-07"] = ['Dirk Nowitzki', 'Tim Duncan', 'Amar\'e Stoudemire', 'Steve Nash', 'Kobe Bryant']

#******SPLITTING DATA INTO TRAIN,VAL,TEST SETS
train = data.loc[(data["season"] >= "1996") & (data["season"] < "2012"), ["player_name", "gp", "pts", "reb", "ast", "net_rating", "oreb_pct", "dreb_pct", "usg_pct", "ts_pct", "ast_pct", "season"]]

y_train = []
temp = []
for index,row in train.iterrows():
  year = row["season"]
  playerName = row["player_name"]
  if playerName in allNBADict[year]:
    y_train.append(1)
    temp.append((playerName, year))
  else:
    y_train.append(0)

val = data.loc[(data["season"] >= "2012") & (data["season"] < "2015"), ["player_name", "gp", "pts", "reb", "ast", "net_rating", "oreb_pct", "dreb_pct", "usg_pct", "ts_pct", "ast_pct", "season"]]
y_val = []
for index,row in val.iterrows():
  year = row["season"]
  
  playerName = row["player_name"]
  if playerName in allNBADict[year]:
    y_val.append(1)
  else:
    y_val.append(0)



test = data.loc[(data["season"] >= "2015"), ["player_name", "gp", "pts", "reb", "ast", "net_rating", "oreb_pct","dreb_pct", "usg_pct", "ts_pct", "ast_pct", "season"]]
y_test = []
for index,row in test.iterrows():
  year = row["season"]
  playerName = row["player_name"]
  if playerName in allNBADict[year]:
    y_test.append(1)
  else:
    y_test.append(0)


del train["player_name"]
del train["season"]
del val["player_name"]
del val["season"]
del test["player_name"]
del test["season"]

#******MODEL TREE
model = DecisionTreeClassifier(max_depth=5)
model.fit(train, y_train)
y_pred_val = model.predict(val)
print("acc_val_tree:", accuracy_score(y_val, y_pred_val))

y_pred_test = model.predict(test)
print("acc_test_tree", accuracy_score(y_test, y_pred_test))

#******MODEL FOREST
model2 = RandomForestClassifier(n_estimators=250, max_depth=5, bootstrap=True)
model2.fit(train, y_train)

y_pred_val2 = model2.predict(val)
print("acc_val_forest:", accuracy_score(y_val, y_pred_val2))

y_pred_test2 = model2.predict(test)
print("acc_val_forest:", accuracy_score(y_test, y_pred_test2))