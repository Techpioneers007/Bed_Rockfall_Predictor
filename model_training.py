import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier   
from sklearn.metrics import accuracy_score  
import matplotlib.pyplot as plt
import pickle


df = pd.read_csv("Rockfall_Risk_Data.csv")
print(df.head())
print(df['rockfall_risk'].unique())

df['rockfall_risk'] = df['rockfall_risk'].map({'Low': 0, 'Medium': 1, 'High': 2}) #Convert target into numbers
df=df.drop("Unnamed: 0", axis = 1) #remove useless column
drop_cols = ['rockfall_density', 'rockfall_volume'] #drop target and other columns that are not useful for prediction
X = df.drop(drop_cols + ['rockfall_risk'], axis=1) #Split data into features and target
y = df['rockfall_risk']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) #Trained Model

model = RandomForestClassifier()
model.fit(X_train,y_train)

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

#Check Accuracy
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

#-----Prediction for new data point-----
print("Columns order:", list(X.columns))

Sample_data = [[
    50,  # slope_angle
    40,  # rock_strength
    100, # rainfall_mm
    30,  # seismic_activity
    15,  # temperature_variation
    25,  # soil_moisture
    60,  # vegetation_cover
    200, # mining_depth
    20,  # blasting_frequency
    60,  # weathering_index
    70,  # human_activity_index
    80,  # rainfall_intensity
    65   # slope_stability_index
]]

print('Running prediction...')
sample_df = pd.DataFrame(Sample_data, columns=X.columns)
prediction = model.predict(sample_df)

risk_map = {0: 'Low', 1: 'Medium', 2: 'High'}
print("Predicted Rockfall Risk:", risk_map[prediction[0]])

# Importance of features
importances = model.feature_importances_
feature_names = X.columns

plt.figure(figsize=(8, 6))
plt.barh(feature_names, importances)
plt.xlabel("Feature Importance")
plt.title("Feature Importance in Rockfall Risk Prediction")
plt.tight_layout()
plt.savefig("feature_importance.png")
print("Feature importance chart saved to feature_importance.png")

