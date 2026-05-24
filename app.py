from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

#load the dataset:
df = pd.read_csv("Rockfall_Risk_Data.csv")

#create graph Automatically:
def create_plot():
    plt.figure()

df['rockfall_risk'].value_counts().plot(kind='bar')
plt.title('Distribution of Rockfall Risk Levels')

if not os.path.exists('static'):
    os.makedirs('static')

plt.savefig("static/plot.png")
plt.close()

 


app = Flask(__name__)

# Load the trained model
model = pickle.load(open("model.pkl","rb"))

risk_map = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}

@app.route('/')
def home():
    create_plot()
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():  
    values = [float(x) for x in request.form.values()]

    final_input = np.array(values).reshape(1, -1)

    prediction = model.predict(final_input)

    result = risk_map[prediction[0]]

    return render_template('index.html', prediction_text='The predicted risk level is: {}'.format(result))

if __name__ == "__main__":
    app.run(debug=True)        