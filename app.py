
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import json
import joblib



app = Flask(__name__)
price_model = pickle.load(open('regressor.pkl','rb'))
scaler = pickle.load(open('scaler.pkl','rb'))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/Prediction')
def prediction():
    return render_template('Prediction.html')

@app.route('/map')
def map():
    return render_template('Map.html')

@app.route('/Analysis')
def prediction():
    return render_template('Analysis.html')    

@app.route('/prediction',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    with open('categorical_vars.json') as f:
        convert_cats = json.load(f)
    
    bedrooms  = float(request.form['Bedrooms'])
    bathrooms  = float(request.form['Bathrooms'])
    carspot  = float(request.form['cspots'])
    landsize  = float(request.form['area'])

    #convert categorical variables
    types_dict = {value:key for key, value in convert_cats['Type'].items()}
    types = int(types_dict[str(request.form['type'])])
    region_dict = {value:key for key, value in convert_cats['Regionname'].items()}
    region = int(region_dict[str(request.form['region'])])
    features = [[bedrooms, bathrooms, carspot, 
                landsize,types, region]]

    # features = [[int(x) for x in request.form.values()]]
    
    scale_features = scaler.transform(features)
    output = round(price_model.predict(scale_features)[0],2)
    print(output)
    
    return render_template('Prediction.html', prediction_text='Your house is worth: ${}'.format(output))


if __name__ == "__main__":
        app.run(debug=False)