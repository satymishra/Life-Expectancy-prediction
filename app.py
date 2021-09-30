#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 22:49:12 2021

@author: satyammishra
"""

# importing modules
import numpy as np
from flask import Flask,request,jsonify,url_for,render_template,make_response
import pickle


app=Flask(__name__,template_folder='templates')
model=pickle.load(open('model.pkl','rb'))


@app.route("/")
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on html GUI
    '''
    int_features =[float(x) for x in request.form.values()]
    final_features=[np.array(int_features)]
    prediction=model.predict(final_features)
    output = round(prediction[0],2)
    
    return render_template('index.html', prediction_text=f'Life expectancy in your region is ===>  {output} years')

if __name__=="__main__":
    app.run(debug=True)


