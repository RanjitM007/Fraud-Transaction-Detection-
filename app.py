
from flask import Flask, render_template, request
import pickle
import numpy as np
import joblib


model=joblib.load(open('Fraud_txn_final.pkl','rb'))

app=Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def get_pred():
    data1=request.form['data1']
    data2=request.form['data2']
    data3=request.form['data3']
    data4=request.form['data4']
    data=np.array([[data1,data2,data3,data4]])
    prediction=model.predict(data)
    return render_template('result.html',data=prediction)
    
            
if __name__=='__main__':
    app.run(debug=True)