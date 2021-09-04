from flask import Flask, request,redirect,url_for
#rom keras.models import load_model
from flask import Flask, render_template
#from data_input import data_in
import numpy as np
import pickle 


app = Flask(__name__,template_folder='templates')

model1= pickle.load(open('F:/aniketproject/heart_model.pkl', 'rb')) 
model= pickle.load(open('F:/aniketproject/RFC_diabetes.pkl', 'rb')) 

    
@app.route('/')
def home():
    return render_template('homepage.html')


@app.route('/heart_page')
def heart_page():
    title='Heart Disease'
    return(render_template("heart_disease.html", title=title))


@app.route('/predict_hdc', methods =['POST'])
def predict_hdc():
    
    # Put all form entries values in a list 
    features = [float(i) for i in request.form.values()]
    # Convert features to array
    array_features = [np.array(features)]
    # Predict features
    prediction = model1.predict(array_features)
    
    output = prediction
    
    # Check the output values and retrive the result with html tag based on the value
    if output == 1:
        return render_template('heart_disease.html', 
                               result = 'You may not likely to have heart disease!')
    else:
        return render_template('heart_disease.html', 
                               result = 'You may likely to have heart disease!')
 
@app.route('/diab_page')
def diab_page():
    title='Diabetes Predcition'
    return(render_template("diabetes.html", title=title))


@app.route('/predict_diab', methods = ['POST'])
def predict_diab():

        preg = int(request.form['pregnancies'])
        glucose = int(request.form['glucose'])
        bp = int(request.form['bloodpressure'])
        st = int(request.form['skinthickness'])
        insulin = int(request.form['insulin'])
        bmi = float(request.form['bmi'])
        dpf = float(request.form['dpf'])
        age = int(request.form['age'])
        
        data = np.array([[preg, glucose, bp, st, insulin, bmi, dpf, age]])
        prediction = model.predict(data)
        
        
        return render_template('result.html', prediction=prediction)
if __name__ == '__main__':
    app.run(debug=True)
