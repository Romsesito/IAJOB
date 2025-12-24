from flask import Flask, render_template, request
import pandas as pd
import joblib
import tensorflow as tf
import numpy as np

app = Flask(__name__)

model = tf.keras.models.load_model('modelo_sueldos.keras')
preprocessor = joblib.load('preprocesador.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    
    data = {
        'years_experience': [float(request.form['years_experience'])],
        'remote_ratio': [float(request.form['remote_ratio'])],
        'benefits_score': [float(request.form['benefits_score'])],
        'experience_level': [request.form['experience_level']],
        'employment_type': [request.form['employment_type']],
        'company_size': [request.form['company_size']],
        'education_required': [request.form['education_required']],
        'company_location': [request.form['company_location']],
        'industry': [request.form['industry']]
    }
    
    df_input = pd.DataFrame(data)
    input_processed = preprocessor.transform(df_input).toarray()
    prediction = model.predict(input_processed)
    resultado = round(float(prediction[0][0]), 2)
    
    return render_template('index.html', prediction_text=f'El sueldo estimado es: ${resultado:,.2f} USD')

if __name__ == "__main__":
    app.run(debug=True)