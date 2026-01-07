from flask import Flask, render_template, request
import tensorflow as tf
import joblib
import pandas as pd
import os

app = Flask(__name__)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


model = tf.keras.models.load_model(os.path.join(BASE_DIR, 'modelo_sueldos.keras'))
preprocessor = joblib.load(os.path.join(BASE_DIR, 'preprocesador.joblib'))

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
    

    df = pd.DataFrame(data)
    

    X_processed = preprocessor.transform(df).toarray()
    prediction = model.predict(X_processed)
    

    salary = prediction[0][0]
    output = f'Salario Estimado: ${salary:,.2f} USD/año'
    
    return render_template('index.html', prediction_text=output)

if __name__ == '__main__':
    app.run(debug=True)