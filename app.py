from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    val1 = request.form.get('bedrooms')
    val2 = request.form.get('bathrooms')
    val3 = request.form.get('floors')
    val4 = request.form.get('yr_built')

    if not val1 or not val2 or not val3 or not val4:
        error_message = "Please enter the complete details."
        return render_template('index.html', error=error_message)

    arr = np.array([val1, val2, val3, val4])
    arr = arr.astype(np.float64)
    pred = model.predict([arr])

    return render_template('index.html', data=int(pred))


if __name__ == '__main__':
    app.run(debug=True)
