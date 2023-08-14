import pandas as pd
import numpy as np
import sklearn.model_selection as ms
from sklearn.linear_model import LinearRegression
import sklearn.linear_model as lm
import matplotlib.pyplot as plt
from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='No file part')

        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error='No selected file')

        dados = pd.read_excel(file)
        dados = dados.dropna()
        dados1 = dados[["Unnamed: 2", "Unnamed: 6"]]
        dados1['Unnamed: 2'] = pd.to_numeric(dados1['Unnamed: 2'], errors='coerce')
        df = dados1.drop(2)
        X = df.iloc[:, :-1]
        Y = df.iloc[:, -1]
        X_train, X_test, Y_train, Y_test = ms.train_test_split(X, Y, test_size=1/5, random_state=0)

        regressor = lm.LinearRegression()
        regressor.fit(X_train, Y_train)

        y_pred = regressor.predict(X_test)

        plt.scatter(X_train, Y_train, color='red')
        plt.plot(X_train, regressor.predict(X_train), color='blue')
        plt.title('Treino')
        plt.savefig('static/train_plot.png')
        plt.close()

        plt.scatter(X_test, Y_test, color='red')
        plt.plot(X_train, regressor.predict(X_train), color='blue')
        plt.title('Teste')
        plt.savefig('static/test_plot.png')
        plt.close()

        specific_value = regressor.predict([[10]])[0]

        return render_template('index.html', train_plot='static/train_plot.png', test_plot='static/test_plot.png', specific_value=specific_value)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
