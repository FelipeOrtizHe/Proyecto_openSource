from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
from model import load_model, classify_text
import hvplot.pandas  # Importar hvplot para gráficos
import holoviews as hv

app = Flask(__name__)

# Variables globales para almacenar datos y resultados
new_data = None
results = []

# Cargar el modelo al inicio de la aplicación
classifier = load_model()

@app.route('/')
def index():
    global new_data, results
    # Reiniciar los datos al cargar la página principal
    new_data = None
    results = []
    return render_template('index.html', new_data=new_data, results=results)

@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    global new_data, results
    if 'csvFile' not in request.files:
        return redirect(url_for('index'))
    
    file = request.files['csvFile']
    if file.filename == '':
        return redirect(url_for('index'))
    
    if file:
        df = pd.read_csv(file)
        new_data = df.iloc[0, 0]  # Ajustar esto según la estructura del CSV
        results = classify_text(classifier, new_data)
        
        # Generar el gráfico de barras
        plot = results.hvplot.bar(x='Emociones', y='Porcentaje', title='Predicciones', width=500, height=400).opts(tools=['hover'])

        # Convertir el gráfico a HTML y pasar a la plantilla
        plot_html = hv.render(plot, backend='bokeh', holomap='auto', dpi=72)
        return render_template('index.html', new_data=new_data, results=results, plot=plot_html)
    
    return redirect(url_for('index'))

@app.route('/submit_tweet', methods=['POST'])
def submit_tweet():
    global new_data, results
    tweet_text = request.form['tweetText']
    new_data = tweet_text
    results = classify_text(classifier, new_data)
    
    # Generar el gráfico de barras
    plot = results.hvplot.bar(x='Emociones', y='Porcentaje', title='Predicciones', width=500, height=400).opts(tools=['hover'])

    # Convertir el gráfico a HTML y pasar a la plantilla
    plot_html = hv.render(plot, backend='bokeh', holomap='auto', dpi=72)
    return render_template('index.html', new_data=new_data, results=results, plot=plot_html)

# Rutas para otras páginas
@app.route('/tutorial.html')
def tutorial():
    return render_template('tutorial.html')

@app.route('/classificator.html')
def classificator():
    return render_template('classificator.html')

@app.route('/contact.html')
def contact():
    return render_template('contact.html')

@app.route('/history.html')
def history():
    return render_template('history.html')

if __name__ == "__main__":
    app.run(debug=True)
