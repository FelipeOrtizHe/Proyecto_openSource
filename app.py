from flask import Flask, render_template, request, send_file, jsonify
import pandas as pd
from model import initialize_classifier, plot_emotion_scores
from data import detect_text_column
import os
import zipfile

app = Flask(__name__)

# Inicializamos el pipeline de clasificación de texto
classifier = initialize_classifier()

# Ruta para la página principal
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'csvFile' in request.files:
            csv_file = request.files['csvFile']
            if csv_file.filename != '':
                # Guardar el archivo CSV subido en el servidor
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], csv_file.filename)
                csv_file.save(file_path)

                # Leer el archivo CSV especificando la codificación
                data = pd.read_csv(file_path, encoding='utf-8')

                # Detectar la columna con las frases
                column_name = detect_text_column(data)
                if not column_name:
                    return jsonify({'success': False})

                # Crear una carpeta para guardar las gráficas
                output_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'calificacion')
                os.makedirs(output_dir, exist_ok=True)

                # Iterar sobre las frases en la columna detectada
                for index, row in data.iterrows():
                    user_input = row[column_name]

                    # Clasificar el texto
                    prediction = classifier(user_input)

                    # Generar un nombre de archivo único para cada gráfica
                    filename = os.path.join(output_dir, f"calificacion_{index}.png")
                    plot_emotion_scores(prediction, filename, title=user_input)

                # Generar un archivo ZIP con las imágenes generadas
                zip_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'calificacion.zip')
                with zipfile.ZipFile(zip_filename, 'w') as zip_file:
                    for root, dirs, files in os.walk(output_dir):
                        for file in files:
                            zip_file.write(os.path.join(root, file), file)

                # Devolver la URL para descargar el archivo ZIP
                zip_link = f"/download/{os.path.basename(zip_filename)}"
                return jsonify({'success': True, 'zip_link': zip_link, 'zip_filename': os.path.basename(zip_filename)})

    # Renderizar la plantilla HTML con el formulario de carga
    return render_template('index.html')

# Ruta para descargar archivos ZIP
@app.route('/download/<filename>')
def download(filename):
    zip_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    return send_file(zip_path, mimetype='application/zip', as_attachment=True, download_name=filename)



@app.route('/tutorial')
def tutorial():
    return render_template('tutorial.html')

@app.route('/classificator')
def classificator():
    return render_template('classificator.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/history')
def history():
    return render_template('history.html')

if __name__ == "__main__":
    app.config['UPLOAD_FOLDER'] = 'uploads'
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
