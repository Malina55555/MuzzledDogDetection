import os
import cv2
import json
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file, send_from_directory, url_for
from werkzeug.utils import secure_filename
from model import MuzzleDetectorModel

# Конфигурация
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Создаем папки, если их нет
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# Инициализация модели
detector = MuzzleDetectorModel()


def allowed_file(filename):
    # проверка на расширение
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    #===Главная страница===
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    # подгрузка файла
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        # Генерируем уникальное имя файла
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        original_filename = secure_filename(file.filename)
        filename = f"original_{timestamp}_{original_filename}"
        original_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # Сохраняем файл
        file.save(original_path)
        #print(f"Файл сохранен: {original_path}")  # debug
        # процессинг
        return process_image(original_path, original_filename)

    return jsonify({'error': 'File type not allowed'}), 400


def process_image(image_path, original_filename):
    # Обработка изображения
    #print(f"Обработка изображения: {image_path}")  # debug

    # Получаем предсказания от модели
    detections, processed_image = detector.predict(image_path)

    if processed_image is None:
        return jsonify({'error': 'Failed to process image'}), 500

    # Сохраняем обработанное изображение
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    processed_filename = f"processed_{timestamp}_{original_filename}"
    processed_path = os.path.join(app.config['UPLOAD_FOLDER'], processed_filename)

    # Сохраняем изображение
    success = cv2.imwrite(processed_path, processed_image)
    if not success:
        return jsonify({'error': 'Failed to save processed image'}), 500

    #print(f"Обработанный файл сохранен: {processed_path}")  # debug

    # Сохраняем в историю
    record = detector.save_to_history(original_filename, detections, processed_filename)

    # Подготовка ответа
    result = {
        'success': True,
        'original_filename': original_filename,
        'processed_filename': processed_filename,
        'original_url': url_for('uploaded_file', filename=os.path.basename(image_path)),
        'processed_url': url_for('uploaded_file', filename=processed_filename),
        'detections': detections,
        'stats': record['stats']
    }

    #print(f"Результат: {result}")  # debug
    return jsonify(result)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    # Подгружаем файл из uploads на страницу
    try:
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    except FileNotFoundError:
        # Если файл не найден, возвращаем placeholder
        return send_from_directory('static', 'placeholder.jpg')


@app.route('/history')
def get_history():
    # история обработки
    history = detector.get_history()

    # Добавляем полные URL к каждому элементу истории
    for record in history:
        if 'processed_image' in record:
            record['processed_url'] = url_for('uploaded_file', filename=record['processed_image'])

    return jsonify(history)


@app.route('/report')
def generate_report():
    # Генерирует и отдает отчет
    report_path = detector.generate_pdf_report()
    if report_path and os.path.exists(report_path):
        return send_file(report_path, as_attachment=True)
    return jsonify({'error': 'Failed to generate report'}), 500


@app.route('/clear_history', methods=['POST'])
def clear_history():
    try:
        # Получаем список всех файлов в папке
        for filename in os.listdir(UPLOAD_FOLDER):
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Ошибка при удалении файла {file_path}: {e}")

        with open('history.json', 'w', encoding='utf-8') as f:
            json.dump([], f)

        return jsonify({'success': True, 'message': 'История и файлы очищены'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == "__main__":
    # Создаем placeholder изображение, если его нет
    placeholder_path = "static/placeholder.jpg"
    if not os.path.exists(placeholder_path):
        import numpy as np

        placeholder_img = np.zeros((300, 300, 3), dtype=np.uint8)
        cv2.putText(placeholder_img, 'No image', (80, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imwrite(placeholder_path, placeholder_img)

    app.run(debug=True, host='0.0.0.0', port=5000)
