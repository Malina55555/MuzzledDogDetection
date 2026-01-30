import cv2
from ultralytics import YOLO
import json
from datetime import datetime
from pathlib import Path


MODEL_PATH = "best_muzzle_model_yolo26m.pt"
HISTORY_FILE = "history.json"
FONT_PATH = "./DejaVuSans.ttf"
CONFIDENCE_THRESHOLD = 0.5
REPORTS_PATH = "./reports"


class MuzzleDetectorModel:

    def __init__(self, model_path=MODEL_PATH, history_file=HISTORY_FILE):

        self.device = 'cpu'
        print(f"Используется устройство: {self.device}")
        # Загружаем модель
        print(f"Загрузка модели из файла весов: {model_path}")
        try:
            self.model = YOLO(model_path)
            print(f"Модель успешно загружена")
            #print(f"Имена классов модели: {self.model.names}") #debug
        except Exception as e:
            print(f"Ошибка загрузки модели: {e}")
            raise

        self.history_file = history_file
        if not Path(self.history_file).exists():
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump([], f)

    def predict(self, image_path, confidence_threshold=CONFIDENCE_THRESHOLD):
        # Загружаем изображение
        img = cv2.imread(image_path)
        if img is None:
            print(f"Не удалось загрузить изображение: {image_path}")
            return [], img
        # инференс
        try:
            results = self.model(image_path, conf=confidence_threshold, device=self.device)
        except Exception as e:
            print(f"Ошибка во время инференса: {e}")
            return [], img

        detections = []

        # Получаем результаты
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy()

            for i, (box, conf, class_id) in enumerate(zip(boxes, confidences, class_ids)):
                # Получаем имя класса
                class_id_int = int(class_id)
                if class_id_int < len(self.model.names):
                    label = self.model.names[class_id_int]
                else:
                    label = f"class_{class_id_int}"

                detections.append({
                    "bbox": box.tolist(),  # [x1, y1, x2, y2]
                    "label": label,
                    "confidence": float(conf),
                    "class_id": class_id_int
                })

            #print(f"Обнаружено объектов: {len(detections)}")  #debug

        # Получаем изображение с аннотациями
        output_img = results[0].plot() if len(results) > 0 else img.copy()

        return detections, output_img

    def save_to_history(self, filename, detections, processed_filename):

        with open(self.history_file, 'r', encoding='utf-8') as f:
            history = json.load(f)

        # статистика по изображению
        stats = {
            "total_dogs": len(detections),
            "with_muzzle": sum(1 for d in detections if d.get("label") == "with_muzzle"),
            "without_muzzle": sum(1 for d in detections if d.get("label") == "without_muzzle"),
        }

        # Создаем новую запись
        record = {
            "timestamp": datetime.now().isoformat(),
            "filename": filename,
            "processed_image": processed_filename,
            "detections": detections,
            "stats": stats
        }

        # Добавляем запись и сохраняем
        history.append(record)
        with open(self.history_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

        #print(f"Результат сохранен в историю")
        return record

    def get_history(self, limit=50):
        try:
            with open(self.history_file, 'r', encoding='utf-8') as f:
                history = json.load(f)
            return history[-limit:]
        except Exception as e:
            print(f"Ошибка загрузки истории: {e}")
            return []

    def generate_pdf_report(self):
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.lib.utils import ImageReader
        import os

        history = self.get_history()
        # if not history:
        #     print("История пустая, отчет не сгенерирован")
        #     return None

        # имя файла отчета
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = f"{REPORTS_PATH}/отчет_детекция_собак_{timestamp}.pdf"

        try:
            # Создаем PDF
            c = canvas.Canvas(report_path, pagesize=letter)
            width, height = letter

            # подгрузка шрифта
            font_name = "RussianFont"
            font_path = FONT_PATH
            if os.path.exists(font_path):
                try:
                    pdfmetrics.registerFont(TTFont(font_name, font_path))
                    #print(f"Зарегистрирован шрифт с кириллицей: {font_path}")
                except Exception as e:
                    print(f"Не удалось зарегистрировать {font_path}: {e}")

            # ====== СОЗДАНИЕ ОТЧЕТА ======

            # Заголовок
            c.setFont(font_name, 16)
            c.drawString(50, height - 50, "ОТЧЕТ ПО ДЕТЕКЦИИ СОБАК БЕЗ НАМОРДНИКОВ")

            # Информация о генерации
            c.setFont(font_name, 10)

            c.drawString(50, height - 80, f"Дата генерации: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            if history:
                c.drawString(50, height - 100, f"Всего записей в истории: {len(history)}")
            else:
                c.drawString(50, height - 100, f"Всего записей в истории: {0}")

            # Общая статистика
            y_position = height - 140
            c.setFont(font_name, 12)
            c.drawString(50, y_position, "ОБЩАЯ СТАТИСТИКА:")

            # Считаем общую статистику
            if history:
                total_dogs = sum(r['stats']['total_dogs'] for r in history)
                total_with = sum(r['stats']['with_muzzle'] for r in history)
                total_without = sum(r['stats']['without_muzzle'] for r in history)
            else:
                total_dogs = 0
                total_with = 0
                total_without = 0

            y_position -= 25
            c.setFont(font_name, 10)
            c.drawString(70, y_position, f"Всего обнаружено собак: {total_dogs}")
            y_position -= 20
            if total_dogs > 0:
                with_percent = total_with / total_dogs * 100
                c.drawString(70, y_position, f"С намордниками: {total_with} ({total_with / total_dogs * 100:.1f}%)")
                # Полоска для "с намордниками"
                c.setFillColorRGB(0, 0.6, 0)  # Зеленый
                c.rect(270, y_position, with_percent * 2, 10, fill=1)
                c.setFillColorRGB(0, 0, 0)  # Возвращаем черный
            else:
                c.drawString(70, y_position, f"С намордниками: 0 (0.0%)")

            y_position -= 20
            if total_dogs > 0:
                without_percent = total_without / total_dogs * 100
                c.drawString(70, y_position,
                             f"Без намордников: {total_without} ({total_without / total_dogs * 100:.1f}%)")
                c.setFillColorRGB(0.9, 0, 0)  # Красный
                c.rect(270, y_position, without_percent * 2, 10, fill=1)
                c.setFillColorRGB(0, 0, 0)  # Возвращаем черный
            else:
                c.drawString(70, y_position, f"Без намордников: 0 (0.0%)")

            # Раздел с изображениями последних записей
            y_position -= 40
            c.setFont(font_name, 12)
            c.drawString(50, y_position, "ИЗОБРАЖЕНИЯ ПОСЛЕДНИХ ОБНАРУЖЕНИЙ:")

            if history:
                recent_history = history[-10:]  # Последние 10 записей

                # Константы для изображений
                IMAGE_WIDTH = 250  # Ширина изображения в PDF
                IMAGE_HEIGHT = 180  # Высота изображения в PDF
                IMAGE_SPACING = 220  # Расстояние между изображениями по вертикали

                # Добавляем изображения перед текстовыми записями
                for i, record in enumerate(reversed(recent_history)):
                    y_position -= IMAGE_SPACING

                    # Если мало места на странице, создаем новую
                    if y_position < 100:
                        c.showPage()
                        y_position = height - 250
                        c.setFont(font_name, 10)

                    # Получаем путь к изображению
                    image_path = f"./static/uploads/{record['processed_image']}"

                    # Проверяем существование файла
                    if os.path.exists(image_path):
                        try:
                            # Добавляем подпись к изображению
                            timestamp_str = datetime.fromisoformat(record['timestamp']).strftime('%H:%M:%S')
                            c.setFont(font_name, 9)

                            # Обрезаем имя файла если слишком длинное
                            filename = os.path.basename(record['filename'])
                            if len(filename) > 30:
                                filename = filename[:27] + "..."

                            # Заголовок изображения
                            c.drawString(70, y_position + IMAGE_HEIGHT + 20,
                                         f"Изображение {i + 1}: {filename} - {timestamp_str}")

                            # Статистика для этого изображения
                            stats_text = (f"Собак: {record['stats']['total_dogs']}, "
                                          f"с намордником: {record['stats']['with_muzzle']}, "
                                          f"без: {record['stats']['without_muzzle']}")
                            c.drawString(70, y_position + IMAGE_HEIGHT + 5, stats_text)

                            # Добавляем само изображение
                            c.drawImage(ImageReader(image_path),
                                        70,  # X позиция
                                        y_position,  # Y позиция
                                        width=IMAGE_WIDTH,
                                        height=IMAGE_HEIGHT,
                                        preserveAspectRatio=True,
                                        mask='auto')

                            # Добавляем рамку вокруг изображения
                            c.setStrokeColorRGB(0.7, 0.7, 0.7)  # Серый цвет
                            c.setLineWidth(0.5)
                            c.rect(68, y_position - 1, IMAGE_WIDTH + 4, IMAGE_HEIGHT + 2, stroke=1, fill=0)
                            c.setStrokeColorRGB(0, 0, 0)  # Возвращаем черный

                        except Exception as img_error:
                            print(f"Ошибка при добавлении изображения {image_path}: {img_error}")
                            # Если не удалось вставить изображение, показываем заглушку
                            c.setFont(font_name, 8)
                            c.setFillColorRGB(0.8, 0.8, 0.8)
                            c.rect(70, y_position, IMAGE_WIDTH, IMAGE_HEIGHT, fill=1, stroke=0)
                            c.setFillColorRGB(0.4, 0.4, 0.4)
                            c.drawString(70 + IMAGE_WIDTH / 2 - 30, y_position + IMAGE_HEIGHT / 2,
                                         "Изображение не найдено")
                            c.setFillColorRGB(0, 0, 0)
                    else:
                        # Если файл не существует
                        c.setFont(font_name, 8)
                        c.setFillColorRGB(0.8, 0.8, 0.8)
                        c.rect(70, y_position, IMAGE_WIDTH, IMAGE_HEIGHT, fill=1, stroke=0)
                        c.setFillColorRGB(0.4, 0.4, 0.4)
                        c.drawString(70 + IMAGE_WIDTH / 2 - 40, y_position + IMAGE_HEIGHT / 2,
                                     "Файл изображения отсутствует")
                        c.setFillColorRGB(0, 0, 0)

                        # Все равно показываем информацию о записи
                        timestamp_str = datetime.fromisoformat(record['timestamp']).strftime('%H:%M:%S')
                        c.setFont(font_name, 9)
                        filename = os.path.basename(image_path) if len(
                            os.path.basename(image_path)) < 30 else os.path.basename(image_path)[:27] + "..."
                        c.drawString(70, y_position + IMAGE_HEIGHT + 20,
                                     f"Изображение {i + 1}: {filename} - {timestamp_str}")

            # Заключение
            y_position -= 40
            c.setFont(font_name, 12)
            c.drawString(50, y_position, "ЗАКЛЮЧЕНИЕ:")

            y_position -= 20
            c.setFont(font_name, 10)

            if total_dogs > 0:
                if total_without / total_dogs > 0.5:
                    conclusion = "ВНИМАНИЕ! Обнаружено много собак без намордников."
                    c.setFillColorRGB(1, 0, 0)  # Красный цвет
                else:
                    conclusion = "Ситуация нормальная, большинство собак в намордниках."
                    c.setFillColorRGB(0, 0.5, 0)  # Зеленый цвет
            else:
                conclusion = "Собаки не обнаружены."
                c.setFillColorRGB(0, 0, 0)  # Черный цвет

            c.drawString(70, y_position, conclusion)
            c.setFillColorRGB(0, 0, 0)  # Возвращаем черный цвет

            # Рекомендации (если нужно)
            if total_dogs > 0 and total_without > 0:
                y_position -= 40
                c.setFont(font_name, 11)
                c.drawString(50, y_position, "РЕКОМЕНДАЦИИ:")
                recommendations = [
                    "1. Усилить контроль в местах обнаружения",
                    "2. Разместить предупреждающие знаки",
                    "3. Провести беседы с владельцами собак"
                ]
                for i, rec in enumerate(recommendations):
                    y_position -= 20
                    if y_position < 50:
                        c.showPage()
                        y_position = height - 50
                        c.setFont(font_name, 10)
                    c.setFont(font_name, 9)
                    c.drawString(70, y_position, rec)

            # Подпись
            c.setFillColorRGB(0.75, 0.75, 0.75)  # Светло серый
            y_position -= 40
            c.setFont(font_name, 9)
            c.drawString(50, y_position, "Сгенерировано системой детекции собак без намордников в общественных метсах")

            # Информация о модели
            y_position -= 15
            c.setFont(font_name, 8)
            c.drawString(50, y_position, f"Модель: YOLOv26m | Дата: {datetime.now().strftime('%d.%m.%Y')}")

            # Сохраняем PDF
            c.save()
            #print(f"Отчет сохранен: {report_path}")

            # Проверяем размер файла
            if os.path.exists(report_path):
                file_size = os.path.getsize(report_path) / 1024  # в KB
                #print(f"Размер файла: {file_size:.1f} KB")

            return report_path

        except Exception as e:
            print(f"Ошибка генерации отчета: {e}")
            import traceback
            traceback.print_exc()
