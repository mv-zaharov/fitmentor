from fastapi import FastAPI, Request, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import os
import cv2
import shutil
import uvicorn


# Инициализация приложения FastAPI
app = FastAPI(title=__name__)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Загрузка модели YOLO с обработкой исключений
try:
    model = YOLO("yolo11n-pose.pt")
except Exception as e:
    raise RuntimeError(f"Не удалось загрузить модель YOLO: {e}")

# Папка для временного хранения загруженных файлов
UPLOAD_FOLDER = '/tmp/uploads/'
RESULT_FOLDER = '/tmp/result/'
TEMP_PIC_FOLDER = os.path.join(RESULT_FOLDER, 'pic/')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TEMP_PIC_FOLDER, exist_ok=True)

@app.post("/process/")
async def process_file(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No selected file")

    # Секция  try-except для сохранения файла во временную директорию
    try:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(file_path, 'wb') as f:
            f.write(await file.read())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save the file: {e}")

    # Выполнение предсказания на полученном файле
    try:
        results = model(file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {e}")

    # Обработка результатов и создание кадров
    try:
        # Определяем видео запись
        #output_video_path = os.path.join(RESULT_FOLDER, 'result.mp4')
        output_video_path = os.path.join(RESULT_FOLDER, file.filename)
        frame_height, frame_width = results[0].orig_shape
        out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 20, (frame_width, frame_height))

        for index, result in enumerate(results):
            # Генерация текущего кадра
            frame = result.plot()
            
            # Сохранение изображения во временную папку
            #temp_image_path = os.path.join(TEMP_PIC_FOLDER, f"output_{index}.jpg")
            #cv2.imwrite(temp_image_path, frame)

            # Добавление кадра в видео
            out.write(frame)

        # Освобождение ресурса Видеозаписи
        out.release()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing results failed: {e}")
    finally:
        # Удаляем все временные изображения из папки даже при ошибках
        shutil.rmtree(TEMP_PIC_FOLDER, ignore_errors=True)
        os.makedirs(TEMP_PIC_FOLDER, exist_ok=True)
        
    #Проверяем, создан ли файл
    if not os.path.exists(output_video_path):
        raise HTTPException(status_code=500, detail="Processed video file not found")

    #Возвращаем обработанный видеофайл в ответе
    return FileResponse(
        path=output_video_path,
        media_type="video/mp4",
        filename=f"ok_{file.filename}"  #Файл будет загружаться с префиксом "ok_"
    )        

    #return JSONResponse(content={"message": "File processed successfully"})


@app.get("/process/")
async def process_get():
    return JSONResponse(content={"message": "GET method is not allowed"})

if __name__ == "__main__":
    # Устанавливаем параметры запуска сервера
    try:
        #uvicorn.run(app, host="localhost", port=5000, log_level="info")
        uvicorn.run(app=app, host='localhost', port=5000, workers=1)      
        
    except Exception as e:
        print(f"Failed to start server: {e}")