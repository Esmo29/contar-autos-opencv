import cv2  # Importa OpenCV

# Cargar el video
capture = cv2.VideoCapture("videoplayback.avi")

# Cargar el archivo de clasificación XML (modelo de entrenamiento)
carros = cv2.CascadeClassifier("haarcascade_car.xml")

# Bucle para leer los cuadros del video
while True:
    ret, frames = capture.read()  # Captura cada cuadro del video
    
    if not ret:  # Si no se puede leer el cuadro, salir del bucle
        break
    
    # Convertir los cuadros a escala de grises para mejorar la detección
    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
    
    # Detección de autos en el cuadro
    cars = carros.detectMultiScale(gray, 1.1, 1)
    
    # Dibujar un rectángulo alrededor de cada auto detectado
    for (x, y, w, h) in cars:
        cv2.rectangle(frames, (x, y), (x + w, y + h), (0, 0, 255), 2)
    
    # Mostrar el video con las detecciones
    cv2.imshow("Video - Contador de Autos", frames)
    
    # Salir si se presiona la tecla 'ESC' (código ASCII 27)
    if cv2.waitKey(33) == 27:
        break

# Liberar los recursos
capture.release()
cv2.destroyAllWindows()

