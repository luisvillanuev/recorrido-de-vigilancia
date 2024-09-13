import cv2
import face_recognition
import numpy as np
import os

class SistemaBiometrico:
    def __init__(self, directorio_imagenes):
        self.encodings_conocidos = []
        self.nombres_conocidos = []
        self.cargar_imagenes_conocidas(directorio_imagenes)

    def cargar_imagenes_conocidas(self, directorio):
        for filename in os.listdir(directorio):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                path = os.path.join(directorio, filename)
                nombre = os.path.splitext(filename)[0]
                imagen = face_recognition.load_image_file(path)
                encoding = face_recognition.face_encodings(imagen)[0]
                self.encodings_conocidos.append(encoding)
                self.nombres_conocidos.append(nombre)
        print(f"Cargadas {len(self.encodings_conocidos)} imágenes conocidas.")

    def reconocer(self, imagen):
        
        rgb_imagen = imagen[:, :, ::-1]

        
        localizaciones = face_recognition.face_locations(rgb_imagen)
        encodings = face_recognition.face_encodings(rgb_imagen, localizaciones)

        nombres = []
        for encoding in encodings:
            
            coincidencias = face_recognition.compare_faces(self.encodings_conocidos, encoding)
            nombre = "Desconocido"

            
            distancias = face_recognition.face_distance(self.encodings_conocidos, encoding)
            mejor_coincidencia = np.argmin(distancias)
            if coincidencias[mejor_coincidencia]:
                nombre = self.nombres_conocidos[mejor_coincidencia]

            nombres.append(nombre)

        return localizaciones, nombres

    def ejecutar_reconocimiento_camara(self):
        captura = cv2.VideoCapture(0)

        while True:
            ret, frame = captura.read()

            localizaciones, nombres = self.reconocer(frame)

            for (top, right, bottom, left), nombre in zip(localizaciones, nombres):
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, nombre, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

            cv2.imshow('Reconocimiento Biométrico', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        captura.release()
        cv2.destroyAllWindows()


directorio_imagenes = "ruta/a/tus/imagenes/conocidas"
sistema = SistemaBiometrico(directorio_imagenes)
sistema.ejecutar_reconocimiento_camara()