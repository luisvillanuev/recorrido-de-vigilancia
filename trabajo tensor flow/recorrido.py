import tensorflow as tf
import numpy as np
import random
import time

class Guardia:
    def __init__(self, nombre):
        self.nombre = nombre
        self.posicion_actual = "Entrada"

    def mover(self, nueva_posicion):
        print(f"{self.nombre} se mueve de {self.posicion_actual} a {nueva_posicion}")
        self.posicion_actual = nueva_posicion
        time.sleep(1)  
        
    def vigilar(self, ubicacion, modelo_anomalia):
        print(f"{self.nombre} está vigilando {ubicacion}")
        
        dato = np.random.normal(0, 1, (1, 1))  
        prediccion = modelo_anomalia.predict(dato)
        if prediccion > 0.5:
            print(f"¡Alerta! Posible anomalía detectada en {ubicacion}")
        time.sleep(2)  

def crear_modelo_anomalia():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(8, activation='relu', input_shape=(1,)),
        tf.keras.layers.Dense(4, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model

def entrenar_modelo(modelo):
    
    X_train = np.random.normal(0, 1, (1000, 1))
    y_train = (X_train > 1.5).astype(int).reshape(-1, 1)
    modelo.fit(X_train, y_train, epochs=5, verbose=0)

def recorrido_vigilancia(guardia, ubicaciones, num_rondas, modelo_anomalia):
    for ronda in range(num_rondas):
        print(f"\nIniciando ronda {ronda + 1}")
        ubicaciones_ronda = ubicaciones.copy()
        random.shuffle(ubicaciones_ronda)  
        
        for ubicacion in ubicaciones_ronda:
            guardia.mover(ubicacion)
            guardia.vigilar(ubicacion, modelo_anomalia)

    print(f"\nRecorrido de vigilancia completado. {guardia.nombre} regresa a la Entrada.")
    guardia.mover("Entrada")


guardia = Guardia("Guardia 1")
ubicaciones = ["Pasillo A", "Oficina 1", "Sala de conferencias", "Pasillo B", "Oficina 2", "Cafetería"]
num_rondas = 2


modelo_anomalia = crear_modelo_anomalia()
entrenar_modelo(modelo_anomalia)


recorrido_vigilancia(guardia, ubicaciones, num_rondas, modelo_anomalia)