
# Biométrico Yunet 

El proyecto parte de un prototipo funcional y se encuentra en proceso de evolución técnica, definiendo un **plan de implementación** para mejorar la precisión, robustez, escalabilidad y seguridad del reconocimiento facial, manteniendo compatibilidad con el hardware IoT existente.

---

## Arquitectura del sistema

El proyecto se divide en dos componentes principales:

- **Servidor/**  
  Aplicación Flask ejecutada en una PC local, responsable de la gestión de usuarios, base de datos, procesamiento biométrico, identificación, liveness/anti-spoofing y registro de asistencia.

- **Cliente/**  
  Aplicación para Raspberry Pi encargada de la captura de imágenes, lectura del sensor de huellas dactilares, interfaz gráfica (pantalla táctil) y comunicación con el servidor mediante MQTT.

---

## Reconocimiento facial (plan de implementación)

El sistema de reconocimiento facial **se implementará** utilizando un pipeline moderno basado en OpenCV:

- **Detección facial:** OpenCV **YuNet** (modelo ONNX).
- **Alineación y embeddings:** OpenCV **SFace**.
- **Identificación:** comparación de embeddings faciales (1:N).

Este enfoque reemplazará métodos tradicionales basados en Haar Cascades y dlib, eliminando dependencias pesadas y mejorando la tolerancia a variaciones de iluminación, pose y distancia. Los embeddings faciales se almacenarán en base de datos y se utilizarán para identificación eficiente en bases de varios miles de usuarios.

---

## Enrolamiento biométrico

Durante el enrolamiento facial **planificado**:

- Se capturarán múltiples imágenes por usuario (10–15 muestras).
- Cada imagen será procesada para extraer embeddings faciales.
- Se almacenará un embedding promedio por usuario junto con un conjunto reducido de embeddings individuales para verificación.
- El enrolamiento de huella dactilar se mantendrá mediante el sensor AS608.

Este procedimiento está orientado a mejorar la estabilidad del reconocimiento en condiciones reales.

---

## Liveness y anti-spoofing

El sistema **incorporará** mecanismos de **liveness activo** mediante retos dinámicos (challenge-response), como movimientos controlados del rostro, validados mediante análisis de cambios en la posición facial y landmarks.

De forma opcional, se contempla la integración futura de **anti-spoofing pasivo** mediante modelos de clasificación ejecutados en el servidor local para detectar intentos de suplantación con fotografías o pantallas.

---

## Optimización IoT y comunicaciones

Como parte del plan de mejora:

- El cliente enviará únicamente **recortes faciales** y eventos biométricos relevantes.
- Se evitará la transmisión continua de video.
- MQTT se mantendrá como canal de mensajería entre cliente y servidor.

El sistema queda preparado para evolucionar hacia una configuración MQTT segura (TLS y control por dispositivo).

---

## Estado del proyecto

El sistema se encuentra en fase de implementación incremental, tomando como base el prototipo existente. Las mejoras se desarrollan priorizando estabilidad operativa, validación experimental y preparación para uso real en entornos controlados.
