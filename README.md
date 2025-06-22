# MNIST Handwritten Digit Generator

## Estructura del proyecto

```
mnist_digit_generator/
│
├── app.py                  # Web app principal (Streamlit)
├── model/
│   ├── gan.py              # Arquitectura y funciones del modelo (ejemplo: GAN)
│   └── train_gan.ipynb     # Script de entrenamiento en Colab
├── utils/
│   └── generate.py         # Funciones para cargar modelo y generar imágenes
├── requirements.txt        # Dependencias para reproducir el entorno
└── README.md               # Instrucciones de uso y despliegue
```

## Pasos para finalizar la app

1. Entrena el modelo en Google Colab usando `model/train_gan.ipynb` y guarda el archivo `generator.pth` en `model/`.
2. Completa la función de generación en `utils/generate.py` para que use tu modelo entrenado.
3. Ejecuta la app localmente con:
   ```
   streamlit run app.py
   ```
4. Despliega en [Streamlit Cloud](https://share.streamlit.io/) para obtener un enlace público.

## Notas
- Modifica y completa los scripts según tu arquitectura y necesidades.
- Asegúrate de que `requirements.txt` tenga todas las dependencias necesarias.
