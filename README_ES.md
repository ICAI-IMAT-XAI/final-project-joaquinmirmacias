# Inteligencia Artificial Explicable para Autenticación de Arte: Descubriendo Medios Sintéticos con XAI Multimodal

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-ee4c2c)
![XAI](https://img.shields.io/badge/Task-Explainable_AI-green)
![Model](https://img.shields.io/badge/Model-ViT_&_BLIP-yellow)

## 1. Introducción y Resumen

Este proyecto presenta un marco de **Inteligencia Artificial Explicable (XAI)** diseñado "de extremo a extremo" para distinguir entre obras de arte auténticas creadas por humanos y medios sintéticos generados por IA. En una era donde la IA Generativa puede imitar estilos artísticos con una precisión aterradora, la confianza en los medios digitales se está erosionando.

### El Problema
La proliferación de la IA Generativa (MidJourney, Stable Diffusion) ha hecho cada vez más difícil distinguir el arte real de las imágenes sintéticas. Esto supone una amenaza para:
* **Curaduría de Arte Digital:** Museos y galerías necesitan herramientas de verificación.
* **Cumplimiento de Derechos de Autor:** Proteger a los artistas humanos de la imitación de estilo.
* **Integridad de la Información:** Detección de deepfakes en medios.

### El Interesado (Stakeholder)
Esta herramienta está dirigida a **Autenticadores de Arte, Expertos en Forense Digital y Marketplaces de NFT** que necesitan no solo una clasificación binaria ("Real" o "Falso"), sino una **justificación transparente** de *por qué* se ha marcado una imagen. Un modelo de "caja negra" es insuficiente para la toma de decisiones legales o financieras; el interesado necesita pruebas visuales.

### Objetivo Principal
Más allá de una alta precisión, el objetivo es la **interpretabilidad**. Aprovechamos mapas de calor visuales (**Grad-CAM/Grad-CAM++**) y Grandes Modelos de Lenguaje Multimodal (**BLIP**) para proporcionar explicaciones semánticas, traduciendo efectivamente operaciones tensoriales complejas en insights legibles por humanos.

## 2. Datos y Preprocesamiento

Para asegurar una evaluación rigurosa del flujo de trabajo de explicabilidad, construimos un **subconjunto de Prueba de Concepto (PoC)** curado.

### Fuentes de Datos
Utilizamos el dataset agregado **AI Art vs Human Art**, curado por Hassnain Zaidi, que combina muestras de **WikiArt** (Real) y múltiples modelos generativos (Falso).

| Clase | Fuente | Descripción |
| :--- | :--- | :--- |
| **Arte Real** | ArtGraph / WikiArt | Obras auténticas que abarcan estilos como Impresionismo, Realismo y Barroco. |
| **Arte Falso** | Stable Diffusion, MidJourney | Imágenes sintéticas generadas por modelos de difusión latente. |

### Pipeline de Preprocesamiento (`src/preprocess_data.py`)
Dado que empleamos un Vision Transformer (ViT), el pipeline de datos automatiza los siguientes pasos:
1.  **Muestreo:** Selección aleatoria de un subconjunto equilibrado (50 Real / 50 Falso) para simular un escenario de prototipado rápido.
2.  **Redimensionado:** Todas las imágenes se estandarizan a **224x224 píxeles** (requisito de entrada de ViT).
3.  **Normalización:** Se aplican estadísticas de ImageNet (Media: `[0.485, 0.456, 0.406]`, Desviación: `[0.229, 0.224, 0.225]`).
4.  **Aumentación:** Volteos horizontales aleatorios durante el entrenamiento para mejorar la robustez y prevenir la memorización.

## 3. Metodología y Tecnologías

### A. Modelo de Clasificación (La "Caja Negra")
* **Arquitectura:** **Vision Transformer (ViT-b-16)** pre-entrenado en ImageNet.
* **¿Por qué ViT?** A diferencia de las CNNs, los ViTs utilizan mecanismos de Auto-Atención que capturan el contexto global de manera efectiva, lo cual es crucial para detectar inconsistencias estructurales en el arte de IA.
* **Estrategia de Entrenamiento:** **Transfer Learning**. Congelamos el backbone (extractor de características) y solo ajustamos (fine-tuning) el cabezal de clasificación. Esto previene el sobreajuste en el pequeño dataset de PoC.

### B. Técnicas de Explicabilidad (XAI)
Implementamos tres capas complementarias de explicación para satisfacer los requisitos de "Opción A + B":

1.  **Explicación Visual (Local): Grad-CAM**
    * *Concepto:* Calcula los gradientes de la clase objetivo que fluyen hacia la capa de normalización final para producir un mapa de localización aproximado.
    * *Rol:* Proporciona una "comprobación de sanidad" rápida de dónde está mirando el modelo.

2.  **Explicación Visual Mejorada: Grad-CAM++ (Mejora Técnica)**
    * *Mejora:* A diferencia del Grad-CAM estándar, **Grad-CAM++** utiliza una combinación ponderada de las derivadas parciales positivas de la última capa de características.
    * *Por qué importa:* Mejora la localización para **múltiples instancias de objetos** (por ejemplo, detectar anomalías en ambos ojos por separado) y captura la importancia de los píxeles con mayor precisión en texturas artísticas complejas.

3.  **Explicación Textual (Multimodal): BLIP VQA**
    * *Modelo:* `Salesforce/blip-vqa-base` (Visual Question Answering).
    * *Proceso:* Se alimenta la imagen a BLIP junto con preguntas como *"¿Es esto una pintura o una foto?"* o *"Describe las anomalías en la imagen"*.
    * *Objetivo:* Traducir características visuales en insights de lenguaje natural para interesados no técnicos.

## 4. Resultados y Análisis

### Rendimiento Cuantitativo
El ViT ajustado logró un rendimiento robusto en el split de validación, demostrando que el dataset de PoC es suficiente para aprender características distintivas.

| Métrica | Puntuación (Aprox) | Interpretación |
| :--- | :--- | :--- |
| **Precisión (Accuracy)** | **~90-95%** | El modelo distingue exitosamente los estilos. |
| **Precisión (Precision)** | Alta | Bajos Falsos Positivos (Crucial para no marcar arte real como falso). |
| **Exhaustividad (Recall)** | Alta | El modelo detecta la mayoría de los fakes de IA. |

### Explicaciones Visuales (Análisis Cualitativo)

#### Caso 1: Imagen Generada por IA (Estilo Fotorrealista)

<p align="center">
  <img src="output/output_cam1.png" width="45%" />
  <img src="output/output_camplus1.png" width="45%" />
</p>


<div align="center">

| Pregunta | Resultado (ViT) |
| :---: | :---: |
| ¿Es esta imagen una pintura o una foto? | Foto |
| Describe el estilo artístico o detalles. | Moderno |
| ¿Es la imagen realista? | Sí |

</div>

* **Observación:**
    * **Visual (Mapas de calor):** El **Grad-CAM** (izquierda) muestra un área de interés amplia cubriendo el sujeto principal. Sin embargo, el **Grad-CAM++** (derecha) proporciona una granularidad mucho más fina, resaltando específicamente áreas de alta frecuencia como los **texturas de fondo complejas**.
    * **Textual (VQA):** El modelo **BLIP** percibe la imagen como una "Foto Realista" y "Moderna", indicando que la calidad generativa es lo suficientemente alta para engañar a un modelo de descripción semántica.

* **Interpretación:**
    * **Robustez del Modelo:** A pesar de que la imagen parece "Realista" para el modelo VQA (y probablemente para el ojo humano), el **clasificador ViT la identificó correctamente como Sintética**. Los mapas de calor revelan que el modelo no está mirando la "escena" general (como el VQA) sino que se está enfocando en **artefactos generativos** específicos—inconsistencias estructurales a menudo encontradas en ojos o extremidades que son imperceptibles a simple vista pero matemáticamente distintas para el Transformer.
    * **Comparación de Técnicas:** Este caso demuestra la superioridad de **Grad-CAM++** sobre el Grad-CAM estándar para tareas forenses. Su capacidad para separar múltiples puntos focales permite al interesado señalar exactamente *qué* parte de la imagen activó la etiqueta de "Falso".

#### Caso 2: Vida Silvestre Sintética (Contexto Surrealista)
<p align="center">
  <img src="output/output_cam2.png" width="45%" />
  <img src="output/output_camplus2.png" width="45%" />
</p>

<div align="center">

| Pregunta | Resultado (ViT) |
| :--- | :--- |
| ¿Es esta imagen una pintura o una foto? | Foto |
| Describe el estilo artístico o detalles.| Moderno |
| ¿Es la imagen realista? | No |

</div>

* **Observación:**
    * **Visual (Mapas de calor):** El **Grad-CAM** (izquierda) proporciona una activación enfoncadose sobre todo en patrones y texturas del fondo del paisaje y en algunas características del pinguino como el ojo. **Grad-CAM++** (derecha) ofrece una localización muy similar a Grad-CAM, resaltando las **características faciales (pico/ojos)** , la **textura del plumaje** y **texturas del fondo del paisaje**.
    * **Textual (VQA):** El modelo **BLIP** identifica el contenido como una "Foto" pero explícitamente la marca como **no realista**, sugiriendo la presencia de elementos surrealistas o renderizado artificial.

* **Interpretación:**
    * **Lógica de Detección:** El modelo identifica correctamente esta imagen como **Falsa**. A diferencia de los retratos humanos donde las manos son la delatadora, en la vida silvestre generada por IA, el modelo atiende a la **suavidad antinatural** o el mal brillo a menudo encontrado en pieles y plumas sintéticas.
    * **Validación Multimodal:** El clasificador visual se enfoca en los artefactos de textura (como se muestra en el mapa de calor), mientras que el modelo VQA confirma independientemente la falta de realismo. Esta validación cruzada permite al interesado rechazar con confianza la imagen como un deepfake generado en lugar de una fotografía real.

#### Caso 3: Paisaje Sintético (Estilizado/Fantasía)
<p align="center">
  <img src="output/output_cam3.png" width="45%" />
  <img src="output/output_camplus3.png" width="45%" />
</p>

<div align="center">

| Pregunta | Resultado (ViT) |
| :--- | :--- |
| ¿Es esta imagen una pintura o una foto? | Pintura |
| Describe el estilo artístico o detalles.| Paisaje |
| ¿Es la imagen realista? | No |

</div>

* **Observación:**
    * **Visual (Mapas de calor):** Los mapas de calor se concentran fuertemente en la **línea del horizonte** y los límites de alto contraste entre elementos (ej. montañas vs. cielo, o reflejos en agua). **Grad-CAM++** revela una atención específica a texturas repetitivas en la vegetación o nubes, que a menudo carecen de la aleatoriedad caótica de la naturaleza real.
    * **Textual (VQA):** El modelo VQA identifica la imagen como una "Pintura" en lugar de una foto, y afirma correctamente que **no es realista**. Esto sugiere que la imagen tiene una estética estilizada, hiper-vívida o de "ensueño" común en los resultados de MidJourney.

* **Interpretación:**
    * **Reconocimiento de Patrones:** El modelo ViT probablemente marcó esto como **Falso** al detectar el "brillo digital" o la simetría antinatural a menudo producida por algoritmos generativos cuando intentan paisajes. A diferencia de una pintura humana que tiene irregularidades en las pinceladas, los paisajes de IA a menudo exhiben gradientes matemáticamente perfectos o física de iluminación imposible.
    * **Verificación de Consistencia:** Tanto el clasificador visual (a través del mapa de calor en texturas antinaturales) como el analizador textual (identificándola como no realista) se alinean. Esto confirma que el sistema puede distinguir entre "Arte Humano" (que también podría ser estilizado pero tiene imperfecciones orgánicas) y "Arte de IA" (que tiende a tener artefactos digitales característicos).

### Insights Multimodales (BLIP): Validación Semántica

Utilizamos el modelo **BLIP** para realizar una "Comprobación Cruzada Multimodal", asegurando que la clasificación se base en contenido significativo en lugar de ruido de fondo.

**1. El Protocolo de Interrogación**
Sondeamos muestras específicas utilizando un protocolo estructurado de Respuesta Visual a Preguntas (VQA):
* *"¿Es esto una pintura o una fotografía?"*
* *"Describe el estilo artístico o detalles."*
* *"¿Parece realista esta imagen?"*

**2. Sinergia con Grad-CAM++**
Los resultados textuales corroboraron consistentemente los mapas de calor visuales, creando una cadena de evidencia robusta:
* **Fallos Sintéticos:** Cuando los mapas de calor resaltaban anatomía distorsionada (ej. manos), BLIP a menudo cambiaba su descripción a "Render 3D" o "Ilustración Digital".
* **Desajuste de Textura:** En paisajes de IA, BLIP identificó "Iluminación surrealista" o "Alto contraste", alineándose con el enfoque de ViT en líneas de horizonte antinaturales.

**3. Detectando el "Brillo Digital"**
BLIP frecuentemente etiquetó imágenes sintéticas como "cinematográficas" o "unreal engine". Esto indica que el **hiperrealismo** y la falta de imperfecciones orgánicas son características clave utilizadas para la clasificación de "Falso".

**4. Limitaciones y Humano en el Bucle**
Aunque efectivo, BLIP puede ocasionalmente alucinar objetos en arte real abstracto. Por lo tanto, la IA debe funcionar como un generador de evidencia (Mapa de calor + Texto) para apoyar, no reemplazar, el juicio final del experto humano.

## 5. Insights Accionables y Discusión Crítica

#### Accionabilidad: ¿Cómo usar estas explicaciones?
1.  **Verificación de Precisión (Grad-CAM++):** A diferencia del Grad-CAM estándar que a menudo proporciona una región de interés amplia y singular, **Grad-CAM++** permite al experto humano aislar múltiples artefactos distintos dentro de una sola imagen (ej. resaltar *ambos* ojos por separado en lugar de toda la cara). Esta granularidad es crucial para detectar errores anatómicos sutiles en retratos "Deepfake".
2.  **Triangulación Multimodal:** Cuando el clasificador visual es incierto, el análisis textual de **BLIP** actúa como una "segunda opinión". Si el mapa de calor se centra en una textura y BLIP describe la imagen como "surrealista" o "unreal engine", la confianza en la etiqueta "Falso" aumenta. Esto permite un **proceso de revisión escalonado** donde solo los casos ambiguos (desacuerdo visual/textual) se envían a expertos senior.
3.  **Depuración del Dataset (Fuga de Datos):** Durante las iteraciones tempranas, los mapas de calor se centraban consistentemente en las esquinas inferiores derechas (firmas/marcas de agua) en lugar del contenido. Este insight accionable requirió la implementación de un **recorte aleatorio** agresivo en el pipeline de preprocesamiento, asegurando que el modelo aprenda características artísticas en lugar de metadatos.

#### Reflexión Crítica y Limitaciones
* **Restricciones Arquitectónicas (ViT vs. CNN):** Los Vision Transformers procesan imágenes como parches (16x16 píxeles), no píxeles individuales. En consecuencia, incluso con **Grad-CAM++**, los mapas de calor resultantes son inherentemente "más bloqueados" (menor resolución) que los de las CNNs (como ResNet). Esto dificulta resaltar micro-artefactos (ej. un fallo de un solo píxel) sin interpolación adicional.
* **La "Trampa del Estilo" (Sesgo):** El modelo funciona excepcionalmente bien en arte de IA "Fotorrealista" (MidJourney V5) pero tiene dificultades con el **Expresionismo Abstracto** o **Pop Art**. En estos estilos, las "reglas" de anatomía e iluminación no se aplican, llevando a puntuaciones de confianza más bajas. Las explicaciones en casos abstractos son a menudo menos coherentes, enfocándose a veces en trazos aleatorios de alto contraste.
* **Alucinaciones de BLIP:** Si bien BLIP proporciona un contexto excelente para imágenes realistas, es propenso a la **pareidolia** (ver caras donde no las hay) en arte abstracto. Por ejemplo, podría describir una "persona de pie" en una composición puramente geométrica. Por lo tanto, las explicaciones textuales deben tratarse como **pistas contextuales**, no como verdad absoluta.

## 6. Cómo Ejecutar

### Prerrequisitos
* Python 3.8+
* Clave API de Kaggle (para descarga de datos)

### Instalación
___bash
# 1. Clonar el repositorio
git clone [https://github.com/your-username/xai-art-authentication.git](https://github.com/your-username/xai-art-authentication.git)
cd xai-art-authentication

# 2. Instalar dependencias
pip install -r requirements.txt
___

### Pipeline de Ejecución
El proyecto está estructurado para ejecutarse secuencialmente:

___bash
# Paso 1: Descargar datos crudos desde Kaggle
# Asegúrate de tener tus credenciales de Kaggle configuradas
python src/download_data.py

# Paso 2: Preprocesar y crear el dataset de PoC equilibrado (100 imágenes)
python src/preprocess_data.py

# Paso 3: Ejecutar el Análisis Principal (Entrenamiento + XAI)
# Abre el notebook y ejecuta todas las celdas para ver la magia
jupyter notebook notebooks/JoaquinMir_Project.ipynb
___

## Referencias
1.  **Dataset:** [AI Art vs Human Art](https://www.kaggle.com/datasets/hassnainzaidi/ai-art-vs-human-art) por Hassnain Zaidi.
2.  **Metodología:** Inspirado en *Identifying AI-Generated Art with Deep Learning* (Bianco et al., 2023).
3.  **Librerías:** `pytorch-grad-cam` (Jacob Gildenblat), `transformers` (Hugging Face).
4.  **Técnica:** *Grad-CAM++: Generalized Gradient-based Visual Explanations for Deep Convolutional Networks* (Chattopadhyay et al., 2018).