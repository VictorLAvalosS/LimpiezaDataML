# 🧹 Data Cleaner PRO + Machine Learning Suite 🧠
## Práctica del Primer Parcial — Ciencia de Datos y Machine Learning
### Lic. Patricia Rodríguez Bilbao

---

## 📘 Descripción General

Plataforma interactiva construida en **Python + Streamlit** para cumplir al 100% con la metodología de la asignatura, organizada en **TRES AVANCES INCREMENTALES** según lo solicitado:

| Avance | Requisito Docente | Funcionalidad en el Sistema |
|--------|-------------------|------------------------------|
| **1️⃣ Primer Avance** | **a)** Limpieza 3 tipos + acciones **b)** Train/Test Split | Pestañas 1 a 4: Auditoría, Pipeline, Comparativa Antes/Después, Resultados. **Pestaña 7**: Split 80/20 con justificación y gráficos. |
| **2️⃣ Segundo Avance** | **c)** Implementar Modelos Regresionales + Árboles de Decisión | Pestaña 5 (Configuración) + Pestaña 6 (Comparativa 6+ familias: Regresión, Logística, Árboles Gini/Entropía, Random Forest, SVM, KNN, RNA) |
| **📄 Informe Final** | **d)** Validación ≥4 parámetros + Matriz Confusión + Evaluar | **Todo auto-generado**: Pestaña 6 (métricas, Matriz, ROC-AUC) + **Pestaña 11 (Reporte Automático completo I-VII listo para Word)** |

---

## 🏗️ Estructura del Proyecto (Arquitectura Modular)

```
Limpieza Data Set/
├── 📄 LIBRERIAS_USADAS.md     # ← Docente pregunta "¿Qué librerías usan?" → AQUÍ ESTÁ LA RESPUESTA
├── 📄 README.md               # ← ESTE ARCHIVO: guía completa de instalación y uso
├── 📄 requirements.txt        # 12 dependencias (pip install -r requirements.txt)
├── 📄 .env.example            # Plantilla para API Key Gemini (OPCIONAL, NO OBLIGATORIA)
├── 📄 app.py                  # 🖥️ INTERFAZ GRÁFICA: 11 pestañas interactivas (Streamlit)
├── 📄 run.bat                 # Inicio rápido en Windows
├── 📄 install.bat             # Instalación + configuración inicial automática (Windows)
│
├── 📁 src/                    # 🧠 NÚCLEO LÓGICO (Motor de Código)
│   ├── cleaner.py             # DataCleaner: Auditoría, 3 tipos de limpieza, Codificación Categórica
│   └── models.py              # ModelTrainer: 6+ familias ML, GridSearchCV, Cross-Validation, Exportar
│
└── 📁 data/                   # 💾 Almacenamiento de sesión (auto-gestionado)
    ├── raw/                   # Dataset crudo temporal (current_session.pkl = persistencia)
    └── processed/             # Modelos exportados .joblib, datasets limpios
```

### 🔗 Archivos Críticos (Haz clic para ver):
| Archivo | Responsabilidad |
|---|---|
| [app.py](file:///c:/Users/User/Downloads/DML/Limpieza%20Data%20Set/app.py) | Orquesta la UI, 11 pestañas, estado de sesión, integración limpieza + ML |
| [src/cleaner.py](file:///c:/Users/User/Downloads/DML/Limpieza%20Data%20Set/src/cleaner.py) | Auditoría, limpieza 3 tipos, codificación One-Hot + Label Encoding, protección Target |
| [src/models.py](file:///c:/Users/User/Downloads/DML/Limpieza%20Data%20Set/src/models.py) | 22 modelos, GridSearchCV, CV 5-fold, detección Overfitting, ROC-AUC, Predict Individual, Export |
| [LIBRERIAS_USADAS.md](file:///c:/Users/User/Downloads/DML/Limpieza%20Data%20Set/LIBRERIAS_USADAS.md) | Detalle de 12 librerías, para qué sirven, justificación académica |

---

## ⚙️ Instalación Paso a Paso (Para CUALQUIER Persona)

### 🪟 Método 1 — Instalación RÁPIDA en Windows (Recomendado para tu grupo)
```bash
# Doble clic en el archivo:
install.bat
```
Este script **automáticamente**:
1. Crea un entorno virtual `venv`
2. Activa el entorno
3. Instala todas las dependencias de `requirements.txt`
4. Copia `.env.example` como `.env` (listo para configurar)
5. ¡Abre una terminal lista para ejecutar!

---

### 💻 Método 2 — Instalación Manual (Paso a paso)
Funciona en Windows / macOS / Linux:

#### PASO 1 — Tener Python 3.10+ instalado
```bash
python --version
# Debe mostrar: Python 3.10.x, 3.11.x, 3.12.x o 3.13.x
```

#### PASO 2 — (Opcional pero BUENA PRÁCTICA) Crear entorno virtual
```bash
# En Windows:
python -m venv venv
venv\Scripts\activate

# En macOS/Linux:
python3 -m venv venv
source venv/bin/activate
```

#### PASO 3 — Instalar las 12 librerías automáticamente
```bash
pip install -r requirements.txt
```
⏳ Tarda ~3-5 minutos (descarga scikit-learn + pandas + streamlit).

#### PASO 4 — (OPCIONAL, NO REQUERIDO) Configurar API Key Gemini
Si tu grupo quiere usar el chat IA:
```bash
# Windows (Powershell/CMD):
copy .env.example .env
# Abre .env con bloc de notas y reemplaza:
GEMINI_API_KEY=tu_api_key_real_aqui
```
> 📌 **NO ES OBLIGATORIO.** Si no lo haces, el sistema funciona exactamente igual. Solo desactiva el chat con IA.

---

## 🚀 Cómo INICIAR / EJECUTAR el Sistema

### Opción A — Windows (más fácil):
```bash
# Doble clic en:
run.bat
```

### Opción B — Cualquier sistema operativo:
```bash
# Asegúrate de estar en la carpeta del proyecto "Limpieza Data Set"
streamlit run app.py
```

✅ **Resultado esperado:** Se abrirá automáticamente tu navegador web predeterminado en `http://localhost:8501` con la interfaz.

> 🔔 Si no se abre solo, copia manualmente la URL que aparece en la terminal.

---

## 🧭 GUÍA DE USO POR PESTAÑAS (Flujo Completo)

El sistema tiene **11 PESTAÑAS ORDENADAS** — sigue la numeración de izquierda a derecha:

---

### 📥 CARGA INICIAL (Antes de pestañas)
1. Arrastra/sube tu dataset asignado (`.csv`, `.xlsx` o `.xls`).
2. **IMPORTANTE:** Si eliges **🤖 Limpieza Inteligente con IA (Klaro)** o **🛠️ Limpieza Manual**, haz clic en **🚀 EJECUTAR** arriba a la derecha.
3. Espera a que el pipeline termine (spinner azul).

---

### 🔹 PESTAÑA 1 → 📊 Auditoría
- Diagnóstico automático del dataset crudo:
  - Duplicados, columnas con nulos >5%, texto corrupto (mojibake), variables tipo mixto, outliers detectados.
  - **Sirve para tu informe:** Copia la tabla en "Marco Referencial".

---

### 🔹 PESTAÑA 2 → ⚙️ Pipeline
- 🎯 **Paso OBLIGATORIO:** Selecciona la **columna TARGET** (lo que quieres predecir). El sistema la PROTEGE — nunca se modifica la verdad de datos.
- Bitácora de operaciones ejecutadas: fecha, acción, detalles, justificación TÉCNICA.

---

### 🔹 PESTAÑA 3 → 🔍 Comparativa Antes vs Después
- 📝 **Sustento Metodológico listo para tu informe** (A. Integridad, B. Saneamiento, C. Calidad Estadística).
- Tabla de estadísticas descriptivas (min, Q1, mediana, Q3, max) antes y después del Capping IQR.
- Expande "📄 Texto para el Informe Universitario" → copia y pega directo.

---

### 🔹 PESTAÑA 4 → 💾 Resultados
- Dataset 100% limpio, listo para ML.
- Botón para descargar `dataset_limpio.csv`.

---

### 🔹 PESTAÑA 5 → 🎯 Target (Predicciones ML)
1. Confirma la variable a predecir **(Y)**.
2. Elige variables predictoras **(X)** — por defecto, todas excepto Y.
3. Selecciona **Modelo A** y **Modelo B** para la comparativa "Cara a Cara".
4. Clic en **🚀 Ejecutar Predicciones**. ⏳ Este paso tarda más porque usa **GridSearchCV + Validación Cruzada**.

---

### 🔹 PESTAÑA 6 → 🏆 Comparación ML (El más importante para tu nota)
- 🥊 **Cara a Cara Modelo A vs Modelo B:**
  - 4+ métricas numéricas
  - Expande "📊 Análisis Avanzado": **Overfitting detection + CV 5-Fold**
  - Matriz de Confusión (Clasificación) o Dispersión (Regresión)
  - Extra: Curva ROC-AUC / Análisis Residuales / Importancia de Variables Top 10
- 📊 **Gráfico Comparación Global:** Todos los modelos con todas las métricas.
- 📋 **Tabla Resumen Global:** Métricas + Mejores Hiperparámetros + CV Mean.
- 🏆 **Mejor Modelo Global** destacado automáticamente.

---

### 🔹 PESTAÑA 7 → 🎯 Train/Test Split
- ✅ Justificación METODOLÓGICA (80/20, random_state=42, stratify).
- Métricas de forma X_train, X_test, y_train, y_test.
- 📊 **GRÁFICO de Distribución de Clases (Train vs Test)** → Sirve para demostrar a la docente que el split fue balanceado.
- Descarga los 4 CSV para entregarlos junto al informe.

---

### 🔹 PESTAÑA 8 → 📄 Bitácora ML
- Todas las acciones de limpieza en formato tabla (Fase, Detalles, Justificación).

---

### 🔹 PESTAÑA 9 → 🔮 Predicción Individual (WOW!)
- Selecciona cualquier modelo entrenado.
- Formulario dinámico: **para cada variable X, muestra un `number_input` con rango real o un `selectbox` con categorías**.
- Clic en **🎯 Realizar Predicción** → Obtén el valor predicho + **PROBABILIDADES POR CLASE** (barras de progreso!).

---

### 🔹 PESTAÑA 10 → 💾 Exportar Modelos
- Descarga cualquier modelo entrenado en formato `.joblib`.
- Incluye el modelo, el scaler, label_encoders y nombres de features.
- **Tienes snippet de Python listo** para cargarlo y usarlo en otro script.

---

### 🔹 PESTAÑA 11 → 📝 Reporte Académico AUTOMÁTICO 🔥
✅ **ESTA PESTAÑA TE AHORRA HORAS DE REDACCIÓN.**

Genera automáticamente el informe COMPLETO con las secciones que pide la docente:
```
I.   Introducción
II.  Objetivos
III. Marco Teórico (Regresión / Árboles Decisión + GridSearchCV + CV)
IV.  Marco Referencial (tu dataset específico)
V.   Ingeniería
     a) Descripción del problema
     b) Tecnologías
     c) Planteamiento de la solución
     d) IMPLEMENTACIÓN (Primer Avance → Segundo Avance → Informe Final) con NÚMEROS REALES
     e) Pruebas y Validación 4+ métricas + Matriz Confusión
VI.  Conclusiones (4 puntos)
VII. Bibliografía (5 referencias académicas oficiales)
```
📥 Botón **Descargar Reporte (.md)** y botón **Copiar al Portapapeles** → pégalo directo en Word.

---

## 🤖 Modelos de Machine Learning Incluidos

| Familia | Modelos Regresionales (Y continua) | Árboles / Clasificación (Y discreta) |
|---|---|---|
| **1. Regresión y Regularización** | Lineal Múltiple, Polinómica Gr2/Gr5, Ridge L2, Lasso L1 | Regresión Logística Binaria + Softmax Multiclase |
| **2. Árboles y Ensambles** | DecisionTreeRegressor + RandomForestRegressor | DecisionTree (GINI, Entropía) + RandomForestClassifier |
| **3. SVM / MVS** | SVR (Lineal, RBF, Poly, Sigmoide) | SVC (Lineal, RBF, Poly, Sigmoide) |
| **4. KNN** | KNeighborsRegressor (K=3/5/7) | KNeighborsClassifier (K=3/5/7) |
| **5. Redes Neuronales** | MLPRegressor (Perceptrón Multicapa) | MLPClassifier (Perceptrón Multicapa) |

> 🔧 **Todos los modelos tienen optimización GridSearchCV** — no se quedan con parámetros por defecto.

---

## ❓ Solución de Errores Comunes

| Error | Solución |
|-------|----------|
| `ModuleNotFoundError: No module named 'streamlit'` | No instalaste las dependencias. Ejecuta: `pip install -r requirements.txt` |
| `Error: Streamlit no se reconoce` | Activa tu entorno virtual: `venv\Scripts\activate` (Windows) o usa `python -m streamlit run app.py` |
| Los modelos tardan mucho | Es NORMAL: GridSearchCV prueba múltiples combinaciones de hiperparámetros + 5-Fold CV. Es el costo de la calidad académica. |
| `UnicodeDecodeError` al cargar CSV | El sistema ya intenta UTF-8 y Latin-1 automáticamente. Si falla, guarda tu CSV como UTF-8. |
| `ValueError: Found unknown categories` | Variable categórica con valores raros → vuelve a ejecutar Limpieza Manual o contacta. |
| Modelo sale con Overfitting | Es una ALERTA académica BUENA. Menciona en conclusiones: "Se detectó sobreajuste en X modelo, se recomienda recortar profundidad o aumentar datos." |

---

## 🎓 Checklist para ENTREGA FINAL (Antes de presentar)

- [ ] `pip install -r requirements.txt` ejecutado sin errores
- [ ] `streamlit run app.py` abre la interfaz en el navegador
- [ ] Cargaste tu dataset asignado y ejecutaste Limpieza
- [ ] Pestaña 5: Elegiste Target, ejecutaste Predicciones
- [ ] Pestaña 6: Capturas de pantalla de comparativa Modelo A/B + gráfico global + Mejor Modelo
- [ ] Pestaña 7: Captura de gráfico de distribución split y justificación
- [ ] Pestaña 11: Descargaste `Informe_Academico_ML.md`, lo pegaste en Word y lo adaptaste con nombres de integrantes
- [ ] Adjuntas `dataset_limpio.csv` + `X_train/X_test/y_train/y_test.csv` al email/classroom
- [ ] Adjuntas archivo de código fuente (app.py, src/)

---

## 📋 Archivos Entregables

| Archivo | Proporcionar a la docente |
|---|---|
| `Informe impreso + digital (Word/PDF)` | ✅ OBLIGATORIO (usa Pestaña 11) |
| `Carpeta de código fuente (app.py, src/)` | ✅ OBLIGATORIO |
| `requirements.txt` | ✅ OBLIGATORIO |
| `LIBRERIAS_USADAS.md` | ✅ Anexarlo SI la docente pregunta por librerías |
| `dataset_limpio.csv` | ✅ Recomendado anexar |
| `X_train.csv, X_test.csv, y_train.csv, y_test.csv` | ✅ Recomendado anexar |
| `MejorModelo.joblib` (Pestaña 10) | ➕ Extra: Demuestra capacidad de deploy |

---

## 👥 Integrantes del Grupo
*(Edita esta sección con los nombres de tu equipo, 5-8 integrantes)*

1. Integrante 1 - Avalos Serrano Victor Luis
2. Integrante 2 - Nombre Completo
3. Integrante 3 - Nombre Completo
4. Integrante 4 - Nombre Completo
5. Integrante 5 - Nombre Completo
6. Integrante 6 - Nombre Completo

---

## ✅ Verificación Rápida (Antes de entregar)
```bash
# Si todo instaló bien, esto funciona:
python -c "import pandas, numpy, sklearn, streamlit, seaborn, joblib; print('✅ TODAS LAS LIBRERIAS OK')"
```
Muestra **✅ TODAS LAS LIBRERIAS OK** → listo para presentar.

---

> 📬 Dudas: Revisa primero `LIBRERIAS_USADAS.md` para entender cada pieza, luego re-ejecuta el flujo desde la Pestaña 1.
