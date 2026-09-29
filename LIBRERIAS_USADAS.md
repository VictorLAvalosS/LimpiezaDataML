# 📚 LIBRERÍAS UTILIZADAS EN EL PROYECTO
## Ciencia de Datos y Machine Learning — Lic. Patricia Rodríguez Bilbao

---

## 📋 RESUMEN EJECUTIVO
Este proyecto de Machine Learning académico se apoya en un stack **100% Python** de código abierto, alineado con el ecosistema estándar de la industria y el plan de estudios universitario. A continuación, se detalla cada dependencia, su propósito técnico y su justificación pedagógica dentro de la práctica.

---

## 🔢 1. PROCESAMIENTO NUMÉRICO Y ETL (Extract, Transform, Load)

### 1.1 **pandas**  — versión actual en `requirements.txt`
- **Categoría:** Manipulación y Análisis de Datos Estructurados (DataFrames)
- **Propósito en el proyecto:**
  - Carga agnóstica de datasets CSV/XLSX desde [app.py L20-L35](file:///c:/Users/User/Downloads/DML/Limpieza%20Data%20Set/app.py#L20-L35)
  - Auditoría de nulos, duplicados y tipos de dato en [cleaner.py L59-L115](file:///c:/Users/User/Downloads/DML/Limpieza%20Data%20Set/src/cleaner.py#L59-L115)
  - Pipeline de limpieza: imputación, One-Hot Encoding, Label Encoding.
  - Construcción de matrices X/y para los modelos de ML.
- **Justificación académica:** Librería *de facto* para Data Science; enseña la estructura tabular (filas/columnas) y operaciones vectorizadas necesarias antes de cualquier modelo.

---

### 1.2 **numpy**
- **Categoría:** Álgebra Lineal y Computación Numérica
- **Propósito en el proyecto:**
  - Cálculo de límites IQR para detección de Outliers: [cleaner.py L275-L284](file:///c:/Users/User/Downloads/DML/Limpieza%20Data%20Set/src/cleaner.py#L275-L284)
  - `np.clip()` para Winsorización (Capping) de valores atípicos.
  - Soporte de arrays para matrices de features de scikit-learn.
  - Cálculo de RMSE = √MSE y operaciones de broadcasting.
- **Justificación académica:** Base computacional de todo el ecosistema ML; demuestra la naturaleza vectorizada de los cálculos detrás de regresiones y métricas.

---

### 1.3 **scipy**
- **Categoría:** Cálculo Científico Avanzado
- **Propósito en el proyecto:**
  - Motor estadístico subyacente a pandas y scikit-learn (distribuciones, tests estadísticos implícitos en GridSearchCV).
  - Optimización numérica utilizada internamente por `LinearRegression`, `SVC`, `MLP`.
- **Justificación académica:** Fundamentación matemática: solvers de matrices, descenso de gradiente, validación de significancia estadística.

---

### 1.4 **openpyxl**
- **Categoría:** Interfaz con formato Excel
- **Propósito en el proyecto:**
  - Permite a `pd.read_excel()` cargar archivos `.xlsx`/`.xls` sin perder información.
  - Compatible con la carga de datasets entregados por la docente en formato Excel.
- **Justificación académica:** En entornos empresariales y universitarios, los datos frecuentemente vienen en Excel, no solo CSV.

---

## 🤖 2. MOTOR DE MACHINE LEARNING

### 2.1 **scikit-learn**  (Núcleo del Proyecto)
- **Categoría:** Librería Estándar de ML Clásico Supervisado
- **Propósito en el proyecto (core):**
  | Módulo | Aplicación en nuestro sistema |
  |--------|-------------------------------|
  | `sklearn.model_selection.train_test_split` | División 80/20 Train/Test con stratify |
  | `sklearn.model_selection.GridSearchCV` | Optimización de hiperparámetros |
  | `sklearn.model_selection.cross_val_score` | Validación Cruzada 5-Fold |
  | **MODELOS REGRESIONALES** | `LinearRegression`, `PolynomialFeatures`, `Ridge`, `Lasso`, `DecisionTreeRegressor`, `RandomForestRegressor`, `SVR`, `KNeighborsRegressor`, `MLPRegressor` |
  | **ÁRBOLES DE DECISIÓN y CLASIFICADORES** | `DecisionTreeClassifier` (Gini/Entropía), `RandomForestClassifier`, `LogisticRegression`, `SVC` (4 kernels), `KNeighborsClassifier`, `MLPClassifier` |
  | **PREPROCESAMIENTO** | `StandardScaler` (Z-Score), `LabelEncoder`, `PolynomialFeatures` |
  | **MÉTRICAS** | `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `confusion_matrix`, `roc_auc_score`, `roc_curve`, `r2_score`, `mean_squared_error`, `mean_absolute_error` |
- **Justificación académica:** Implementación didáctica y uniforme de todos los algoritmos vistos en clase; homogeneiza `fit()/predict()` permitiendo comparar modelos de familias distintas con las mismas métricas. Cumple EXACTAMENTE con las áreas del programa: **a) Modelos Regresionales** y **b) Modelos de Árboles de Decisión**.

---

## 📊 3. VISUALIZACIÓN ESTADÍSTICA

### 3.1 **matplotlib**
- **Categoría:** Motor Base de Gráficos 2D
- **Propósito en el proyecto:**
  - `plt.subplots()` → gráficos de barras comparativos de métricas globales
  - Matriz de Confusión (Heatmap axes), Diagrama de Dispersión Real vs Predicho, Curva ROC, Histogramas de Residuales.
- **Justificación académica:** Librería base de visualización; demostrar que podemos customizar cada elemento de un plot (etiquetas, títulos, colores) según normativa de informe.

---

### 3.2 **seaborn**
- **Categoría:** Visualización Estadística Interfaz de Alto Nivel
- **Propósito en el proyecto:**
  - `seaborn.heatmap()` → Matrices de Confusión con anotaciones numéricas y cmap='Blues'
  - `seaborn.barplot()` → Importancia de Características ordenada + paleta viridis
  - `seaborn.scatterplot()` / `lineplot()` → Análisis exploratorio cruzado
- **Justificación académica:** Simplifica la creación de plots estadísticos "publication-ready" tal como se esperan en un informe universitario final.

---

## 💻 4. INTERFAZ GRÁFICA Y DEPLOYMENT INTERACTIVO

### 4.1 **streamlit**
- **Categoría:** Framework Web Reactivo para Data Apps
- **Propósito en el proyecto:**
  - UI completa del sistema: 11 pestañas interactivas.
  - Widgets: `st.file_uploader()` (carga dataset), `st.selectbox()`, `st.multiselect()`, `st.form()`, `st.metric()`, `st.data_editor()` (limpieza manual), `st.tabs()`.
  - Persistencia de estado con `st.session_state` → avance incremental (Primer Avance → Segundo Avance → Informe Final).
  - Botones de descarga de CSV (Train/Test split, dataset limpio, auditoría) y modelos .joblib.
- **Justificación académica:** Convierte un script estático `.py` en un entregable profesional e interactivo que la docente puede ejecutar y manipular sin necesidad de escribir código; demuestra despliegue de producto ML mínimo viable.

---

## 🌐 5. TRADUCCIÓN Y ENRIQUECIMIENTO IA (OPCIONAL)

### 5.1 **deep-translator**
- **Categoría:** Traducción automática multi-servicio
- **Propósito en el proyecto:**
  - Toggle 🌐 Traducir en [app.py L56-L64](file:///c:/Users/User/Downloads/DML/Limpieza%20Data%20Set/app.py#L56-L64)
  - Traduce nombres de columnas y valores categóricos de inglés → español automáticamente si el dataset viene en inglés.
- **Justificación académica:** Muchos datasets públicos vienen en inglés; esta dependencia demuestra adaptabilidad a formatos internacionales y alineación con hispanohablantes.

---

### 5.2 **google-generativeai** (Gemini API — Desactivable)
- **Categoría:** Interfaz con LLM Google Gemini
- **Propósito en el proyecto:**
  - Antes desactivado; en esta versión protegido mediante `python-dotenv` + carga condicional [app.py L17-L21](file:///c:/Users/User/Downloads/DML/Limpieza%20Data%20Set/app.py#L17-L21).
  - Solo se activa si el usuario configura `GEMINI_API_KEY` en `.env`.
- **Justificación académica:** Módulo de IA Generativa opcional para consultas en lenguaje natural sobre el dataset. Se incluye como **valor agregado** pero NO es requisito obligatorio para la ejecución del pipeline.

---

## 🔒 6. GESTIÓN DE CONFIGURACIÓN Y SERIALIZACIÓN

### 6.1 **python-dotenv**
- **Categoría:** Gestión segura de Secretos y Variables de Entorno
- **Propósito en el proyecto:**
  - `load_dotenv()` → lee `GEMINI_API_KEY` desde un archivo `.env` privado.
  - **NUNCA** se commitea la API Key al repositorio (buena práctica DevSecOps).
- **Justificación académica:** Enseña seguridad básica en proyectos de software: separar credenciales del código fuente. Corrige el riesgo inicial de tener la key hardcodeada.

---

### 6.2 **joblib**
- **Categoría:** Serialización Eficiente de Objetos Numéricos / Modelos ML
- **Propósito en el proyecto:**
  - `joblib.dump(bundle, path)` en [models.py L371-L384](file:///c:/Users/User/Downloads/DML/Limpieza%20Data%20Set/src/models.py#L371-L384) → exporta modelo entrenado + scaler + encoders en un único archivo `.joblib`.
  - Permite reutilizar el modelo entrenado en otros entornos sin reentrenar.
- **Justificación académica:** Representa la etapa de **deploy** o puesta en producción del modelo; pieza fundamental del ciclo de vida completo de un proyecto ML (CRISP-DM).

---

## 📊 CUADRO COMPARATIVO RÁPIDO (Preguntas docente)

| # | Librería | ¿Para qué la usamos? | Tipo |
|---|----------|-----------------------|------|
| 1 | **pandas** | Cargar, limpiar y manipular tablas (DataFrames) | ETL / Limpieza |
| 2 | **numpy** | Cálculos numéricos, IQR, arrays numéricos | Álgebra |
| 3 | **scipy** | Optimizadores y estadística para scikit-learn | Cálculo Científico |
| 4 | **openpyxl** | Leer archivos Excel (.xlsx) de la docente | Formato |
| 5 | **scikit-learn** | ❗ MOTOR PRINCIPAL: TODOS los modelos (regresión, árboles, SVM, KNN, RNA), GridSearchCV, Validación Cruzada y métricas (Accuracy, R², ROC-AUC) | Machine Learning |
| 6 | **matplotlib** | Crear gráficos base (barras, dispersión, ROC, residuales) | Visualización |
| 7 | **seaborn** | Heatmaps de la Matriz de Confusión e Importancia de Variables | Visualización Estadística |
| 8 | **streamlit** | Interfaz gráfica web de 11 pestañas para interactuar sin código | UI / Dashboard |
| 9 | **deep-translator** | Traducir columnas/valores de inglés a español | Internacionalización |
| 10 | **google-generativeai** | Conexión opcional a Gemini para chat IA | AI Generativa (OPCIONAL) |
| 11 | **python-dotenv** | Guardar la API Key en archivo .env sin exponerla | Seguridad / Config |
| 12 | **joblib** | Guardar y cargar modelos ML ya entrenados | Serialización / Deploy |

---

## ✅ INSTALACIÓN (Verificación rápida para la docente)
Todas las librerías se instalan con una sola línea, a partir del archivo `requirements.txt`:
```bash
pip install -r requirements.txt
```
