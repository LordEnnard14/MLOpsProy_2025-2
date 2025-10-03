# Telco Customer Churn - MLOps Pipeline 🚀

El presente proyecto implementa un pipeline de MLOps que permite entrenar, versionar, monitorear y desplegar un modelo de clasificación para predecir la cancelación de clientes de una empresa de telecomunicaciones.

## 📁 Estructura del Repositorio

```
├── data/
│   ├── raw/                    # Dataset original
│   └── processed/              # Dataset limpio
├── src/
│   ├── features.py             # Script para Preprocesamiento de datos
│   ├── train.py                # Script para Entrenamiento y log de modelos en MLflow
│   └── service/
│       └── serve.py           # Despliegue del modelo con FastAPI
├── artifacts/                  # Métricas y modelo entrenado
├── mlruns/                     # Registro local de experimentos MLflow
├── dvc.yaml / dvc.lock         # Pipeline de DVC
├── mlflow.db                   # Base de datos de tracking MLflow
└── README.md                   # Documentación markdown
```

---

## ⚙️ Inicialización del entorno

En git bash
# Crear repositorio Git
git init

# Crear y activar entorno virtual
python -m venv .venv
source .venv/Scripts/activate   # Windows (Git Bash)

# Instalar dependencias
pip install -U pip
pip install "dvc[ssh]" scikit-learn pandas mlflow fastapi uvicorn[standard] pydantic pandera pytest
```

---

## 📦 Versionamiento de datos con DVC

```bash
# Inicializar DVC
dvc init

# Agregar dataset original
dvc add data/raw/Telco-Customer-Churn.csv

# Agregar a Git
git add data/raw/Telco-Customer-Churn.csv.dvc .dvcignore
git commit -m "Versionando dataset Telco con DVC"
```

---

## 🧹 Preprocesamiento (features.py)

```bash
# Ejecución manual del script
python src/features.py --infile data/raw/Telco-Customer-Churn.csv --outfile data/processed/telco_clean.csv

# Añadir como etapa del pipeline
dvc stage add -n preprocess \
  -d src/features.py -d data/raw/Telco-Customer-Churn.csv \
  -o data/processed/telco_clean.csv \
  python src/features.py --infile data/raw/Telco-Customer-Churn.csv --outfile data/processed/telco_clean.csv

git add dvc.yaml data/processed/.gitignore
git commit -m "Añadir etapa de preprocesamiento al pipeline"
```

---

## 🤖 Entrenamiento y log de modelo con MLflow

```bash
# Entrenamiento manual
python src/train.py --data data/processed/telco_clean.csv --out artifacts

# Añadir como etapa DVC
dvc stage add -n train \
  -d src/train.py -d data/processed/telco_clean.csv \
  -o artifacts/metrics.json -o artifacts/model.pkl \
  python src/train.py --data data/processed/telco_clean.csv --out artifacts

git add dvc.yaml dvc.lock
git commit -m "Añadir etapa de entrenamiento al pipeline"
```

---

## 🔁 Ejecutar pipeline completo

```bash
# Ejecutar el pipeline completo

dvc repro
```

---

## 📊 Visualización con MLflow UI

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Ingresar en el navegador a:

```
(Ingresar con el url que aparece con el comando anterior)
```

Promover modelo a producción:

* Desactiva el "New Model Registry UI"
* Promueve la mejor versión a **Production** (PRD)

---

## 🚀 Despliegue con FastAPI

```bash
# Ejecutar el servidor de predicción
uvicorn src.service.serve:app --reload
```

Ingresar a Swagger UI en:

```
http://127.0.0.1:8000/docs
```

### 📥 Ejemplo de entrada JSON:

```json
{
  "gender": "Male",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 70.5,
  "TotalCharges": 850.6
}
```

### 📤 Respuesta esperada:

```json
{
  "prediction": 1
}
```

---
🌱 Ejemplo de flujo
# Cambiar a develop
git checkout develop

# Hacer cambios en features.py o train.py
nano src/features.py

# Ejecutar pipeline y actualizar artefactos
dvc repro

# Preparar commit
git add src/features.py src/train.py dvc.yaml dvc.lock
git commit -m "feat: actualización de preprocesamiento y entrenamiento"
git push origin develop


Mostrar resultados en MLflow UI y probar el servicio en FastAPI.

Cuando la versión es estable:

# Pasar a rama main
git checkout main

# Fusionar cambios desde develop
git merge develop

# Subir cambios estables
git push origin main


## ✅ Criterios de Éxito (Resumen)

* ✔️ Dataset versionado con DVC
* ✔️ Pipeline automatizado (`dvc repro`)
* ✔️ Registro de métricas y modelos con MLflow
* ✔️ Promoción de modelos a producción (Model Registry)
* ✔️ Despliegue funcional con FastAPI (`/predict`)

---

---

> Proyecto elaborado como práctica de DevOps/MLOps - Universidad de Lima, 2025

---