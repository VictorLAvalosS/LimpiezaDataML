import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    median_absolute_error, mean_absolute_percentage_error, explained_variance_score,
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    roc_auc_score, roc_curve, cohen_kappa_score, jaccard_score
)
import joblib
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    def __init__(self, X_train, X_test, y_train, y_test, area_type):
        self.X_train_raw = X_train.copy()
        self.X_test_raw = X_test.copy()
        self.y_train = y_train
        self.y_test = y_test
        
        self.feature_names_ = []
        self.label_encoders_ = {}
        self._prepare_features()
        
        if "Regresional" in str(area_type) or "Regresión" in str(area_type):
            self.task = "regression"
        else:
            self.task = "classification"
            
        self.scaler = StandardScaler()
        if len(self.feature_names_) > 0:
            self.scaler.fit(self.X_train)
        self.X_train_scaled = self.scaler.transform(self.X_train) if len(self.feature_names_) > 0 else np.array([])
        self.X_test_scaled = self.scaler.transform(self.X_test) if len(self.feature_names_) > 0 else np.array([])
        
        self.models_dict = self._init_models()
        self.param_grids = self._init_param_grids()
        self.trained_models_ = {}
        
    def _prepare_features(self):
        X_tr = self.X_train_raw.copy()
        X_te = self.X_test_raw.copy()
        
        categorical_cols = X_tr.select_dtypes(include=['object', 'string', 'category']).columns.tolist()
        for col in categorical_cols:
            if X_tr[col].nunique(dropna=True) <= 10:
                combined = pd.concat([X_tr[[col]], X_te[[col]]], axis=0)
                dummies = pd.get_dummies(combined, columns=[col], prefix=col, drop_first=False, prefix_sep='_')
                X_tr = pd.concat([X_tr.drop(columns=[col]), dummies.iloc[:len(X_tr)]], axis=1)
                X_te = pd.concat([X_te.drop(columns=[col]), dummies.iloc[len(X_tr):]], axis=1)
            else:
                le = LabelEncoder()
                all_vals = pd.concat([X_tr[col].astype(str), X_te[col].astype(str)])
                le.fit(all_vals)
                X_tr[col] = le.transform(X_tr[col].astype(str))
                X_te[col] = le.transform(X_te[col].astype(str))
                self.label_encoders_[col] = le
        
        for col in X_tr.columns:
            if not pd.api.types.is_numeric_dtype(X_tr[col]):
                X_tr[col] = pd.to_numeric(X_tr[col], errors='coerce')
                X_te[col] = pd.to_numeric(X_te[col], errors='coerce')
        
        num_cols_tr = X_tr.select_dtypes(include=[np.number]).columns
        num_cols_te = X_te.select_dtypes(include=[np.number]).columns
        common_cols = [c for c in num_cols_tr if c in num_cols_te]
        X_tr = X_tr[common_cols].fillna(0)
        X_te = X_te[common_cols].fillna(0)
        
        self.feature_names_ = common_cols
        self.X_train = X_tr[common_cols].values
        self.X_test = X_te[common_cols].values
        
    def _init_models(self):
        return {
            "Regresión Lineal Múltiple": {"model": LinearRegression(), "params": "default", "requires_scale": True, "model_task": "regression", "family": "📁 1. Regresión y Regularización (y ∈ ℝ)"},
            "Regresión Polinómica (Grado 2)": {"model": LinearRegression(), "is_poly": True, "poly_degree": 2, "params": "degree=2", "requires_scale": True, "model_task": "regression", "family": "📁 1. Regresión y Regularización (y ∈ ℝ)"},
            "Regresión Polinómica (Grado 5)": {"model": LinearRegression(), "is_poly": True, "poly_degree": 5, "params": "degree=5", "requires_scale": True, "model_task": "regression", "family": "📁 1. Regresión y Regularización (y ∈ ℝ)"},
            "Regresión Ridge (Regularización L2)": {"model": Ridge(alpha=1.0), "params": "alpha=1.0", "requires_scale": True, "model_task": "regression", "family": "📁 1. Regresión y Regularización (y ∈ ℝ)"},
            "Regresión Lasso (Regularización L1)": {"model": Lasso(alpha=1.0), "params": "alpha=1.0", "requires_scale": True, "model_task": "regression", "family": "📁 1. Regresión y Regularización (y ∈ ℝ)"},
            
            "Regresión Logística Binaria": {"model": LogisticRegression(max_iter=1000), "params": "max_iter=1000", "requires_scale": True, "model_task": "classification", "family": "📁 2. Clasificación Lineal y Logística (y discreta)"},
            "Regresión Softmax (Multiclase)": {"model": LogisticRegression(solver='lbfgs', max_iter=1000), "params": "Softmax Automático (solver='lbfgs')", "requires_scale": True, "model_task": "classification", "family": "📁 2. Clasificación Lineal y Logística (y discreta)"},
            
            "Árbol de Decisión por Índice GINI": {"model": DecisionTreeClassifier(criterion='gini', random_state=42, max_depth=10), "params": "criterion='gini', max_depth=10", "requires_scale": False, "model_task": "classification", "family": "📁 3. Árboles de Decisión (CART / ID3)"},
            "Árbol de Decisión por Entropía": {"model": DecisionTreeClassifier(criterion='entropy', random_state=42, max_depth=10), "params": "criterion='entropy', max_depth=10", "requires_scale": False, "model_task": "classification", "family": "📁 3. Árboles de Decisión (CART / ID3)"},
            "Árbol de Decisión para Regresión": {"model": DecisionTreeRegressor(criterion='squared_error', random_state=42, max_depth=10), "params": "criterion='squared_error', max_depth=10", "requires_scale": False, "model_task": "regression", "family": "📁 3. Árboles de Decisión (CART / ID3)"},
            "Random Forest Clasificador": {"model": RandomForestClassifier(n_estimators=100, random_state=42), "params": "n_estimators=100", "requires_scale": False, "model_task": "classification", "family": "📁 3. Árboles de Decisión (CART / ID3)"},
            "Random Forest Regresor": {"model": RandomForestRegressor(n_estimators=100, random_state=42), "params": "n_estimators=100", "requires_scale": False, "model_task": "regression", "family": "📁 3. Árboles de Decisión (CART / ID3)"},
            
            "SVM Lineal (Clasificación)": {"model": SVC(kernel='linear', probability=True), "params": "kernel='linear'", "requires_scale": True, "model_task": "classification", "family": "📁 4. Máquinas de Vectores de Soporte (SVM / MVS)"},
            "SVM Kernel RBF (Clasificación)": {"model": SVC(kernel='rbf', probability=True), "params": "kernel='rbf'", "requires_scale": True, "model_task": "classification", "family": "📁 4. Máquinas de Vectores de Soporte (SVM / MVS)"},
            "SVM Kernel Polinomial (Clasificación)": {"model": SVC(kernel='poly', probability=True), "params": "kernel='poly'", "requires_scale": True, "model_task": "classification", "family": "📁 4. Máquinas de Vectores de Soporte (SVM / MVS)"},
            "SVM Kernel Sigmoide (Clasificación)": {"model": SVC(kernel='sigmoid', probability=True), "params": "kernel='sigmoid'", "requires_scale": True, "model_task": "classification", "family": "📁 4. Máquinas de Vectores de Soporte (SVM / MVS)"},
            "SVR Lineal (Regresión)": {"model": SVR(kernel='linear'), "params": "kernel='linear'", "requires_scale": True, "model_task": "regression", "family": "📁 4. Máquinas de Vectores de Soporte (SVM / MVS)"},
            "SVR Kernel RBF (Regresión)": {"model": SVR(kernel='rbf'), "params": "kernel='rbf'", "requires_scale": True, "model_task": "regression", "family": "📁 4. Máquinas de Vectores de Soporte (SVM / MVS)"},
            "SVR Kernel Polinomial (Regresión)": {"model": SVR(kernel='poly'), "params": "kernel='poly'", "requires_scale": True, "model_task": "regression", "family": "📁 4. Máquinas de Vectores de Soporte (SVM / MVS)"},
            "SVR Kernel Sigmoide (Regresión)": {"model": SVR(kernel='sigmoid'), "params": "kernel='sigmoid'", "requires_scale": True, "model_task": "regression", "family": "📁 4. Máquinas de Vectores de Soporte (SVM / MVS)"},
            
            "KNN Clasificador": {"model": KNeighborsClassifier(n_neighbors=5), "params": "n_neighbors=5", "requires_scale": True, "model_task": "classification", "family": "📁 5. K-Vecinos más Cercanos (KNN)"},
            "KNN Regresor": {"model": KNeighborsRegressor(n_neighbors=5), "params": "n_neighbors=5", "requires_scale": True, "model_task": "regression", "family": "📁 5. K-Vecinos más Cercanos (KNN)"},
            
            "Perceptrón Multicapa Clasificador": {"model": MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42), "params": "hidden_layer_sizes=(100,)", "requires_scale": True, "model_task": "classification", "family": "📁 6. Redes Neuronales Artificiales (RNA / Perceptrón Multicapa - PMC)"},
            "Perceptrón Multicapa Regresor": {"model": MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=42), "params": "hidden_layer_sizes=(100,)", "requires_scale": True, "model_task": "regression", "family": "📁 6. Redes Neuronales Artificiales (RNA / Perceptrón Multicapa - PMC)"}
        }

    def _init_param_grids(self):
        return {
            "Regresión Ridge (Regularización L2)": {"alpha": [0.1, 1.0, 10.0]},
            "Regresión Lasso (Regularización L1)": {"alpha": [0.01, 0.1, 1.0]},
            "Regresión Logística Binaria": {"C": [0.1, 1.0, 10.0]},
            "Regresión Softmax (Multiclase)": {"C": [0.1, 1.0, 10.0]},
            "Árbol de Decisión por Índice GINI": {"max_depth": [5, 10, 20], "min_samples_split": [2, 5]},
            "Árbol de Decisión por Entropía": {"max_depth": [5, 10, 20], "min_samples_split": [2, 5]},
            "Árbol de Decisión para Regresión": {"max_depth": [5, 10, 20], "min_samples_split": [2, 5]},
            "Random Forest Clasificador": {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20]},
            "Random Forest Regresor": {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20]},
            "SVM Lineal (Clasificación)": {"C": [0.1, 1.0]},
            "SVM Kernel RBF (Clasificación)": {"C": [0.1, 1.0], "gamma": ["scale", "auto"]},
            "SVR Lineal (Regresión)": {"C": [0.1, 1.0]},
            "SVR Kernel RBF (Regresión)": {"C": [0.1, 1.0], "gamma": ["scale", "auto"]},
            "KNN Clasificador": {"n_neighbors": [3, 5, 7]},
            "KNN Regresor": {"n_neighbors": [3, 5, 7]},
            "Perceptrón Multicapa Clasificador": {"hidden_layer_sizes": [(50,), (100,), (50, 50)], "alpha": [0.0001, 0.001, 0.01], "learning_rate_init": [0.001, 0.01]},
            "Perceptrón Multicapa Regresor": {"hidden_layer_sizes": [(50,), (100,), (50, 50)], "alpha": [0.0001, 0.001, 0.01], "learning_rate_init": [0.001, 0.01]}
        }

    def get_available_models(self):
        return list(self.models_dict.keys())

    def _get_transformed_data(self, config):
        requires_scale = config.get("requires_scale", False)
        is_poly = config.get("is_poly", False)
        
        if requires_scale:
            X_tr = self.X_train_scaled
            X_te = self.X_test_scaled
        else:
            X_tr = self.X_train
            X_te = self.X_test
            
        if is_poly:
            poly = PolynomialFeatures(degree=config.get('poly_degree', 2), include_bias=False)
            X_tr = poly.fit_transform(X_tr)
            X_te = poly.transform(X_te)
            
        return X_tr, X_te

    def evaluate_model(self, model_name, use_gridsearch=True, cv_folds=5):
        if model_name not in self.models_dict:
            raise ValueError(f"Modelo {model_name} no compatible o no encontrado.")
            
        config = self.models_dict[model_name]
        base_model = config["model"]
        model_task = config["model_task"]
        
        incompatible = (self.task != model_task)
        
        X_tr, X_te = self._get_transformed_data(config)
        
        y_tr_fit = self.y_train.copy()
        y_te_eval = self.y_test.copy()
        
        if incompatible and model_task == "classification":
            try:
                y_tr_fit = pd.qcut(y_tr_fit, q=3, labels=False, duplicates='drop')
                y_te_eval = pd.qcut(y_te_eval, q=3, labels=False, duplicates='drop')
            except Exception:
                y_tr_fit = y_tr_fit.astype(int)
                y_te_eval = y_te_eval.astype(int)
        
        best_params = None
        if use_gridsearch and model_name in self.param_grids and not incompatible:
            try:
                grid = GridSearchCV(base_model, self.param_grids[model_name], cv=cv_folds, n_jobs=-1, verbose=0)
                grid.fit(X_tr, y_tr_fit)
                model = grid.best_estimator_
                best_params = grid.best_params_
            except Exception:
                model = base_model
                model.fit(X_tr, y_tr_fit)
        else:
            from sklearn.base import clone
            model = clone(base_model)
            model.fit(X_tr, y_tr_fit)
        
        self.trained_models_[model_name] = model
        
        y_pred_test = model.predict(X_te)
        y_pred_train = model.predict(X_tr)
        
        metrics = {}
        train_metrics = {}
        extra_data = {}
        
        if model_task == "regression":
            mse = mean_squared_error(y_te_eval, y_pred_test)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_te_eval, y_pred_test)
            median_ae = median_absolute_error(y_te_eval, y_pred_test)
            try:
                mape = mean_absolute_percentage_error(y_te_eval, y_pred_test) * 100.0
            except Exception:
                mape = np.nan
            r2 = r2_score(y_te_eval, y_pred_test)
            expl_var = explained_variance_score(y_te_eval, y_pred_test)
            metrics = {"R² Score": r2, "Exp. Variance": expl_var, "RMSE": rmse, "MAE": mae, "Median AE": median_ae, "MAPE %": float(mape) if not np.isnan(mape) else 0.0}
            
            mse_tr = mean_squared_error(y_tr_fit, y_pred_train)
            r2_tr = r2_score(y_tr_fit, y_pred_train)
            mae_tr = mean_absolute_error(y_tr_fit, y_pred_train)
            train_metrics = {"R² Score (Train)": r2_tr, "RMSE (Train)": np.sqrt(mse_tr), "MAE (Train)": mae_tr}
            
            extra_data = {"y_pred": y_pred_test, "y_pred_train": y_pred_train,
                          "residuals": np.array(y_te_eval) - y_pred_test,
                          "train_metrics": train_metrics}
            if incompatible:
                extra_data["warning"] = f"Modelo de regresión aplicado a variable discreta (se discretizó)."
        else:
            acc = accuracy_score(y_te_eval, y_pred_test)
            prec = precision_score(y_te_eval, y_pred_test, average='weighted', zero_division=0)
            rec = recall_score(y_te_eval, y_pred_test, average='weighted', zero_division=0)
            f1 = f1_score(y_te_eval, y_pred_test, average='weighted', zero_division=0)
            kappa = cohen_kappa_score(y_te_eval, y_pred_test)
            jaccard = jaccard_score(y_te_eval, y_pred_test, average='weighted', zero_division=0)
            cm = confusion_matrix(y_te_eval, y_pred_test)
            labels = sorted(pd.Series(y_te_eval).unique())
            metrics = {"Accuracy": acc, "F1-Score": f1, "Precision": prec, "Recall": rec, "Cohen Kappa": kappa, "Jaccard (IoU)": jaccard}
            
            acc_tr = accuracy_score(y_tr_fit, y_pred_train)
            f1_tr = f1_score(y_tr_fit, y_pred_train, average='weighted', zero_division=0)
            kappa_tr = cohen_kappa_score(y_tr_fit, y_pred_train)
            train_metrics = {"Accuracy (Train)": acc_tr, "F1-Score (Train)": f1_tr, "Kappa (Train)": kappa_tr}
            
            extra_data = {"cm": cm, "labels": labels, "y_pred": y_pred_test, "y_pred_train": y_pred_train,
                          "train_metrics": train_metrics}
            
            unique_classes = sorted(pd.Series(y_tr_fit).unique().tolist())
            if len(unique_classes) == 2 and hasattr(model, "predict_proba"):
                try:
                    y_proba = model.predict_proba(X_te)[:, 1]
                    roc_auc = roc_auc_score(y_te_eval, y_proba)
                    fpr, tpr, _ = roc_curve(y_te_eval, y_proba)
                    metrics["ROC-AUC"] = roc_auc
                    extra_data["roc_curve"] = (fpr, tpr, roc_auc)
                except Exception:
                    pass
            
            if incompatible:
                extra_data["warning"] = f"Modelo de clasificación aplicado a variable continua (se discretizó)."
        
        cv_scores = None
        if not incompatible and len(X_tr) >= cv_folds * 2:
            try:
                scoring = "r2" if model_task == "regression" else "f1_weighted"
                cv_scores = cross_val_score(model, X_tr, y_tr_fit, cv=cv_folds, scoring=scoring)
                extra_data["cv_scores"] = cv_scores
                extra_data["cv_mean"] = cv_scores.mean()
                extra_data["cv_std"] = cv_scores.std()
            except Exception:
                pass
        
        feature_importance = self._extract_feature_importance(model, model_name, config)
        if feature_importance is not None:
            extra_data["feature_importance"] = feature_importance
        
        if best_params:
            extra_data["best_params"] = best_params
        
        return {
            "name": model_name,
            "metrics": metrics,
            "params": config["params"],
            "extra": extra_data,
            "requires_scale": config.get("requires_scale", False),
            "model_task": model_task,
            "incompatible": incompatible
        }

    def _extract_feature_importance(self, model, model_name, config):
        if len(self.feature_names_) == 0:
            return None
        try:
            if hasattr(model, "feature_importances_"):
                imp = model.feature_importances_
                is_poly = config.get("is_poly", False)
                names = self.feature_names_ if not is_poly else [f"Feat_{i}" for i in range(len(imp))]
                imp_df = pd.DataFrame({"feature": names[:len(imp)], "importance": imp})
                return imp_df.sort_values("importance", ascending=False).head(15)
            if hasattr(model, "coef_"):
                coef = np.abs(model.coef_).flatten()
                is_poly = config.get("is_poly", False)
                names = self.feature_names_ if not is_poly else [f"Feat_{i}" for i in range(len(coef))]
                imp_df = pd.DataFrame({"feature": names[:len(coef)], "importance": coef})
                return imp_df.sort_values("importance", ascending=False).head(15)
        except Exception:
            return None
        return None

    def train_all_and_compare(self):
        results = []
        for name, config in self.models_dict.items():
            if config["model_task"] == self.task:
                try:
                    res = self.evaluate_model(name)
                    results.append(res)
                except Exception:
                    pass
                    
        if self.task == "regression":
            results.sort(key=lambda x: x["metrics"]["R² Score"], reverse=True)
        else:
            results.sort(key=lambda x: x["metrics"].get("F1-Score", x["metrics"].get("Accuracy", 0)), reverse=True)
            
        return results

    def evaluate_six_models(self, selected_models=None):
        results = []
        for name, config in self.models_dict.items():
            if config["model_task"] == self.task:
                try:
                    res = self.evaluate_model(name)
                    if not res.get("incompatible", False):
                        results.append(res)
                except Exception:
                    pass

        if self.task == "regression":
            results.sort(key=lambda x: x["metrics"]["R² Score"], reverse=True)
        else:
            results.sort(key=lambda x: x["metrics"].get("F1-Score", x["metrics"].get("Accuracy", 0)), reverse=True)

        return results

    def predict_single(self, model_name, input_dict):
        if model_name not in self.trained_models_:
            self.evaluate_model(model_name)
        if model_name not in self.trained_models_:
            return None
            
        model = self.trained_models_[model_name]
        config = self.models_dict[model_name]
        
        X_df = pd.DataFrame([input_dict])
        for col in self.feature_names_:
            if col not in X_df.columns:
                if col in self.label_encoders_:
                    X_df[col] = 0
                else:
                    base = col.split('_')[0]
                    if base in X_df.columns:
                        val = X_df[base].iloc[0]
                        suffix = '_'.join(col.split('_')[1:])
                        X_df[col] = 1 if str(val) == suffix else 0
                    else:
                        X_df[col] = 0
        X = X_df[self.feature_names_].fillna(0).values
        
        requires_scale = config.get("requires_scale", False)
        is_poly = config.get("is_poly", False)
        
        if requires_scale:
            X = self.scaler.transform(X)
        if is_poly:
            poly = PolynomialFeatures(degree=config.get('poly_degree', 2), include_bias=False)
            X = poly.fit_transform(X)
        
        pred = model.predict(X)
        proba = None
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(X)[0]
            except Exception:
                pass
        return pred[0], proba

    def export_model(self, model_name, path):
        if model_name not in self.trained_models_:
            self.evaluate_model(model_name)
        bundle = {
            "model": self.trained_models_.get(model_name),
            "config": self.models_dict.get(model_name),
            "feature_names": self.feature_names_,
            "scaler": self.scaler,
            "label_encoders": self.label_encoders_,
            "task": self.task,
            "model_name": model_name
        }
        joblib.dump(bundle, path)
        return path
