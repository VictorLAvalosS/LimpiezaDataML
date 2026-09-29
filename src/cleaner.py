import pandas as pd
import numpy as np
import re
from datetime import datetime

class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        self.df_raw = df.copy()
        self.df = df.copy()
        self.audit_log = []
        self.cell_audit_log = []
        self.tukey_limits = {}
        self.excluded_cols = []

    def set_excluded_cols(self, cols: list):
        self.excluded_cols = cols

    def is_excluded(self, col: str) -> bool:
        return col in self.excluded_cols

    def log(self, action: str, details: str, theoretical_justification: str):
        self.audit_log.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "action": action,
            "details": details,
            "justification": theoretical_justification
        })

    def log_cell_changes(self, mask, col_name, before_series, after_series, reason):
        if not mask.any():
            return
        idx_changed = mask[mask].index
        
        for idx in idx_changed:
            val_orig = before_series.loc[idx]
            val_new = after_series.loc[idx]
            
            if pd.isna(val_orig) and pd.isna(val_new):
                continue
                
            self.cell_audit_log.append({
                "# Fila Dataset": idx + 1,
                "# Fila Excel": idx + 2,
                "Columna": col_name,
                "Valor Original Sucio": str(val_orig) if pd.notna(val_orig) else "NaN",
                "Valor Limpio Corregido": str(val_new) if pd.notna(val_new) else "NaN",
                "Tipo de Error": reason
            })

    def infer_ml_area(self, target_col):
        if not target_col or target_col not in self.df.columns:
            return None, None
        s = self.df[target_col].dropna()
        if pd.api.types.is_numeric_dtype(s) and (pd.api.types.is_float_dtype(s) or s.nunique() > 10):
            return "Modelos Regresionales", "Variable continua y ∈ ℝ. Minimización de error cuadrático MSE."
        else:
            return "Modelos de Árboles de Decisión", "Variable ordinal/discreta con clases finitas. Segmentación por Gini/Entropía."

    def auto_diagnose(self) -> pd.DataFrame:
        diagnosis = []
        
        dups = self.df.duplicated().sum()
        if dups > 0:
            diagnosis.append({
                "Problema": f"Filas duplicadas detectadas ({dups})",
                "Columna": "Dataset Completo",
                "Recomendación": "Desduplicar para evitar sesgos en el entrenamiento."
            })
            
        for col in self.df.columns:
            pct = self.df[col].isna().mean()
            if pct > 0.05:
                rec = "Descartar columna (>60% nulos)." if pct > 0.6 else "Imputar nulos según tipo (Media/Mediana/Moda)."
                diagnosis.append({
                    "Problema": f"Alto % de nulos ({pct*100:.1f}%)",
                    "Columna": col,
                    "Recomendación": rec
                })
                
        for col in self.df.select_dtypes(include=['object', 'string']).columns:
            if self.df[col].astype(str).str.contains(r'[ÃÕÂ`´â€™]', regex=True).any():
                diagnosis.append({
                    "Problema": "Caracteres corruptos/Mojibake detectados",
                    "Columna": col,
                    "Recomendación": "Reparar codificación textual y normalizar apóstrofes."
                })

            try:
                cleaned = self.df[col].astype(str).str.replace(r'[\$,]', '', regex=True)
                if cleaned.str.match(r'^[+-]?\d*\.?\d+$').mean() > 0.3:
                    diagnosis.append({
                        "Problema": "Tipo mixto o caracteres residuales (e.g. moneda)",
                        "Columna": col,
                        "Recomendación": "Limpiar caracteres de texto y convertir a numérico."
                    })
            except Exception:
                pass
                
        for col in self.df.select_dtypes(include=[np.number]).columns:
            q1 = self.df[col].quantile(0.25)
            q3 = self.df[col].quantile(0.75)
            iqr = q3 - q1
            lb = q1 - (1.5 * iqr)
            ub = q3 + (1.5 * iqr)
            if ((self.df[col] < lb) | (self.df[col] > ub)).any():
                diagnosis.append({
                    "Problema": "Presencia de valores atípicos severos (Outliers)",
                    "Columna": col,
                    "Recomendación": "Tratar con Winsorización (Capping) IQR."
                })
                
        if not diagnosis:
            diagnosis.append({"Problema": "Ninguno", "Columna": "Todas", "Recomendación": "Dataset en excelentes condiciones."})
            
        return pd.DataFrame(diagnosis)

    def normalize_columns(self, target_col=None):
        original_cols = self.df.columns.tolist()
        new_cols = []
        for col in original_cols:
            if self.is_excluded(col) or col == target_col:
                new_cols.append(col)
                continue
            c = str(col).lower().strip()
            c = re.sub(r'\s+', '_', c)
            c = re.sub(r'[^\w\s]', '', c)
            new_cols.append(c)
        
        self.df.columns = new_cols
        
        changes = [f"'{o}' -> '{n}'" for o, n in zip(original_cols, new_cols) if o != n]
        if changes:
            self.log("Normalización Nombres", f"{len(changes)} cols modificadas.", "Estandarización snake_case.")
            
        return new_cols

    def deduplicate(self):
        initial_rows = len(self.df)
        self.df = self.df.drop_duplicates()
        final_rows = len(self.df)
        if initial_rows != final_rows:
            self.log("Desduplicación", f"{initial_rows - final_rows} filas duplicadas eliminadas.", "Prevención de sesgos (overfitting).")

    def repair_text_encoding(self, target_col=None):
        mojibake_dict = {
            'Ã•': "'", 
            'Õ': "'", 
            'â€™': "'", 
            'Ã¢â‚¬â„¢': "'", 
            '`': "'", 
            '´': "'",
            'Ã©': 'é', 
            'Ã¡': 'á', 
            'Ã³': 'ó', 
            'Ãº': 'ú', 
            'Ã±': 'ñ', 
            'Ã': 'í', 
            'Â': '', 
            'Ã”': '"', 
            'Ã“': '"'
        }
        cols_repaired = []
        for col in self.df.select_dtypes(include=['object', 'string']).columns:
            before = self.df[col].copy()
            na_mask = self.df[col].isna()
            try:
                temp_col = self.df[col].astype(object).astype(str)
            except Exception:
                temp_col = self.df[col].map(lambda x: str(x) if pd.notna(x) else 'nan')
            for bad_char, good_char in mojibake_dict.items():
                temp_col = temp_col.str.replace(bad_char, good_char, regex=False)
            
            temp_col = temp_col.str.replace(r'[\x00-\x1F\x7F-\x9F]', '', regex=True)
            temp_col = temp_col.where(~na_mask, np.nan)
            
            try:
                self.df[col] = temp_col
            except Exception:
                self.df[col] = temp_col.astype(object)
            
            mask = ~(before == self.df[col]) & ~(before.isna() & self.df[col].isna())
            corrupted_count = mask.sum()
            
            if corrupted_count > 0:
                self.log_cell_changes(mask, col, before, self.df[col], "Ruido Léxico / Mojibake")
                cols_repaired.append(f"{col} ({corrupted_count} filas)")
                
        if cols_repaired:
            self.log("Reparación Mojibake", f"Corregidos: {', '.join(cols_repaired)}.", "Limpieza léxica de caracteres corruptos por error de encoding.")

    def sanitize_and_convert(self, target_col=None):
        null_tokens = ['n/a', 'null', 'nan', '?', '-', '']
        changed = []
        for col in self.df.select_dtypes(include=['object', 'string']).columns:
            if self.is_excluded(col) or col == target_col: continue
            
            before = self.df[col].copy()
            
            self.df[col] = self.df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
            self.df[col] = self.df[col].replace(
                {tok: np.nan for tok in null_tokens + [tok.upper() for tok in null_tokens] + [tok.capitalize() for tok in null_tokens]}
            )
            
            try:
                try:
                    cleaned_series = self.df[col].astype(object).astype(str).str.replace(r'[\$,]', '', regex=True)
                except Exception:
                    cleaned_series = self.df[col].map(lambda x: str(x).replace('$', '').replace(',', '') if pd.notna(x) else x)
                if cleaned_series.str.match(r'^[+-]?\d*\.?\d+$').any():
                    converted = pd.to_numeric(cleaned_series, errors='ignore')
                    if converted.dtype != 'object':
                        self.df[col] = converted
                elif cleaned_series.str.match(r'^\d{2,4}[-/]\d{1,2}[-/]\d{1,4}').any():
                    converted_date = pd.to_datetime(self.df[col], errors='ignore')
                    if np.issubdtype(converted_date.dtype, np.datetime64):
                        self.df[col] = converted_date.dt.strftime('%Y-%m-%d')
            except Exception: pass
            
            mask = ~(before == self.df[col]) & ~(before.isna() & self.df[col].isna())
            if mask.sum() > 0:
                self.log_cell_changes(mask, col, before, self.df[col], "Saneamiento de Formato (Números/Fechas/Texto)")
                changed.append(col)
            
        if changed:
            self.log("Saneamiento y Conversión", f"Procesadas {len(changed)} columnas.", "Homogeneizar nulos y castear strings numéricos.")

    def treat_nulls(self, target_col=None, drop_threshold=0.6, num_strategy='median', cat_strategy='mode', cols_to_impute=None):
        if target_col and target_col in self.df.columns:
            nulos = self.df[target_col].isna().sum()
            if nulos > 0:
                self.df = self.df.dropna(subset=[target_col])
                self.log("Protección Target", f"Eliminadas {nulos} filas nulas de '{target_col}'.", "No imputar etiqueta objetivo para evitar sesgos.")

        thresh_count = len(self.df) * (1 - drop_threshold)
        cols_before = set(self.df.columns)
        for col in list(self.df.columns):
            if self.is_excluded(col) or col == target_col: continue
            if self.df[col].count() < thresh_count:
                self.df = self.df.drop(columns=[col])
                
        dropped = cols_before - set(self.df.columns)
        if dropped:
            self.log("Descarte por Nulidad", f"Eliminadas: {', '.join(dropped)}", "Falta de información severa (>60%).")

        if cols_to_impute is None:
            cols_to_impute = self.df.columns.tolist()

        c_imputed = []
        for col in cols_to_impute:
            if self.is_excluded(col) or col == target_col or col not in self.df.columns: continue
            
            before = self.df[col].copy()
            nulos_col = self.df[col].isna().sum()
            
            if nulos_col > 0:
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    val = self.df[col].median() if num_strategy == 'median' else self.df[col].mean()
                    self.df[col] = self.df[col].fillna(val)
                    c_imputed.append(col)
                else:
                    val = self.df[col].mode()[0] if not self.df[col].mode().empty and cat_strategy == 'mode' else 'Faltante'
                    self.df[col] = self.df[col].fillna(val)
                    c_imputed.append(col)
                    
                mask = ~(before == self.df[col]) & ~(before.isna() & self.df[col].isna())
                if mask.sum() > 0:
                    self.log_cell_changes(mask, col, before, self.df[col], f"Imputación de Nulos ({num_strategy if pd.api.types.is_numeric_dtype(self.df[col]) else cat_strategy})")
                    
        if c_imputed:
            self.log("Imputación", f"Variables: {', '.join(c_imputed)}", "Rellenar faltantes sin tocar Target.")

    def treat_outliers(self, multiplier=1.5, target_col=None, cols_to_treat=None):
        if cols_to_treat is None:
            cols_to_treat = self.df.select_dtypes(include=[np.number]).columns.tolist()
            
        treated = []
        for col in cols_to_treat:
            # Protección estricta del target
            if self.is_excluded(col) or str(col).strip().lower() == str(target_col).strip().lower() or col not in self.df.columns: continue
            
            before = self.df[col].copy()
            if pd.api.types.is_numeric_dtype(self.df[col]):
                q1 = self.df[col].quantile(0.25)
                q3 = self.df[col].quantile(0.75)
                iqr = q3 - q1
                lb = q1 - (multiplier * iqr)
                ub = q3 + (multiplier * iqr)
                
                # Calcular outliers y guardar límites para TODA variable predictora procesada
                outliers_mask = (self.df[col] < lb) | (self.df[col] > ub)
                outliers_count = outliers_mask.sum()
                self.tukey_limits[col] = {'lim_inf': lb, 'lim_sup': ub, 'total_outliers': int(outliers_count)}
                
                if outliers_count > 0:
                    self.df[col] = np.clip(self.df[col], lb, ub)
                    treated.append(col)
                    
                    mask = ~(before == self.df[col]) & ~(before.isna() & self.df[col].isna())
                    if mask.sum() > 0:
                        self.log_cell_changes(mask, col, before, self.df[col], f"Winsorización Outliers (Capping IQR)")
                        
        if treated:
            self.log("Tratamiento Outliers", f"Winsorización en {len(treated)} variables.", "Mitigar atípicos con Capping IQR sin alterar etiqueta.")

    def restore_columns_to_raw(self, cols_to_restore: list):
        for col in cols_to_restore:
            if col in self.df_raw.columns:
                self.df[col] = self.df_raw[col]
                self.log("Restauración Manual", f"La columna '{col}' fue restaurada a su estado original.", "Decisión técnica del usuario.")

    def encode_categorical(self, target_col=None, max_categories_onehot=10):
        """
        Codifica variables categóricas:
        - One-Hot Encoding para variables con <= max_categories_onehot categorías
        - Label Encoding para variables con más categorías
        Protege la columna target de cualquier modificación.
        Compatible con pandas 2.x y pandas 3.x (evita TypeError con StringDtype).
        """
        from sklearn.preprocessing import LabelEncoder
        
        categorical_cols = self.df.select_dtypes(include=['object', 'string', 'category']).columns.tolist()
        if target_col and target_col in categorical_cols:
            categorical_cols = [c for c in categorical_cols if c != target_col]
            
        encoded_cols_info = []
        onehot_cols = []
        label_cols = []
        
        for col in categorical_cols:
            if self.is_excluded(col):
                continue
            n_unique = self.df[col].nunique(dropna=True)
            if n_unique <= max_categories_onehot and n_unique > 1:
                onehot_cols.append(col)
            elif n_unique > 1:
                label_cols.append(col)
        
        if onehot_cols:
            before_shape = self.df.shape
            try:
                self.df = pd.get_dummies(self.df, columns=onehot_cols, drop_first=False, prefix_sep='_', dtype=int)
            except Exception:
                self.df = pd.get_dummies(self.df, columns=onehot_cols, drop_first=False, prefix_sep='_')
            encoded_cols_info.append(f"One-Hot: {', '.join(onehot_cols)} ({len(onehot_cols)} vars → +{max(0, self.df.shape[1] - before_shape[1])} cols nuevas)")
            
        if label_cols:
            for col in label_cols:
                le = LabelEncoder()
                mask_notna = self.df[col].notna()
                if mask_notna.any():
                    try:
                        valores = self.df.loc[mask_notna, col].astype(object).astype(str).values
                        codificados = le.fit_transform(valores)
                        self.df[col] = self.df[col].astype(object)
                        self.df.loc[mask_notna, col] = codificados
                        self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                    except Exception as e:
                        try:
                            series_obj = self.df[col].astype(object)
                            vals_encoded = pd.Series(index=self.df.index, dtype=float)
                            vals_encoded[mask_notna] = le.fit_transform(series_obj[mask_notna].astype(str).values)
                            self.df[col] = vals_encoded
                        except Exception:
                            encoded_cols_info.append(f"SKIP Label: '{col}' (no se pudo codificar)")
                            continue
                else:
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
            encoded_cols_info.append(f"Label Encoding: {', '.join(label_cols)}")
            
        if encoded_cols_info:
            self.log("Codificación Categórica", " | ".join(encoded_cols_info), 
                     "Convertir variables nominales/ordinales a formato numérico compatible con modelos ML.")
            
        return encoded_cols_info

    def run_pipeline(self, 
                     excluded_cols=[], 
                     restored_cols=[],
                     target_col=None,
                     do_norm=True, 
                     do_dedup=True, 
                     do_mojibake=True,
                     do_sanit=True,
                     do_nulls=True, null_thresh=0.6, num_strat='median', cat_strat='mode', cols_impute=None,
                     do_outliers=True, iqr_mult=1.5, cols_outliers=None,
                     do_encode_categorical=True):
                     
        self.set_excluded_cols(excluded_cols)
        self.cell_audit_log = [] 
        
        if do_norm:
            self.normalize_columns(target_col=target_col)
            
        if do_dedup:
            self.deduplicate()
            
        if do_mojibake:
            self.repair_text_encoding(target_col=target_col)
            
        if do_sanit:
            self.sanitize_and_convert(target_col=target_col)
            
        if do_nulls:
            self.treat_nulls(target_col=target_col, drop_threshold=null_thresh, 
                             num_strategy=num_strat, cat_strategy=cat_strat, 
                             cols_to_impute=cols_impute)
                             
        if do_outliers:
            self.treat_outliers(multiplier=iqr_mult, target_col=target_col, cols_to_treat=cols_outliers)
            
        if do_dedup:
            self.deduplicate()
            
        if restored_cols:
            self.restore_columns_to_raw(restored_cols)
            
        if do_encode_categorical:
            self.encode_categorical(target_col=target_col)
            
        return self.df, self.audit_log, self.cell_audit_log, self.tukey_limits
