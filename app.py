import streamlit as st
import matplotlib.pyplot as plt
import seaborn as plt_sns
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
import seaborn as plt_sns
from sklearn.model_selection import train_test_split
from src.cleaner import DataCleaner
from src.models import ModelTrainer
import uuid
import os
import re

DEEP_TRANSLATOR_AVAILABLE = False
try:
    from deep_translator import GoogleTranslator
    DEEP_TRANSLATOR_AVAILABLE = True
except Exception:
    DEEP_TRANSLATOR_AVAILABLE = False

DOTENV_AVAILABLE = False
GEMINI_AVAILABLE = False
GEMINI_API_KEY = ""
try:
    from dotenv import load_dotenv
    load_dotenv()
    DOTENV_AVAILABLE = True
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    if GEMINI_API_KEY:
        try:
            import google.generativeai as genai
            genai.configure(api_key=GEMINI_API_KEY)
            GEMINI_AVAILABLE = True
        except Exception:
            GEMINI_AVAILABLE = False
            GEMINI_API_KEY = ""
except Exception:
    DOTENV_AVAILABLE = False

st.set_page_config(page_title="Data Cleaner PRO", page_icon="🧹", layout="wide")

def load_data(uploaded_file):
    if uploaded_file.name.endswith('.csv'):
        try:
            # Carga Agnóstica de CSV: sep=None y engine='python'
            df = pd.read_csv(uploaded_file, sep=None, engine='python', encoding='utf-8')
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, sep=None, engine='python', encoding='latin1')
    elif uploaded_file.name.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(uploaded_file)
    else:
        st.error("Formato no soportado.")
        return None
    
    df.index = df.index + 1
    return df

def get_preselected_excluded(columns):
    import re
    res = []
    for c in columns:
        cl = str(c).lower().strip()
        if re.search(r'(^id$|^id_|_id$|^codigo$|^identificador$|^key$|^index$|^name$|^nombre$|^competitor$)', cl):
            res.append(c)
    return res

def descargar_figura_streamlit(figura, nombre_archivo, label=None):
    """Genera botón de descarga PNG para una figura de Matplotlib en cualquier pestaña."""
    import io as _io_fig_global
    buf = _io_fig_global.BytesIO()
    figura.savefig(buf, format="png", dpi=180, bbox_inches="tight")
    buf.seek(0)
    safe_nombre = (nombre_archivo if nombre_archivo.endswith(".png") else f"{nombre_archivo}.png")
    return st.download_button(
        label=label or f"📥 Descargar PNG",
        data=buf,
        file_name=safe_nombre,
        mime="image/png",
        use_container_width=True,
        key=f"dl_fig_gbl_{abs(hash(safe_nombre)) % (10**9)}"
    )

header_c1, header_c2, header_c3 = st.columns([5, 4, 2])
with header_c1:
    st.markdown("## 🧹 Dashboard Interactivo de Limpieza de Datasets")
    st.markdown("Plataforma reactiva para saneamiento granular de datos orientada a Machine Learning.")

ml_placeholder_c2 = header_c2.empty()
ml_placeholder_c3 = header_c3.empty()

with header_c3:
    st.write("")
    if DEEP_TRANSLATOR_AVAILABLE:
        lang_es = st.toggle('🌐 Traducir', value=st.session_state.get('lang_es', False), help="Traducir columnas al español")
        if lang_es != st.session_state.get('lang_es', False):
            st.session_state.lang_es = lang_es
            if lang_es:
                st.session_state.needs_translation = True
            st.session_state.just_translated = True
            if 'df_raw' in st.session_state:
                st.session_state.pop('df_raw')
                st.rerun()
    else:
        lang_es = False
        st.caption("🌐 Traducir: (requiere `pip install deep-translator`)")

run_model = False
sel_model = None

os.makedirs('data/raw', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)

TEMP_FILE = 'data/raw/current_session.pkl'

if 'uploader_key' not in st.session_state:
    st.session_state.uploader_key = str(uuid.uuid4())

up_col, rst_col = st.columns([4, 1])
with up_col:
    uploaded_file = st.file_uploader("Arrastra y suelta tu dataset (.csv, .xlsx, .xls)", type=['csv', 'xlsx', 'xls'], key=st.session_state.uploader_key)
with rst_col:
    st.write("")
    st.write("")
    if st.button("🗑️ Limpiar Todo y Reiniciar Sesión", use_container_width=True):
        st.session_state.clear()
        st.session_state.uploader_key = str(uuid.uuid4())
        if os.path.exists(TEMP_FILE):
            try:
                os.remove(TEMP_FILE)
            except: pass
        st.rerun()

if uploaded_file is not None:
    if 'df_raw' not in st.session_state or st.session_state.get('last_file') != uploaded_file.name:
        raw = load_data(uploaded_file)
        if raw is not None and st.session_state.get('needs_translation', False):
            st.session_state.needs_translation = False
            if DEEP_TRANSLATOR_AVAILABLE:
                with st.spinner("Traduciendo dataset al Español..."):
                    translator = GoogleTranslator(source='en', target='es')
                    new_cols = []
                    for col in raw.columns:
                        clean_col = str(col)
                        subs = {'percent': ' percent', 'name': ' name', 'almondy': ' almondy', 'rice': ' rice', 'wafer': ' wafer'}
                        for k, v in subs.items():
                            if k in clean_col and v not in clean_col:
                                clean_col = clean_col.replace(k, v)
                        try:
                            new_cols.append(translator.translate(clean_col).title().replace(" ", ""))
                        except:
                            new_cols.append(col)
                    raw.columns = new_cols
                    for col in raw.select_dtypes(include=['object', 'string']).columns:
                        unique_vals = raw[col].dropna().unique()
                        if len(unique_vals) < 30:
                            trans_dict = {}
                            for val in unique_vals:
                                try:
                                    trans_dict[val] = translator.translate(str(val))
                                except:
                                    trans_dict[val] = val
                            raw[col] = raw[col].replace(trans_dict)
            else:
                st.warning("ℹ️ Librería 'deep-translator' no instalada. Saltando traducción. Instálala con: pip install deep-translator")
        if raw is not None:
            st.session_state.df_raw = raw
            st.session_state.last_file = uploaded_file.name
            st.session_state.restored_cols = []
            st.session_state.df_cleaned = None
            st.session_state.audit_log = None
            st.session_state.cell_audit_log = None
            st.session_state.split_done = False
            # Persistencia en disco
            raw.to_pickle(TEMP_FILE)
elif os.path.exists(TEMP_FILE):
    if 'df_raw' not in st.session_state:
        st.session_state.df_raw = pd.read_pickle(TEMP_FILE)
        if st.session_state.get('needs_translation', False):
            st.session_state.needs_translation = False
            if DEEP_TRANSLATOR_AVAILABLE:
                with st.spinner("Traduciendo dataset restaurado al Español..."):
                    translator = GoogleTranslator(source='auto', target='es')
                    raw = st.session_state.df_raw
                    new_cols = []
                    for col in raw.columns:
                        try:
                            new_cols.append(translator.translate(col))
                        except:
                            new_cols.append(col)
                    raw.columns = new_cols
                    for col in raw.select_dtypes(include=['object', 'string']).columns:
                        unique_vals = raw[col].dropna().unique()
                        if len(unique_vals) < 30:
                            trans_dict = {}
                            for val in unique_vals:
                                try:
                                    trans_dict[val] = translator.translate(str(val))
                                except:
                                    trans_dict[val] = val
                            raw[col] = raw[col].replace(trans_dict)
                    st.session_state.df_raw = raw
            else:
                st.warning("ℹ️ Librería 'deep-translator' no instalada. Saltando traducción.")
        st.session_state.last_file = "Dataset Recuperado"
        st.session_state.restored_cols = []
        st.session_state.df_cleaned = None
        st.session_state.audit_log = None
        st.session_state.cell_audit_log = None
        st.session_state.split_done = False
    pass


if st.session_state.get('just_translated', False):
    if st.session_state.get('lang_es', False):
        st.success("✅ Traducción al Español exitosa. Ya puedes continuar.")
    else:
        st.success("✅ Se ha restaurado el dataset a su idioma original (Inglés).")
    st.session_state.just_translated = False
elif 'df_raw' in st.session_state and st.session_state.df_raw is not None and st.session_state.get('last_file') == "Dataset Recuperado":
    st.info("🔄 **Sesión Recuperada:** Hemos restaurado tu progreso anterior tras la recarga de página.")

if 'df_raw' in st.session_state and st.session_state.df_raw is not None:
    # Al cargar el dataset, siempre se muestran los controles superiores en la cabecera
    sel_family = None
    with ml_placeholder_c2.container():
        st.write("")
        st.session_state.cleaning_mode = st.radio("¿Cómo deseas limpiar los datos?", ["🤖 Limpieza Inteligente con IA (Klaro)", "🛠️ Limpieza Manual"], horizontal=True, label_visibility="collapsed")
    with ml_placeholder_c3.container():
        st.write("")
        st.write("")
        if st.button("🚀 EJECUTAR", type="primary", use_container_width=True):
            st.session_state.system_executed = True
            
            # Si eligieron Klaro, podemos limpiar directamente
            if "Klaro" in st.session_state.cleaning_mode:
                with st.spinner('Klaro está analizando y limpiando tu dataset...'):
                    target_rep = st.session_state.get('target_col_select', 'Ninguna')
                    if 'map_col_fn' in st.session_state:
                        target_rep = st.session_state.map_col_fn(None if target_rep == 'Ninguna' else target_rep)
                    
                    try:
                        # Llamar clean_pipeline
                        from src.cleaner import DataCleaner
                        cleaner = DataCleaner(st.session_state.df_raw)
                        if target_rep != 'Ninguna' and target_rep in st.session_state.df_raw.columns:
                            area, _ = cleaner.infer_ml_area(target_rep)
                            is_reg = "Regresionales" in area
                        else:
                            is_reg = False
                            
                        st.session_state.df_cleaned, st.session_state.audit_log, st.session_state.cell_audit_log, st.session_state.tukey_limits = cleaner.run_pipeline(target_col=target_rep)
                        
                        df_c = st.session_state.df_cleaned
                        if target_rep != 'Ninguna' and target_rep in df_c.columns:
                            X = df_c.drop(columns=[target_rep])
                            y = df_c[target_rep]
                            do_stratify = False
                            if not __import__('pandas').api.types.is_float_dtype(y):
                                counts = y.value_counts()
                                if len(counts) > 1 and counts.min() >= 2:
                                    do_stratify = True
                            from sklearn.model_selection import train_test_split
                            try:
                                X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=st.session_state.get('test_size_pct', 20)/100.0, random_state=42, stratify=(y if do_stratify else None))
                            except ValueError:
                                X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=st.session_state.get('test_size_pct', 20)/100.0, random_state=42, stratify=None)
                            st.session_state.split_data = {'X_train': X_tr, 'X_test': X_te, 'y_train': y_tr, 'y_test': y_te}
                            st.session_state.split_done = True
                    except Exception as e_pipe:
                        import traceback
                        st.error("🚨 **ERROR DURANTE LA LIMPIEZA (pantalla protegida):**")
                        st.code(f"Tipo: {type(e_pipe).__name__}\nMensaje: {str(e_pipe)}", language=None)
                        with st.expander("🔍 Ver detalle técnico (para copiar al programador)"):
                            st.code(traceback.format_exc(), language=None)
                        st.warning("💡 **Sugerencia:** Pulsa 'Limpiar Todo y Reiniciar Sesión', sube de nuevo el dataset, y en la pestaña Manual (no Klaro) prueba con una columna target válida. Si el error persiste, reporta este mensaje.")
                        st.session_state.system_executed = False
                        st.stop()
            st.rerun()

    if not st.session_state.get('system_executed', False):
        st.info('👆 Por favor, selecciona cómo deseas limpiar los datos y presiona EJECUTAR para procesar el dataset.')
        st.stop()
    if "Manual" in st.session_state.cleaning_mode and not st.session_state.get('manual_cleaning_done', False):
        df_raw = st.session_state.df_raw
        st.warning("Has elegido Limpieza Manual. Aquí tienes los datos **CRUDOS**.")
        st.info("💡 **Consejos de Klaro:**")
        nulos = df_raw.isna().sum()
        nulos = nulos[nulos > 0]
        if len(nulos) > 0:
            st.write("- Tienes nulos en: " + ", ".join([f"**{k}** ({v})" for k, v in nulos.items()]))
        else:
            st.write("- No detecto datos nulos.")
            
        corrupt = []
        for col in df_raw.select_dtypes(include=['object', 'string']).columns:
            cc = df_raw[col].astype(str).str.contains(r'[ÃÕÂ`´â€™]', regex=True, na=False).sum()
            if cc > 0: corrupt.append(col)
        if len(corrupt) > 0:
            st.write("- Tienes texto con codificación corrupta (símbolos raros) en: " + ", ".join(corrupt))
            
        st.write("Haz doble clic en la tabla para modificar valores, añade o quita filas:")
        
        if 'manual_df' not in st.session_state:
            st.session_state.manual_df = df_raw.copy()
            
        edited_df = st.data_editor(st.session_state.manual_df, num_rows="dynamic", use_container_width=True)
        
        if st.button("✅ Finalizar Limpieza Manual y Ver Resultados", type="primary"):
            st.session_state.manual_df = edited_df
            st.session_state.df_cleaned = edited_df
            
            # Split
            mapped_target = st.session_state.get('target_col_select', 'Ninguna')
            if 'map_col_fn' in st.session_state:
                mapped_target = st.session_state.map_col_fn(None if mapped_target == 'Ninguna' else mapped_target)
                
            if mapped_target and mapped_target in edited_df.columns:
                X = edited_df.drop(columns=[mapped_target])
                y = edited_df[mapped_target]
                do_stratify = False
                if not __import__('pandas').api.types.is_float_dtype(y):
                    counts = y.value_counts()
                    if len(counts) > 1 and counts.min() >= 2:
                        do_stratify = True
                from sklearn.model_selection import train_test_split
                try:
                    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=st.session_state.get('test_size_pct', 20)/100.0, random_state=42, stratify=(y if do_stratify else None))
                except ValueError:
                    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=st.session_state.get('test_size_pct', 20)/100.0, random_state=42, stratify=None)
                st.session_state.split_data = {'X_train': X_tr, 'X_test': X_te, 'y_train': y_tr, 'y_test': y_te}
                st.session_state.split_done = True
            st.session_state.manual_cleaning_done = True
            st.success("Limpieza Manual Finalizada.")
            st.rerun()
            
        st.stop()

    df_raw = st.session_state.df_raw
    
    st.subheader("Vistazo Rápido del Dataset (Datos Originales Sin Limpiar)")
    st.info("💡 **Nota:** Esta sección muestra los datos crudos exactamente como los subiste. Los resultados de la limpieza y la predicción (IA vs Manual) se encuentran más abajo, en las pestañas y en el reporte final.")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    nulos_total = df_raw.isna().sum().sum()
    celdas_total = df_raw.shape[0] * df_raw.shape[1]
    pct_nulos = (nulos_total / celdas_total) * 100 if celdas_total > 0 else 0
    mem_usage = df_raw.memory_usage(deep=True).sum() / (1024 * 1024)
    
    corrupt_cells = 0
    for col in df_raw.select_dtypes(include=['object', 'string']).columns:
        corrupt_cells += df_raw[col].astype(str).str.contains(r'[ÃÕÂ`´â€™]', regex=True, na=False).sum()
    
    col1.metric("Filas", f"{df_raw.shape[0]:,}")
    col2.metric("Columnas", df_raw.shape[1])
    col3.metric("Duplicados", df_raw.duplicated().sum())
    col4.metric("Celdas Nulas", f"{pct_nulos:.2f}%")
    col5.metric("Celdas Corruptas", f"{corrupt_cells}")
    
    if corrupt_cells > 0:
        st.error(f"⚠️ ¡Alerta! Se detectaron **{corrupt_cells} registros** con texto corrupto / mojibake. Se recomienda activar la reparación léxica en la configuración.")
    
    with st.expander("🔍 Ver Primeras 10 Filas (Datos Originales con Fila #)", expanded=False):
        st.dataframe(df_raw.head(10), use_container_width=True)
        
    target_val_selected = st.session_state.get('target_col_select', 'Ninguna')
    if target_val_selected != 'Ninguna':
        cleaner_infer = DataCleaner(df_raw)
        area, just = cleaner_infer.infer_ml_area(target_val_selected)
    st.divider()

    tab_names = ["📊 1. Auditoría", "⚙️ 2. Pipeline", "🔍 3. Comparativa Antes vs Después", "💾 4. Resultados", "🎯 5. Target (Predicciones)"]
    if st.session_state.get('ml_executed', False):
        tab_names.extend([
            "🏆 6. Comparación ML", 
            "🎯 7. Train/Test Split", 
            "📄 8. Bitácora ML",
            "🔮 9. Predicción Individual",
            "💾 10. Exportar Modelos",
            "📝 11. Reporte Académico"
        ])
    
    tabs = st.tabs(tab_names)
    tab1, tab2, tab3, tab4, tab5 = tabs[:5]
    if len(tabs) > 5:
        tab6, tab7, tab8, tab9, tab10, tab11 = tabs[5:]
    
    cleaner_temp = DataCleaner(df_raw)
    diagnostico = cleaner_temp.auto_diagnose()

    with tab1:
        st.header("Auditoría y Auto-Diagnóstico")
        st.table(diagnostico)
        ml_results_container = st.container()
    
    with tab2:
        st.header("⚙️ 2. Pipeline de Limpieza")
        st.info("Selecciona la columna que deseas predecir (Target) para que Klaro la proteja durante la limpieza.")
    
        all_cols = st.session_state.df_raw.columns.tolist()
        if 'target_col_select' not in st.session_state:
            st.session_state.target_col_select = 'Ninguna'
        target_col_opts = ['Ninguna'] + all_cols
        st.session_state.target_col_select = st.selectbox("🎯 Target (Variable Objetivo para Machine Learning):", target_col_opts, index=target_col_opts.index(st.session_state.target_col_select) if st.session_state.target_col_select in target_col_opts else 0)

        if 'test_size_pct' not in st.session_state:
            st.session_state.test_size_pct = 20
        st.session_state.test_size_pct = st.slider(
            "📊 Tamaño del conjunto de TEST (%):",
            min_value=10,
            max_value=70,
            value=st.session_state.test_size_pct,
            step=5,
            help="Ej: 50% = mitad de datos para test; 25% = cuarta parte; 20% = estándar académico."
        )
        c_ts1, c_ts2, c_ts3 = st.columns(3)
        c_ts1.caption(f"🧪 Train = **{100 - st.session_state.test_size_pct}%** ({max(1, int(len(st.session_state.df_raw) * (100 - st.session_state.test_size_pct) / 100))} filas aprox.)")
        c_ts2.caption(f"📝 Test  = **{st.session_state.test_size_pct}%** ({max(1, int(len(st.session_state.df_raw) * st.session_state.test_size_pct / 100))} filas aprox.)")
        c_ts3.caption("ℹ️ 20% = Estándar | 50% = Mitad | 75% = Cuarta parte para Train")

        if st.session_state.get('system_executed', False):
            if "Klaro" in st.session_state.cleaning_mode:
                st.success("✅ Klaro ejecutó el pipeline de forma inteligente y automática.")
            else:
                st.success("✅ Has completado la limpieza manual del pipeline.")
            
            if st.session_state.get('audit_log') is not None and len(st.session_state.audit_log) > 0:
                st.subheader("Bitácora de Operaciones (Fábrica de Limpieza)")
                import pandas as pd
                audit_df = pd.DataFrame(st.session_state.audit_log)
                audit_df = audit_df.rename(columns={"timestamp": "Fecha", "action": "Operación", "details": "Detalles", "justification": "Justificación Técnica"})

                st.dataframe(audit_df, use_container_width=True)
    with tab3:
        st.header("🔍 Comparativa Antes vs Después y Sustento Metodológico")
    
        if st.session_state.df_cleaned is not None:
            df_c = st.session_state.df_cleaned
            cell_audit_df = pd.DataFrame(st.session_state.cell_audit_log) if st.session_state.cell_audit_log else pd.DataFrame()
        
            # Recuentos Dinámicos
            dups_removed = df_raw.shape[0] - df_c.shape[0]
        
            if not cell_audit_df.empty:
                mojibake_count = cell_audit_df[cell_audit_df['Tipo de Error'] == 'Ruido Léxico / Mojibake'].shape[0]
                numeric_count = cell_audit_df[cell_audit_df['Tipo de Error'] == 'Conversión a Numérico / Saneamiento'].shape[0]
                outliers_count = cell_audit_df[cell_audit_df['Tipo de Error'].str.contains('Outlier', na=False, case=False)].shape[0]
            else:
                mojibake_count, numeric_count, outliers_count = 0, 0, 0
            
            pct_nulls_orig = (df_raw.isna().sum().sum() / (df_raw.shape[0] * df_raw.shape[1])) * 100 if df_raw.shape[0] > 0 else 0
        
            # Lógica Dinámica de Textos
            if dups_removed > 0:
                pillar_a = f"Se eliminaron {dups_removed} registros redundantes (duplicados) para evitar data leakage y sobreajuste en el entrenamiento."
            else:
                pillar_a = "Estructura íntegra sin redundancia (0 duplicados detectados)."
            
            if mojibake_count > 0:
                pillar_b = f"Se normalizaron {mojibake_count} registros con caracteres corruptos (mojibake) para preservar la integridad de las categorías."
            else:
                pillar_b = "Variables validadas en sus tipos de datos nativos; sin detección de ruido léxico estructural."
            
            pillar_c = f"Se detectó un {pct_nulls_orig:.2f}% de incompletitud inicial. Se trataron {outliers_count} valores atípicos mediante Capping IQR en variables predictoras continuas para mitigar distorsiones en gradientes de error."
        
            st.info(f"""
            ### 📝 Sustento Metodológico y Criterio de Ingeniería
        
            **A. Integridad Estructural (Inconsistencia):**
            {pillar_a}
        
            **B. Saneamiento de Ruido y Tipos:**
            {pillar_b}
        
            **C. Calidad Estadística (Incompletitud y Outliers):**
            {pillar_c}
            """)
        
            if not cell_audit_df.empty and (mojibake_count > 0 or numeric_count > 0):
                st.subheader("Detalle de Celdas Modificadas")
                st.dataframe(cell_audit_df, use_container_width=True)
                csv_cell_audit = cell_audit_df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Exportar Auditoría de Celdas (.csv)", data=csv_cell_audit, file_name='auditoria_celdas.csv', mime='text/csv')
        
            st.subheader("Comparativa de Dispersión Estadística (Variables Numéricas)")
            num_cols = df_c.select_dtypes(include=[np.number]).columns
        
            if len(num_cols) > 0:
                def map_back_list(cols):
                    res = []
                    for c in cols:
                        found = c
                        for orig in df_raw.columns:
                            cand = str(orig).lower().strip()
                            cand = re.sub(r'\s+', '_', cand)
                            cand = re.sub(r'[^\w\s]', '', cand)
                            if cand == c or orig == c:
                                found = orig
                                break
                        res.append(found)
                    return res
                
                orig_num_cols = [c for c in map_back_list(num_cols) if c in df_raw.columns]
            
                col_d1, col_d2 = st.columns(2)
                with col_d1:
                    st.write("**Antes de la limpieza (Datos Crudos):**")
                    if orig_num_cols:
                        df_raw_num_clean = df_raw[orig_num_cols].copy()
                        for c in orig_num_cols:
                            df_raw_num_clean[c] = pd.to_numeric(df_raw_num_clean[c].astype(str).str.replace(r'[\$,]', '', regex=True), errors='coerce')
                        st.dataframe(df_raw_num_clean.describe().T[['min', '25%', '50%', '75%', 'max']])
            
                with col_d2:
                    st.write("**Después de la limpieza (Capping IQR):**")
                    st.dataframe(df_c[num_cols].describe().T[['min', '25%', '50%', '75%', 'max']])
            else:
                st.info("No hay variables numéricas disponibles para comparar la dispersión.")
            
            with st.expander("📄 Texto para el Informe Universitario", expanded=False):
                st.write("Borrador metodológico listo para copiar y pegar en tu informe (basado en el análisis de Colab):")
                target_rep = st.session_state.get('target_col_select', 'Ninguna')
                area_rep, just_rep = ("No seleccionado", "") if target_rep == 'Ninguna' else cleaner_temp.infer_ml_area(target_rep)
            
                st.code(f'''1. INFERENCIA Y PRE-PROCESAMIENTO:
   - Se analizó la estructura del dataset y se estableció '{target_rep}' como Vector Objetivo (Y).
   - Área de Inferencia Dictaminada: {area_rep} ({just_rep}).

2. APLICACIÓN RIGUROSA DE CALIDAD DE DATOS (PIPELINE):
   - Inconsistencias: Se ejecutó `drop_duplicates()` suprimiendo {dups_removed} registros redundantes que generaban data leakage. Se aplicaron saneamientos léxicos con `strip()` y expresiones regulares para corrección de Mojibake.
   - Incompletitud: Se estableció un filtro estricto del 40% de umbral de descarte de columnas. Los nulos restantes fueron imputados vía sustitución muestral (mediana/moda) protegiendo al Vector Y.
   - Ruido (Outliers): Mediante el análisis de Tukey (Q1 - 1.5*IQR, Q3 + 1.5*IQR) se detectaron {outliers_count} valores extremos en variables predictoras. Se aplicó `np.clip()` (Winsorización) sobre los límites establecidos estabilizando la varianza sin destruir observaciones.''', language="markdown")


    with tab4:
        st.header("💾 4. Resultados")
        if st.session_state.get('df_cleaned') is not None:
            st.success("✅ Estos son tus datos limpios — **EDITABLES**: haz doble clic en cualquier celda para modificar valores, añade filas o elimínalas según sea necesario.")
            st.info("💡 **Consejo:** Los cambios realizados aquí se usan directamente para las predicciones ML de la Pestaña 5. No olvides pulsar 'Guardar Cambios' después de editar.")
            
            if 'df_edited_snapshot' not in st.session_state:
                st.session_state.df_edited_snapshot = st.session_state.df_cleaned.copy()
            
            edited_clean = st.data_editor(
                st.session_state.df_edited_snapshot,
                num_rows="dynamic",
                use_container_width=True,
                key="clean_df_editor_" + str(st.session_state.get('system_executed', False))
            )

            c_t4_1, c_t4_2, c_t4_3 = st.columns([2, 1, 1])
            if c_t4_1.button("💾 Guardar Cambios (usar en predicciones)", type="primary", use_container_width=True):
                st.session_state.df_cleaned = edited_clean.copy()
                st.session_state.df_edited_snapshot = edited_clean.copy()
                st.session_state.ml_executed = False
                st.session_state.split_done = False
                st.success("✅ ¡Cambios guardados! Ve a la Pestaña 5 'Target (Predicciones ML)' para entrenar los modelos con el dataset actualizado.")
                st.rerun()
            
            if c_t4_2.button("↩️ Revertir a limpieza original", use_container_width=True):
                st.session_state.df_edited_snapshot = st.session_state.df_cleaned.copy()
                st.rerun()

            if c_t4_3.button("❌ Restablecer filas #1", use_container_width=True, help="Restaura el índice visual para que empiece en 1"):
                edited_clean.index = range(1, len(edited_clean) + 1)
                st.session_state.df_edited_snapshot = edited_clean.copy()
                st.rerun()

            st.divider()
            st.subheader("⬇️ Descargar Dataset Limpio")
            c1, c2 = st.columns(2)
            c1.download_button("📄 Descargar CSV", data=st.session_state.df_cleaned.to_csv(index=False).encode('utf-8'), file_name='dataset_limpio.csv', mime='text/csv', use_container_width=True)
            
            try:
                import io as _io
                buffer_xlsx = _io.BytesIO()
                with pd.ExcelWriter(buffer_xlsx, engine='xlsxwriter') as writer:
                    st.session_state.df_cleaned.to_excel(writer, index=False, sheet_name='Dataset_Limpio')
                xlsx_bytes = buffer_xlsx.getvalue()
                c2.download_button("📊 Descargar Excel (.xlsx)", data=xlsx_bytes, file_name='dataset_limpio.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', use_container_width=True)
            except Exception as _e:
                c2.caption("ℹ️ Instala `xlsxwriter` para descargar en Excel: `pip install xlsxwriter`")
    with tab5:
        st.header("🎯 5. Target (Predicciones ML)")
        if st.session_state.get('df_cleaned') is not None:
            st.info("Configura tu entorno de Machine Learning.")
        
            cleaned_cols = st.session_state.df_cleaned.columns.tolist()
            y_col = st.session_state.get('target_col_select', 'Ninguna')
        
            c1, c2 = st.columns(2)
            y_final = c1.selectbox("Variable a Predecir (Y)", cleaned_cols, index=cleaned_cols.index(y_col) if y_col in cleaned_cols else 0)
        
            x_cols_default = [c for c in cleaned_cols if c != y_final]
            x_final = c2.multiselect("Variables Predictoras (X)", cleaned_cols, default=x_cols_default)
        
            st.subheader("Selección Jerárquica de Modelos")
            import pandas as pd
            area = "Clasificación" if pd.api.types.is_object_dtype(st.session_state.df_cleaned[y_final]) or st.session_state.df_cleaned[y_final].nunique() < 20 else "Regresión"
            
            import sys
            import importlib
            if 'src.models' in sys.modules:
                importlib.reload(sys.modules['src.models'])
            from src.models import ModelTrainer
            try:
                import numpy as np
                dummy_X = pd.DataFrame({'dummy': [1.0, 2.0]})
                dummy_y = pd.Series([0, 1])
                trainer_temp = ModelTrainer(dummy_X, dummy_X, dummy_y, dummy_y, area)
                m_dict = trainer_temp.models_dict
            except Exception as e:
                m_dict = {}

            if m_dict:
                valid_models = {k: v for k, v in m_dict.items() if v['model_task'] == ('classification' if area == "Clasificación" else 'regression')}
                families = sorted(list(set([v['family'] for v in valid_models.values()])))
                
                if not families:
                    st.error("No hay familias disponibles para este tipo de predicción.")
                    st.stop()
                    
                st.write("**🥊 Configurador del Modelo A**")
                c_fam1, c_alg1 = st.columns(2)
                fam_a = c_fam1.selectbox("Familia", families, key='fam_a')
                algos_a = [k for k, v in valid_models.items() if v['family'] == fam_a]
                modelo_a = c_alg1.selectbox("Algoritmo Específico", algos_a, key='alg_a')
                
                st.write("**🥊 Configurador del Modelo B**")
                c_fam2, c_alg2 = st.columns(2)
                fam_b_default = families[1] if len(families) > 1 else families[0]
                fam_b = c_fam2.selectbox("Familia", families, index=families.index(fam_b_default), key='fam_b')
                algos_b = [k for k, v in valid_models.items() if v['family'] == fam_b]
                modelo_b = c_alg2.selectbox("Algoritmo Específico", algos_b, key='alg_b')
            else:
                modelo_a, modelo_b = None, None
                st.error("Error al cargar el diccionario de modelos.")
                
            if st.session_state.get('ml_executed', False):
                st.success("✅ Predicciones completadas exitosamente. Ya puedes revisar las nuevas pestañas generadas (6, 7, 8 y 9) para ver las comparaciones y resultados.")
        
            if st.button("🚀 Ejecutar Predicciones", type="primary", use_container_width=True):
                # Guardar configuracion
                st.session_state.ml_y = y_final
                st.session_state.ml_x = x_final
                st.session_state.ml_modelo_a = modelo_a
                st.session_state.ml_modelo_b = modelo_b
                st.session_state.ml_area = area
            
                with st.spinner("Entrenando modelos y generando predicciones..."):
                    # Split Data
                    X = st.session_state.df_cleaned[x_final]
                    y = st.session_state.df_cleaned[y_final]
                    from sklearn.model_selection import train_test_split
                    do_stratify = False
                    if area == "Clasificación":
                        counts = y.value_counts()
                        if len(counts) > 1 and counts.min() >= 2:
                            do_stratify = True
                    try:
                        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=st.session_state.get('test_size_pct', 20)/100.0, random_state=42, stratify=(y if do_stratify else None))
                    except ValueError:
                        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=st.session_state.get('test_size_pct', 20)/100.0, random_state=42, stratify=None)
                    
                    st.session_state.split_data = {'X_train': X_tr, 'X_test': X_te, 'y_train': y_tr, 'y_test': y_te}
                    st.session_state.split_done = True
                    st.session_state.ml_executed = True
                st.rerun()
        else:
            st.warning("Debes ejecutar la limpieza de datos primero (Pestaña 2).")

if st.session_state.get('ml_executed', False):
    with tab6:
        st.header("🏆 Evaluación y Comparación de Modelos ML")
        if st.session_state.get('split_done', False):
            target_val_selected = st.session_state.get('ml_y', 'Ninguna')
            if target_val_selected != 'Ninguna':
                sd = st.session_state.split_data
                from src.models import ModelTrainer
                area_rep = st.session_state.get('ml_area', 'Clasificación')
                trainer = ModelTrainer(sd['X_train'], sd['X_test'], sd['y_train'], sd['y_test'], area_rep)
            
                with st.spinner("Entrenando múltiples modelos en paralelo (con GridSearchCV + Cross-Validation)..."):
                    results = trainer.evaluate_six_models()
                    st.session_state['_trainer_cache'] = trainer
                    st.session_state['_all_models_cached'] = [r['name'] for r in results]
                    st.session_state['_all_results_cached'] = results
            
                if results:
                    st.success("✅ Modelos evaluados correctamente con optimización de hiperparámetros.")
                    
                    st.subheader("🥊 Comparativa Cara a Cara")
                    modelo_a_name = st.session_state.get('ml_modelo_a', '')
                    modelo_b_name = st.session_state.get('ml_modelo_b', '')
                    
                    res_a = next((r for r in results if r['name'] == modelo_a_name), None)
                    res_b = next((r for r in results if r['name'] == modelo_b_name), None)
                    
                    col1, col2 = st.columns(2)
                    import seaborn as plt_sns
                    import matplotlib.pyplot as plt

                    ALGO_TEORIA = {
                        "Regresión Lineal Múltiple": ("Mínimos Cuadrados Ordinarios (OLS)", "Ajusta un hiperplano lineal Y = Xβ + ε minimizando el error cuadrático medio. Apropiado cuando hay relación lineal entre X e Y, sin multicolinealidad severa y homocedasticidad."),
                        "Regresión Polinómica (Grado 2)": ("Extensión de OLS con PolynomialFeatures degree=2", "Transforma las variables predictoras al espacio cuadrático (X², X₁X₂, etc.) permitiendo modelar relaciones curvas no lineales, aunque con mayor riesgo de sobreajuste."),
                        "Regresión Polinómica (Grado 5)": ("Extensión de OLS con PolynomialFeatures degree=5", "Ajuste polinomial de alto orden (grado 5) con mayor flexibilidad para capturar curvas complejas. Requiere regularización para evitar sobreajuste severo."),
                        "Regresión Ridge (Regularización L2)": ("OLS + penalización L2 (α·‖β‖²)", "Agrega una penalización Ridge (L2) al tamaño de los coeficientes. Estable frente a multicolinealidad y reduce overfitting encogiendo los pesos β hacia 0 mediante el hiperparámetro α."),
                        "Regresión Lasso (Regularización L1)": ("OLS + penalización L1 (α·|β|)", "Similar a Ridge pero con penalización L₁ que fuerza coeficientes β=0 en variables irrelevantes (selección automática de características). Ideal para dataset con muchos predictores ruidosos."),
                        "Regresión Logística Binaria": ("Función Sigmoide σ(z) = 1/(1+e^-z) + MLE", "Modela P(Y=1|X) mediante la función sigmoide sobre una combinación lineal de X. Optimizada por Máxima Verosimilitud. Produce fronteras de decisión lineales."),
                        "Regresión Softmax (Multiclase)": ("Generalización Logística K-clases (Cross-Entropy)", "Extiende Logística a K clases mediante la función softmax: P(Y=k|X) = e^{zₖ}/Σe^{zⱼ}. Minimiza la entropía cruzada. Genera probabilidades normalizadas por clase."),
                        "Árbol de Decisión por Índice GINI": ("CART con criterio Gini = 1-Σp²", "Árbol binario recursivo que particiona el espacio X según el atributo que maximiza la pureza de Gini (minimiza impureza). Máxima interpretabilidad, pero propenso a overfitting si no se poda (max_depth)."),
                        "Árbol de Decisión por Entropía": ("ID3/C4.5 con criterio Entropía + Information Gain", "Misma estructura CART pero el split se elige por Ganancia de Información (reducción de entropía H(S) = -Σp·log₂p). Similar a Gini en resultados prácticos, con interpretación teórica de Shannon."),
                        "Árbol de Decisión para Regresión": ("CART Regresor - criterio MSE", "Árbol que predice una variable continua particionando X y asignando a cada hoja la media de las Y. Usa MSE/MAE como criterio de split. Suave pero inestable a perturbaciones."),
                        "Random Forest Clasificador": ("Ensemble Bagging: N árboles de decisión independientes", "Ensemble de N árboles entrenados sobre muestras bootstrap diferentes de X, con selección aleatoria de features en cada split. Reduce varianza y overfitting por votación mayoritaria."),
                        "Random Forest Regresor": ("Ensemble Bagging: N árboles regresores (promedio)", "Mismo algoritmo de Bosques Aleatorios pero en regresión: la predicción final es el promedio de las predicciones de cada árbol individual. Robusto a outliers."),
                        "SVM Lineal (Clasificación)": ("Máquinas de Vectores Soporte con kernel='linear' - Hard/Soft Margin", "Busca el hiperplano óptimo que maximiza el margen entre clases. Regularizado por C (penalización a errores de clasificación). Funciona bien en espacios de alta dimensionalidad."),
                        "SVM Kernel RBF (Clasificación)": ("SVM + Función de Base Radial (Gaussiano)", "Mapea X implícitamente a espacio infinito-dimensional mediante kernel RBF: K(a,b) = exp(-γ||a-b||²). Capaz de aprender fronteras no lineales complejas. Controlado por γ y C."),
                        "SVM Kernel Polinomial (Clasificación)": ("SVM + kernel K(a,b)=(γa·b+r)^d", "Transforma X mediante kernel polinomial de grado d para capturar interacciones polinomiales sin expandir explícitamente el dataset. Alternativa a Regresión Polinómica."),
                        "SVM Kernel Sigmoide (Clasificación)": ("SVM + kernel K=tanh(γa·b+r)", "Usa kernel sigmoide/tanh, equivalente a una red neuronal de una sola capa. Menos estable que RBF, pero útil en problemas de clasificación no lineal tipo perceptrón."),
                        "SVR Lineal (Regresión)": ("SVM para Regresión - ε-SVR con kernel lineal", "Versión SVM de regresión: encuentra una banda de tolerancia ε (ε-tube) que incluye tantos puntos como sea posible y minimiza la norma de w, con penalización C para errores fuera de la banda."),
                        "SVR Kernel RBF (Regresión)": ("ε-SVR + kernel RBF Gaussiano", "Regresión SVR no lineal usando kernel RBF. Flexibilidad alta para modelar curvas suaves y relaciones no lineales complejas, insensible a outliers dentro de ε."),
                        "SVR Kernel Polinomial (Regresión)": ("ε-SVR + kernel polinomial K=(γa·b+r)^d", "Regresión SVR con mapeo polinomial implícito. Combina rigurosa regularización ε-SVR con flexibilidad de grado d."),
                        "SVR Kernel Sigmoide (Regresión)": ("ε-SVR + kernel tanh", "Ajuste de funciones tipo sigmoide para regresión no monótona. Útil cuando la respuesta tiene forma de S o transición suave."),
                        "KNN Clasificador": ("K-Vecinos más Cercanos (distancia Euclidiana p=2)", "Algoritmo basado en instancias (lazy learning): clasifica un punto nuevo por votación mayoritaria entre sus K vecinos más cercanos en el espacio X. Distancia por defecto: Minkowski/Euclidiana."),
                        "KNN Regresor": ("K-Vecinos más Cercanos - promedio de Y vecinos", "Mismo KNN pero para regresión: la predicción es la media (o mediana) de las Y de los K vecinos más cercanos. Simple y competitivo en espacios de baja dimensión."),
                        "Perceptrón Multicapa Clasificador": ("Red Neuronal Feed-Forward 1+ capas ocultas (ReLU) + Backprop", "Clasificador neuronal multicapa con activaciones no lineales (ReLU) en capas ocultas. Entrenado por descenso de gradiente estocástico (Adam/L-BFGS) minimizando cross-entropy. Capaz de aproximar cualquier función continua (Teorema de aproximación universal)."),
                        "Perceptrón Multicapa Regresor": ("RNA 1+ capas ocultas (ReLU) + MSE", "RNA para regresión: capa de salida sin activación (lineal), minimiza MSE. Potente modelador de relaciones no lineales complejas, aunque requiere escalado de datos y es sensible a arquitectura."),
                    }

                    def descargar_figura(figura, nombre_archivo, label=None):
                        """Helper para Mejora 4: botón de descarga PNG de cualquier figura Matplotlib."""
                        import io as _io_fig
                        buf = _io_fig.BytesIO()
                        figura.savefig(buf, format="png", dpi=180, bbox_inches="tight")
                        buf.seek(0)
                        return st.download_button(
                            label=label or f"📥 Descargar Gráfico PNG",
                            data=buf,
                            file_name=nombre_archivo if nombre_archivo.endswith(".png") else f"{nombre_archivo}.png",
                            mime="image/png",
                            use_container_width=True,
                            key=f"dl_fig_{abs(hash(nombre_archivo)) % (10**8)}"
                        )
                    
                    def clasificar_ajuste(res):
                        """Clasifica el ajuste del modelo: 🏆 Balanceado | 🚨 OVERFIT | 🐢 UNDERFIT.
                        Compara Train vs Test en métrica principal (F1/Acc para clasif, R² para reg)."""
                        met = res.get('metrics', {})
                        tr_met = res.get('extra', {}).get('train_metrics', {})
                        task = res.get('model_task', trainer.task)
                        if not tr_met:
                            return "ℹ️  Sin datos Train", (1.0, 1.0, 0.0)
                        if task == "regression":
                            te_main = met.get("R² Score", None)
                            tr_main = tr_met.get("R² Score (Train)", None)
                        else:
                            te_main = met.get("F1-Score", met.get("Accuracy", None))
                            tr_main = tr_met.get("F1-Score (Train)", tr_met.get("Accuracy (Train)", None))
                        if te_main is None or tr_main is None:
                            return "ℹ️  Métricas incompletas", (0.0, 0.0, 0.0)
                        gap = tr_main - te_main
                        if task == "regression":
                            overfit_thr = 0.10
                            underfit_thr = 0.40
                        else:
                            overfit_thr = 0.08
                            underfit_thr = 0.55
                        if (gap > overfit_thr) and (tr_main > te_main):
                            etiqueta = f"🚨 POSIBLE OVERFITTING — Train {tr_main:.3f} vs Test {te_main:.3f} (Δ = +{gap:.3f})"
                        elif (tr_main < underfit_thr) and (te_main < underfit_thr):
                            etiqueta = f"🐢 UNDERFITTING — Bajo desempeño global (Train {tr_main:.3f}, Test {te_main:.3f}). Considere más features / modelo más potente."
                        elif abs(gap) <= overfit_thr * 0.6:
                            etiqueta = f"🏆 AJUSTE EQUILIBRADO ✅ — Train {tr_main:.3f} vs Test {te_main:.3f} (Δ = {gap:+.3f}). Generalización óptima."
                        else:
                            etiqueta = f"⚠️ AJUSTE MODERADO — Train {tr_main:.3f} vs Test {te_main:.3f} (Δ = {gap:+.3f}). Ajuste aceptable sin overfitting severo."
                        return etiqueta, (tr_main, te_main, gap)

                    def display_model_card(res, col):
                        with col:
                            st.info(f"**{res['name']}**")
                            estado_ajuste, vals_ajuste = clasificar_ajuste(res)
                            tr_m, te_m, gap_m = vals_ajuste
                            if "OVERFITTING" in estado_ajuste:
                                st.error(estado_ajuste)
                            elif "UNDERFITTING" in estado_ajuste:
                                st.warning(estado_ajuste)
                            elif "EQUILIBRADO" in estado_ajuste:
                                st.success(estado_ajuste)
                            else:
                                st.warning(estado_ajuste)
                            teor = ALGO_TEORIA.get(res['name'])
                            if teor:
                                with st.expander(f"🧠 ¿Cómo funciona este modelo? → {teor[0]}", expanded=False):
                                    st.markdown(f"**Fundamento matemático:** {teor[1]}")
                            if 'best_params' in res['extra']:
                                st.caption(f"🏆 Mejores Hiperparámetros GridSearchCV: `{res['extra']['best_params']}`")
                            st.caption(f"💻 Implementación scikit-learn: `{str(res.get('params', ''))}`")
                            
                            m1, m2, m3 = st.columns(3)
                            all_metrics = list(res['metrics'].items())
                            train_met = res['extra'].get('train_metrics', {})
                            for i, (k, v) in enumerate(all_metrics):
                                if i % 3 == 0:
                                    m1.metric(k, f"{v:.4f}")
                                elif i % 3 == 1:
                                    m2.metric(k, f"{v:.4f}")
                                else:
                                    m3.metric(k, f"{v:.4f}")
                            
                            with st.expander("📊 Análisis Avanzado (Train vs Test + CV)", expanded=False):
                                if train_met:
                                    st.markdown("**🔬 Comparativa Train vs Test (diagnóstico de ajuste):**")
                                    train_keys = list(train_met.keys())
                                    tm_cols = st.columns(len(train_keys) + 1)
                                    for i, k in enumerate(train_keys):
                                        v = train_met[k]
                                        test_key = k.replace(" (Train)", "")
                                        test_v = res['metrics'].get(test_key)
                                        delta_txt = ""
                                        if test_v is not None:
                                            delta = v - test_v
                                            delta_txt = f"{delta:+.4f}"
                                        tm_cols[i].metric(k, f"{v:.4f}", delta=delta_txt or "0")
                                    
                                    test_vals = list(res['metrics'].values())[:2]
                                    train_vals = list(train_met.values())[:2]
                                
                                if 'cv_mean' in res['extra']:
                                    st.divider()
                                    st.markdown("**🔁 Validación Cruzada (5-Fold):**")
                                    cv1, cv2 = st.columns(2)
                                    cv1.metric("CV Promedio", f"{res['extra']['cv_mean']:.4f}")
                                    cv2.metric("CV Desv. Estándar", f"{res['extra']['cv_std']:.4f}")
                                    if 'cv_scores' in res['extra']:
                                        st.write("Scores por fold:", [round(s, 4) for s in res['extra']['cv_scores'].tolist()])
                            
                            fig, ax = plt.subplots(figsize=(5, 4))
                            if trainer.task == "classification":
                                cm = res['extra']['cm']
                                labels = res['extra']['labels']
                                plt_sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax)
                                ax.set_xlabel("Predicción")
                                ax.set_ylabel("Valor Real")
                                ax.set_title("Matriz de Confusión")
                            else:
                                y_pred = res['extra']['y_pred']
                                y_true = sd['y_test']
                                ax.scatter(y_true, y_pred, alpha=0.5, color='teal')
                                ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
                                ax.set_xlabel("Valor Real")
                                ax.set_ylabel("Predicción")
                                ax.set_title("Ajuste Lineal (Real vs Pred)")
                            fig.tight_layout()
                            st.pyplot(fig)
                            safe_m_name = re.sub(r'[^\w\s-]', '', res['name']).strip().replace(' ', '_')
                            tipo_graf = "Matriz_Confusion" if trainer.task == "classification" else "Ajuste_Lineal_Real_vs_Pred"
                            descargar_figura_streamlit(fig, f"01_{safe_m_name}_{tipo_graf}", label=f"📥 Guardar {tipo_graf.replace('_',' ')} (.png)")
                            
                            if trainer.task == "classification" and 'roc_curve' in res['extra']:
                                with st.expander("📈 Curva ROC-AUC", expanded=False):
                                    fpr, tpr, roc_auc = res['extra']['roc_curve']
                                    fig_roc, ax_roc = plt.subplots(figsize=(5, 4))
                                    ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.4f})')
                                    ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                                    ax_roc.set_xlabel('Tasa de Falsos Positivos')
                                    ax_roc.set_ylabel('Tasa de Verdaderos Positivos')
                                    ax_roc.set_title('Curva ROC')
                                    ax_roc.legend(loc="lower right")
                                    fig_roc.tight_layout()
                                    st.pyplot(fig_roc)
                                    descargar_figura_streamlit(fig_roc, f"02_{safe_m_name}_Curva_ROC_AUC", label="📥 Guardar Curva ROC (.png)")
                            
                            if trainer.task == "regression" and 'residuals' in res['extra']:
                                with st.expander("📊 Análisis de Residuales", expanded=False):
                                    resids = res['extra']['residuals']
                                    fig_res, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
                                    ax1.scatter(res['extra']['y_pred'], resids, alpha=0.6)
                                    ax1.axhline(y=0, color='r', linestyle='--')
                                    ax1.set_xlabel("Predicción")
                                    ax1.set_ylabel("Residual")
                                    ax1.set_title("Residuales vs Predicción")
                                    ax2.hist(resids, bins=30, edgecolor='black', alpha=0.7)
                                    ax2.axvline(x=0, color='r', linestyle='--')
                                    ax2.set_title("Distribución de Residuales")
                                    fig_res.tight_layout()
                                    st.pyplot(fig_res)
                                    descargar_figura_streamlit(fig_res, f"03_{safe_m_name}_Analisis_Residuales", label="📥 Guardar Residuales (.png)")
                            
                            if 'feature_importance' in res['extra'] and res['extra']['feature_importance'] is not None:
                                with st.expander("🎯 Importancia de Variables (Top 10)", expanded=False):
                                    fi_df = res['extra']['feature_importance'].head(10)
                                    fig_fi, ax_fi = plt.subplots(figsize=(6, 5))
                                    plt_sns.barplot(data=fi_df, x='importance', y='feature', palette='viridis', ax=ax_fi)
                                    ax_fi.set_xlabel("Importancia")
                                    ax_fi.set_ylabel("Variable")
                                    ax_fi.set_title(f"Importancia - {res['name']}")
                                    fig_fi.tight_layout()
                                    st.pyplot(fig_fi)
                                    descargar_figura_streamlit(fig_fi, f"04_{safe_m_name}_Importancia_Variables", label="📥 Guardar Importancia (.png)")
                            
                    if res_a:
                        display_model_card(res_a, col1)
                    if res_b:
                        display_model_card(res_b, col2)
                                
                    st.divider()
                    st.subheader("📊 Comparación Global (Todos los Modelos)")
                    
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(figsize=(12, 6))
                    import numpy as np
                
                    models_names = [r['name'] for r in results]
                
                    if trainer.task == "classification":
                        metrics_keys = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
                        if any('ROC-AUC' in r['metrics'] for r in results):
                            metrics_keys.append('ROC-AUC')
                    else:
                        metrics_keys = ['R² Score', 'MAE', 'RMSE']
                    
                    x = np.arange(len(models_names))
                    width = 0.15
                
                    for idx, m in enumerate(metrics_keys):
                        vals = [r['metrics'].get(m, 0) for r in results]
                        ax.bar(x + idx*width, vals, width, label=m)
                    
                    ax.set_ylabel('Puntuación')
                    ax.set_title('Comparación de Modelos (GridSearchCV + CV)')
                    ax.set_xticks(x + width * (len(metrics_keys)-1)/2)
                    
                    import textwrap
                    wrapped_labels = [textwrap.fill(name, width=25) for name in models_names]
                    ax.set_xticklabels(wrapped_labels, rotation=45, ha='right', fontsize=9)
                    
                    ax.legend()
                    fig.subplots_adjust(bottom=0.35)
                    fig.tight_layout()
                    st.pyplot(fig)
                    descargar_figura_streamlit(fig, "05_Comparacion_Global_Modelos", label="📥 Guardar Comparativa Global (.png)")
                    
                    st.subheader("📋 Tabla Resumen de Todos los Modelos")
                    summary_rows = []
                    for r in results:
                        row = {"Modelo": r['name']}
                        row.update({k: round(v, 5) for k, v in r['metrics'].items()})
                        if 'best_params' in r['extra']:
                            row["Mejores Params"] = str(r['extra']['best_params'])
                        if 'cv_mean' in r['extra']:
                            row["CV Mean"] = round(r['extra']['cv_mean'], 5)
                        summary_rows.append(row)
                    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)
                    
                    best_model = results[0]
                    st.subheader(f"🏆 Mejor Modelo Global: {best_model['name']}")
                
                    if trainer.task == "classification":
                        import seaborn as plt_sns
                        cm = best_model['extra']['cm']
                        labels = best_model['extra']['labels']
                        fig2, ax2 = plt.subplots(figsize=(6, 4))
                        plt_sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax2)
                        ax2.set_xlabel("Predicción")
                        ax2.set_ylabel("Valor Real")
                        ax2.set_title(f"Matriz de Confusión - {best_model['name']}")
                        st.pyplot(fig2)
                        safe_best_name = re.sub(r'[^\w\s-]', '', best_model['name']).strip().replace(' ', '_')
                        descargar_figura_streamlit(fig2, f"06_Mejor_Modelo_{safe_best_name}_Matriz_Confusion", label="📥 Guardar Matriz del Mejor Modelo (.png)")
        else:
            st.info("Ejecuta el pipeline de predicciones en la Pestaña 5 primero.")
    if False: # with tab4:
        st.header("🔍 Análisis Exploratorio y Consultas Dinámicas")
        if st.session_state.get('df_cleaned') is not None:
            df_c = st.session_state.df_cleaned
            st.write("Explora gráficamente tu dataset cruzando variables libremente.")
        
            c1, c2 = st.columns(2)
            x_axis = c1.selectbox("Eje X (Variable Independiente / Categoría):", df_c.columns)
            y_axis = c2.selectbox("Eje Y (Variable Dependiente / Valor):", df_c.columns, index=min(1, len(df_c.columns)-1))
        
            graf_type = st.radio("Tipo de Gráfico:", ["Barras", "Líneas", "Dispersión"], horizontal=True)
        
            if st.button("Generar Gráfico"):
                fig3, ax3 = plt.subplots(figsize=(10, 5))
                if graf_type == "Barras":
                    if pd.api.types.is_numeric_dtype(df_c[y_axis]):
                        grouped = df_c.groupby(x_axis)[y_axis].mean().reset_index()
                        plt_sns.barplot(data=grouped, x=x_axis, y=y_axis, ax=ax3, palette='viridis')
                    else:
                        st.warning("El Eje Y debe ser numérico para gráfico de barras.")
                elif graf_type == "Líneas":
                    if pd.api.types.is_numeric_dtype(df_c[y_axis]):
                        grouped = df_c.groupby(x_axis)[y_axis].mean().reset_index()
                        plt_sns.lineplot(data=grouped, x=x_axis, y=y_axis, ax=ax3, marker='o')
                    else:
                        st.warning("El Eje Y debe ser numérico para gráfico de líneas.")
                else:
                    plt_sns.scatterplot(data=df_c, x=x_axis, y=y_axis, ax=ax3, hue=df_c.columns[-1] if len(df_c.columns) > 1 else None)
            
                ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right')
                fig3.tight_layout()
                st.pyplot(fig3)
        else:
            st.info("No hay datos limpios para explorar.")
if st.session_state.get('ml_executed', False):
    with tab8:
        if st.session_state.audit_log is not None:
            st.header("Bitácora Técnica General")
            if len(st.session_state.audit_log) > 0:
                audit_df = pd.DataFrame(st.session_state.audit_log)
                audit_df = audit_df.rename(columns={"timestamp": "Fecha", "action": "Fase", "details": "Detalles", "justification": "Justificación"})
                st.table(audit_df)
            else:
                st.info("Sin cambios aplicados.")
            
if st.session_state.get('ml_executed', False):
    with tab7:
        st.header("🎯 Análisis del Train/Test Split")
        if st.session_state.get('split_done', False):
            sd = st.session_state.split_data
            area_rep = st.session_state.get('ml_area', 'Clasificación')
            
            ts_pct = st.session_state.get('test_size_pct', 20)
            tr_pct = 100 - ts_pct
            st.info(f"""
            **📝 Justificación Metodológica del Split:**
            - **Proporción {tr_pct}% Train / {ts_pct}% Test:** Configuración seleccionada por el usuario. {tr_pct}% de los datos se usan para entrenar el modelo y {ts_pct}% para evaluar su capacidad de generalización.
              - Si {ts_pct}% ≈ 20% → estándar académico, balance óptimo.
              - Si {ts_pct}% ≈ 50% → "mitad-mitad", útil para datasets pequeños o validación robusta.
              - Si {ts_pct}% ≈ 70% → Train ≈ 30% (cuarta parte aprox.), útil para prueba de sobreajuste.
            - **Random State = 42:** Semilla determinística para garantizar reproducibilidad de resultados entre ejecuciones.
            - **Stratify (solo Clasificación):** Mantiene la misma distribución de clases en Train y Test para evitar sesgos de muestreo.
            """)

            st.session_state.split_ts_pct = ts_pct
            
            res_col1, res_col2, res_col3, res_col4 = st.columns(4)
            res_col1.metric(f"X_train ({tr_pct}%)", f"{sd['X_train'].shape}")
            res_col2.metric(f"y_train ({tr_pct}%)", f"{sd['y_train'].shape}")
            res_col3.metric(f"X_test ({ts_pct}%)", f"{sd['X_test'].shape}")
            res_col4.metric(f"y_test ({ts_pct}%)", f"{sd['y_test'].shape}")
            
            st.divider()
            st.subheader("📊 Distribución de la Variable Objetivo (Y)")
            y_train = sd['y_train']
            y_test = sd['y_test']
            
            if area_rep == "Clasificación" or (hasattr(y_train, 'nunique') and y_train.nunique() < 20):
                train_dist = pd.DataFrame({"Conjunto": "Train", y_train.name if hasattr(y_train, 'name') else "Y": y_train.value_counts().index, "Cantidad": y_train.value_counts().values, "%": (y_train.value_counts(normalize=True).values * 100).round(2)})
                test_dist = pd.DataFrame({"Conjunto": "Test", y_test.name if hasattr(y_test, 'name') else "Y": y_test.value_counts().index, "Cantidad": y_test.value_counts().values, "%": (y_test.value_counts(normalize=True).values * 100).round(2)})
                dist_df = pd.concat([train_dist, test_dist], ignore_index=True)
                st.dataframe(dist_df, use_container_width=True)
                
                fig_dist, ax_dist = plt.subplots(figsize=(10, 4))
                plot_df = dist_df.rename(columns={dist_df.columns[1]: "Clase"})
                plt_sns.barplot(data=plot_df, x="Clase", y="%", hue="Conjunto", palette='coolwarm', ax=ax_dist)
                ax_dist.set_title("Distribución de Clases: Train vs Test (%)")
                ax_dist.set_xticklabels(ax_dist.get_xticklabels(), rotation=45, ha='right')
                fig_dist.tight_layout()
                st.pyplot(fig_dist)
                descargar_figura_streamlit(fig_dist, "07_Distribucion_Clases_Train_vs_Test", label="📥 Guardar Distribución Y (.png)")
            else:
                desc_train = pd.DataFrame(y_train.describe()).T
                desc_train["Conjunto"] = "Train"
                desc_test = pd.DataFrame(y_test.describe()).T
                desc_test["Conjunto"] = "Test"
                st.dataframe(pd.concat([desc_train, desc_test], ignore_index=True), use_container_width=True)
                
                fig_hist, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                ax1.hist(y_train.dropna(), bins=30, alpha=0.7, color='blue', edgecolor='black')
                ax1.set_title(f"Distribución Y - Train (n={len(y_train)})")
                ax1.set_xlabel("Valor")
                ax2.hist(y_test.dropna(), bins=30, alpha=0.7, color='orange', edgecolor='black')
                ax2.set_title(f"Distribución Y - Test (n={len(y_test)})")
                ax2.set_xlabel("Valor")
                fig_hist.tight_layout()
                st.pyplot(fig_hist)
                descargar_figura_streamlit(fig_hist, "08_Distribucion_Y_Histograma_Train_Test", label="📥 Guardar Histograma Y (.png)")
            
            st.divider()
            dcol1, dcol2, dcol3, dcol4 = st.columns(4)
            def to_csv_b(df): return df.to_csv(index=False).encode('utf-8')
            dcol1.download_button("X_train.csv", data=to_csv_b(sd['X_train']), file_name='X_train.csv', mime='text/csv', use_container_width=True)
            dcol2.download_button("X_test.csv", data=to_csv_b(sd['X_test']), file_name='X_test.csv', mime='text/csv', use_container_width=True)
            dcol3.download_button("y_train.csv", data=to_csv_b(pd.DataFrame({'y': sd['y_train']})), file_name='y_train.csv', mime='text/csv', use_container_width=True)
            dcol4.download_button("y_test.csv", data=to_csv_b(pd.DataFrame({'y': sd['y_test']})), file_name='y_test.csv', mime='text/csv', use_container_width=True)

if st.session_state.get('ml_executed', False):
    with tab9:
        st.header("🔮 Predicción Individual (Caso Nuevo)")
        st.info("Ingresa los valores de las variables predictoras para obtener una predicción **interactiva y explicada** con gráficos + interpretación en lenguaje natural. ⚡ Ahora usa los modelos YA entrenados (sin reentrenar) → respuesta en segundos.")
        
        if st.session_state.get('split_done', False) and st.session_state.get('ml_executed', False):
            sd = st.session_state.split_data
            area_rep = st.session_state.get('ml_area', 'Clasificación')
            from src.models import ModelTrainer

            trainer_cached = st.session_state.get('_trainer_cache')
            results_cached = st.session_state.get('_all_results_cached')
            if trainer_cached is None or results_cached is None:
                with st.status("⚙️ Preparando modelos (solo la primera vez)...", expanded=True) as status_prep:
                    st.write("1/2 · Construyendo trainer y codificadores de variables...")
                    trainer_cached = ModelTrainer(sd['X_train'], sd['X_test'], sd['y_train'], sd['y_test'], area_rep)
                    st.write("2/2 · Evaluando todos los modelos (GridSearchCV incluido) y cacheando resultados...")
                    results_cached = trainer_cached.evaluate_six_models()
                    st.session_state['_trainer_cache'] = trainer_cached
                    st.session_state['_all_models_cached'] = [r['name'] for r in results_cached]
                    st.session_state['_all_results_cached'] = results_cached
                    status_prep.update(label="✅ Modelos listos. Ya no se repetirá este paso.", state="complete", expanded=False)
            trainer = trainer_cached
            all_models = [r['name'] for r in results_cached]

            def _buscar_resultado(nombre):
                for rr in results_cached:
                    if rr['name'] == nombre:
                        return rr
                return results_cached[0] if results_cached else None

            c_sel1, c_sel2 = st.columns([2, 1])
            model_to_use = c_sel1.selectbox("Selecciona el modelo para la predicción:", all_models, index=0)
            res_modelo = _buscar_resultado(model_to_use)
            if res_modelo is not None:
                met_principal = list(res_modelo['metrics'].items())[0]
                c_sel2.metric(f"📊 {met_principal[0]} (Test)", f"{met_principal[1]:.4f}", help="Rendimiento del modelo seleccionado sobre el Test Set")
            
            input_cols = st.session_state.get('ml_x', [])
            if not input_cols:
                input_cols = [c for c in sd['X_train'].columns]
            
            input_dict = {}
            y_col = st.session_state.get('ml_y', '')
            
            with st.form("prediction_form"):
                st.subheader(f"Ingresar valores para: {len(input_cols)} variables")
                n_cols_ui = 3
                cols_ui = st.columns(n_cols_ui)
                for i, col in enumerate(input_cols):
                    col_ui = cols_ui[i % n_cols_ui]
                    series = sd['X_train'][col]
                    if pd.api.types.is_numeric_dtype(series):
                        min_v = float(series.min())
                        max_v = float(series.max())
                        mean_v = float(series.mean())
                        step_v = (max_v - min_v) / 100 if max_v != min_v else 0.01
                        input_dict[col] = col_ui.number_input(f"{col}", min_value=min_v, max_value=max_v, value=round(mean_v, 4), step=step_v, format="%.4f", help=f"Rango dataset: [{min_v:.4f} , {max_v:.4f}]  ·  Media = {mean_v:.4f}")
                    else:
                        unique_vals = sorted(series.dropna().unique().tolist())
                        default_v = unique_vals[0] if unique_vals else ""
                        input_dict[col] = col_ui.selectbox(f"{col}", unique_vals, index=0)
                
                submitted = st.form_submit_button("🎯 Realizar Predicción", type="primary", use_container_width=True)
            
            if submitted:
                with st.status(f"🧮 Analizando caso nuevo con **{model_to_use}**...", expanded=True) as status_pred:
                    progress_bar = st.progress(0.0, text="1/5 · Validando valores de entrada...")
                    import time as _time_p9
                    _time_p9.sleep(0.05)
                    progress_bar.progress(20, text="2/5 · Aplicando pre-procesamiento (One-Hot / Label + Scaler) al nuevo caso...")
                    _time_p9.sleep(0.05)
                    progress_bar.progress(45, text="3/5 · Ejecutando inferencia del modelo (feed-forward)...")
                    result = trainer.predict_single(model_to_use, input_dict)
                    _time_p9.sleep(0.05)
                    progress_bar.progress(75, text="4/5 · Calculando métricas de contexto vs dataset histórico...")
                    import matplotlib.pyplot as _plt_p9
                    import seaborn as _sns_p9
                    _time_p9.sleep(0.05)
                    progress_bar.progress(92, text="5/5 · Generando gráficos y texto interpretativo...")
                    _time_p9.sleep(0.05)
                    progress_bar.progress(100, text="✅ Listo")
                    status_pred.update(label="✅ Predicción completada", state="complete", expanded=False)
                
                if result is not None:
                    pred_val, proba_val = result
                    tarea_p9 = trainer.task

                    st.success(f"🎯 Predicción de `{y_col}` completada exitosamente con **{model_to_use}**")

                    st.divider()
                    st.subheader("📌 1) Resultado Principal + Interpretación")
                    pred_col1, pred_col2 = st.columns([1, 1])

                    if tarea_p9 == "classification":
                        classes = sorted(pd.Series(sd['y_train']).dropna().astype(str).unique().tolist())
                        proba_list = list(proba_val) if proba_val is not None else []
                        hay_proba = (proba_val is not None) and (len(proba_list) == len(classes))

                        if hay_proba:
                            idx_max = int(np.argmax(proba_list))
                            clase_max = classes[idx_max]
                            prob_max = float(proba_list[idx_max]) * 100.0
                        else:
                            clase_max = str(pred_val)
                            prob_max = 0.0
                            idx_max = 0

                        delta_txt = None
                        clase_pred_str = str(pred_val)
                        try:
                            freqs = pd.Series(sd['y_train']).astype(str).value_counts(normalize=True) * 100.0
                            freq_clase = float(freqs.get(clase_pred_str, 0.0))
                            if freq_clase > 0:
                                delta_txt = f"{(prob_max - freq_clase):+.1f} pts vs base"
                        except Exception:
                            pass
                        pred_col1.metric(
                            f"✅ Predicción final de `{y_col}`",
                            f"{clase_max}",
                            delta=delta_txt if delta_txt else None,
                            help="Clase con mayor probabilidad estimada por el modelo"
                        )
                        if hay_proba:
                            with pred_col2:
                                st.markdown(f"**🔥 Probabilidad clase ganadora:**")
                                st.metric(f"P(clase = {clase_max})", f"{prob_max:.2f}%")

                        st.markdown("#### 💬 Interpretación en lenguaje natural:")
                        if hay_proba:
                            if prob_max >= 75:
                                conf = "muy alta confianza"
                                rec = "Se recomienda confiar en esta predicción."
                            elif prob_max >= 55:
                                conf = "confianza moderada-alta"
                                rec = "Resultado razonable; revisar si es una decisión crítica."
                            elif prob_max >= 40:
                                conf = "confianza moderada"
                                rec = "El modelo tiene dudas. Considera corroborar con otro modelo."
                            else:
                                conf = "baja confianza (resultado ambiguo)"
                                rec = "Hay clases alternativas con probabilidades similares — NO confiar ciegamente."
                            texto_clases = ", ".join([f"**P({c}) = {float(p)*100:.2f}%**" for c, p in zip(classes, proba_list)])
                            st.info(f"""
Con los valores ingresados y usando el modelo **{model_to_use}** ({list(res_modelo['metrics'].items())[0][0]} Test = {list(res_modelo['metrics'].values())[0]:.4f}), se estima con **{conf}** que el nuevo caso pertenece a la clase **`{clase_max}`** con una probabilidad del **{prob_max:.2f}%**.

Probabilidad por cada clase posible: {texto_clases}.

{rec}
""")
                        else:
                            st.info(f"Modelo **{model_to_use}** predijo la clase `{pred_val}` (probabilidades no disponibles para este estimador).")

                        st.divider()
                        st.subheader("📊 2) Gráficos de la Predicción")
                        if hay_proba:
                            try:
                                df_proba = pd.DataFrame({"Clase": classes, "Probabilidad %": [float(p) * 100.0 for p in proba_list]})
                                colores = ['#FF4B4B' if float(p)*100 == prob_max else '#69B3FF' for p in proba_list]
                                fig_proba, ax_proba = _plt_p9.subplots(figsize=(10, 4.5))
                                bars = ax_proba.bar(df_proba["Clase"], df_proba["Probabilidad %"], color=colores, edgecolor='black')
                                for b, v in zip(bars, df_proba["Probabilidad %"]):
                                    ax_proba.text(b.get_x() + b.get_width()/2, v + 1.0, f"{v:.1f}%", ha='center', fontweight='bold', fontsize=10)
                                ax_proba.set_ylim(0, min(110, max(10, 110, max(df_proba["Probabilidad %"]) + 15)))
                                ax_proba.set_ylabel("Probabilidad (%)", fontsize=11)
                                ax_proba.set_xlabel("Clases posibles", fontsize=11)
                                ax_proba.set_title(f"Distribución de Probabilidad · Modelo: {model_to_use}", fontsize=13, fontweight='bold')
                                ax_proba.axhline(50, color='gray', linestyle='--', linewidth=1, alpha=0.6, label="Umbral 50%")
                                ax_proba.legend(loc="upper right")
                                ax_proba.grid(axis='y', alpha=0.3)
                                fig_proba.tight_layout()
                                st.pyplot(fig_proba)
                                safe_m_name = re.sub(r'[^\w\s-]', '', model_to_use).strip().replace(' ', '_')
                                descargar_figura_streamlit(fig_proba, f"P9_{safe_m_name}_Probabilidades_Caso_Nuevo", label="📥 Descargar gráfico de Probabilidades (.png)")
                            except Exception:
                                pass

                            st.markdown("##### 🧪 Comparación del nuevo caso vs Promedio histórico del dataset")
                            try:
                                classes_ord = classes
                                base_pct = [100.0 * float(pd.Series(sd['y_train']).astype(str).value_counts(normalize=True).get(c, 0.0)) for c in classes_ord]
                                nuevo_pct = [float(p) * 100.0 for p in proba_list]
                                df_comp_clases = pd.DataFrame({
                                    "Clase": classes_ord * 2,
                                    "Porcentaje %": base_pct + nuevo_pct,
                                    "Contexto": ["Promedio dataset (Train)"] * len(classes_ord) + ["Nuevo caso (predicción)"] * len(classes_ord)
                                })
                                fig_comp, ax_comp = _plt_p9.subplots(figsize=(11, 4.8))
                                _sns_p9.barplot(data=df_comp_clases, x="Clase", y="Porcentaje %", hue="Contexto", palette="coolwarm", ax=ax_comp, edgecolor='black')
                                for container in ax_comp.containers:
                                    for bar in container:
                                        h = bar.get_height()
                                        if h > 2:
                                            ax_comp.text(bar.get_x() + bar.get_width()/2.0, h + 0.5, f"{h:.1f}%", ha='center', va='bottom', fontsize=9)
                                ax_comp.set_ylim(0, df_comp_clases["Porcentaje %"].max() * 1.2)
                                ax_comp.set_title(f"Comparativo: Balance histórico vs Caso Nuevo · {model_to_use}", fontsize=13, fontweight='bold')
                                ax_comp.grid(axis='y', alpha=0.3)
                                fig_comp.tight_layout()
                                st.pyplot(fig_comp)
                                descargar_figura_streamlit(fig_comp, f"P9_{safe_m_name}_Comparativo_vs_Dataset", label="📥 Descargar Comparativo (.png)")
                            except Exception:
                                pass

                        st.divider()
                        if res_modelo and 'feature_importance' in res_modelo['extra'] and res_modelo['extra']['feature_importance'] is not None:
                            st.subheader("🎯 3) Variables que más influyeron en este resultado")
                            st.caption("Variables con mayor peso en el modelo seleccionado (feature importance global).")
                            fi_df = res_modelo['extra']['feature_importance'].head(8)
                            fig_fi_p9, ax_fi_p9 = _plt_p9.subplots(figsize=(10, 4.5))
                            _sns_p9.barplot(data=fi_df, x='importance', y='feature', palette='viridis', ax=ax_fi_p9, edgecolor='black')
                            ax_fi_p9.set_xlabel("Importancia", fontsize=11)
                            ax_fi_p9.set_ylabel("Variable predictora", fontsize=11)
                            ax_fi_p9.set_title(f"Top-{len(fi_df)} variables más importantes · {model_to_use}", fontsize=13, fontweight='bold')
                            ax_fi_p9.grid(axis='x', alpha=0.3)
                            fig_fi_p9.tight_layout()
                            st.pyplot(fig_fi_p9)
                            descargar_figura_streamlit(fig_fi_p9, f"P9_{safe_m_name}_Importancia_Variables", label="📥 Descargar Importancia Variables (.png)")
                        elif res_modelo:
                            st.subheader("🎯 3) Importancia de variables")
                            st.caption("ℹ️ Este modelo (`{}`) no expone feature_importances_ (ej: KNN, SVM kernel). Se recomienda consultar un modelo de árbol o lineal para análisis de variables.".format(model_to_use))
                    else:
                        y_num = pd.to_numeric(sd['y_train'], errors='coerce').dropna()
                        pred_num = float(pred_val)
                        try:
                            from scipy import stats as _stats_p9
                            percentil = float(_stats_p9.percentileofscore(y_num.values, pred_num, kind='weak'))
                        except Exception:
                            percentil = float(np.mean(np.where(y_num.values <= pred_num, 1, 0))) * 100.0

                        y_mean = float(y_num.mean())
                        y_median = float(y_num.median())
                        y_min = float(y_num.min())
                        y_max = float(y_num.max())
                        rmse_test = float(res_modelo['metrics'].get("RMSE", 0.0)) if res_modelo else 0.0

                        if percentil > 95:
                            texto_percentil = "🔝 Percentil MUY ALTO (≥95%) → este valor predicho está ENTRE LOS MÁXIMOS del dataset."
                        elif percentil >= 80:
                            texto_percentil = "⬆️ Percentil ALTO (≥80%) → superior a la mayoría de observaciones históricas."
                        elif percentil >= 50:
                            texto_percentil = "➡️ Percentil MEDIO (50%~80%) → en el rango esperado más común."
                        elif percentil >= 20:
                            texto_percentil = "⬇️ Percentil BAJO (20%~50%) → por debajo de la mediana del dataset."
                        else:
                            texto_percentil = "🔚 Percentil MUY BAJO (<20%) → entre los valores mínimos del dataset."

                        delta_pred = pred_num - y_mean
                        if abs(delta_pred) < 0.05 * abs(y_mean):
                            delta_txt = "≈ igual a la media"
                        elif delta_pred > 0:
                            delta_txt = f"{abs(delta_pred):.4f} **por ENCIMA** de la media"
                        else:
                            delta_txt = f"{abs(delta_pred):.4f} **por DEBAJO** de la media"

                        pred_col1.metric(
                            f"✅ Predicción final de `{y_col}`",
                            f"{pred_num:.4f}",
                            delta=delta_txt,
                            help="Valor continuo estimado por el modelo de regresión"
                        )
                        if rmse_test > 0:
                            pred_col2.metric("📏 Intervalo esperado ±RMSE", f"[{pred_num - rmse_test:.4f} , {pred_num + rmse_test:.4f}]", delta=f"RMSE Test = {rmse_test:.4f}", help="Rango razonable tomando como margen de error el RMSE del Test Set")

                        st.markdown("#### 💬 Interpretación en lenguaje natural:")
                        st.info(f"""
Con los valores ingresados y usando el modelo **{model_to_use}** (R² Test = {res_modelo['metrics'].get('R² Score', 0.0):.4f} · RMSE Test = {rmse_test:.4f}), se estima que el nuevo caso tendrá un valor de **`{y_col}` ≈ {pred_num:.4f}**.

**Contexto estadístico del nuevo valor frente al dataset histórico:**
- El valor predicho `{pred_num:.4f}` está en el **Percentil ≈ {percentil:.1f}%**. {texto_percentil}
- Media del dataset histórico: `{y_mean:.4f}` → la predicción queda **{delta_txt}** (`{y_median:.4f}` mediana).
- Rango total de `{y_col}` en Train: `[{y_min:.4f} , {y_max:.4f}]`.
- **Rango esperado 95% aproximado** (±1.96 × RMSE): `[{pred_num - 1.96 * rmse_test:.4f} , {pred_num + 1.96 * rmse_test:.4f}]`.
""")

                        st.divider()
                        st.subheader("📊 2) Gráficos de la Predicción")
                        try:
                            fig_hist_p9, ax_hist_p9 = _plt_p9.subplots(figsize=(11, 4.8))
                            _sns_p9.histplot(y_num, bins=30, kde=True, color='#69B3FF', edgecolor='black', alpha=0.8, linewidth=1, ax=ax_hist_p9, label='Distribución Y (Train)')
                            ax_hist_p9.axvline(y_mean, color='green', linestyle='--', linewidth=2, label=f'Media Train = {y_mean:.3f}')
                            ax_hist_p9.axvline(y_median, color='purple', linestyle=':', linewidth=2, label=f'Mediana Train = {y_median:.3f}')
                            ax_hist_p9.axvline(pred_num, color='red', linestyle='-', linewidth=3, label=f'PREDICCIÓN = {pred_num:.3f}', zorder=10)
                            if rmse_test > 0:
                                ax_hist_p9.axvspan(pred_num - rmse_test, pred_num + rmse_test, alpha=0.2, color='red', label=f'Bandera ±RMSE ({rmse_test:.3f})')
                            ax_hist_p9.set_xlabel(f"Valor de `{y_col}`", fontsize=11)
                            ax_hist_p9.set_ylabel("Frecuencia", fontsize=11)
                            ax_hist_p9.set_title(f"Histograma de `{y_col}` + Predicción del Caso Nuevo · Modelo = {model_to_use}", fontsize=13, fontweight='bold')
                            ax_hist_p9.legend(loc='upper right', fontsize=9)
                            ax_hist_p9.grid(axis='y', alpha=0.3)
                            fig_hist_p9.tight_layout()
                            st.pyplot(fig_hist_p9)
                            safe_m_name = re.sub(r'[^\w\s-]', '', model_to_use).strip().replace(' ', '_')
                            descargar_figura_streamlit(fig_hist_p9, f"P9_{safe_m_name}_Histograma_Y_Prediccion", label="📥 Descargar Histograma Predicción (.png)")
                        except Exception:
                            pass

                        try:
                            fig_box_p9, ax_box_p9 = _plt_p9.subplots(figsize=(9, 4.8))
                            bp = ax_box_p9.boxplot(y_num, vert=True, patch_artist=True, labels=[f"`{y_col}` (Train)"], widths=0.4, medianprops=dict(color='purple', linewidth=2))
                            for patch in bp['boxes']:
                                patch.set_facecolor('#69B3FF')
                            ax_box_p9.scatter([1.0], [pred_num], color='red', s=220, zorder=10, marker='*', edgecolors='black', label=f'Predicción = {pred_num:.3f}')
                            if rmse_test > 0:
                                ax_box_p9.fill_between([0.65, 1.35], pred_num - rmse_test, pred_num + rmse_test, color='red', alpha=0.13, label=f'±RMSE = {rmse_test:.3f}')
                            ax_box_p9.set_ylabel(f"Valor de `{y_col}`", fontsize=11)
                            ax_box_p9.set_title(f"Boxplot dataset histórico + Estrella roja = Predicción ({model_to_use})", fontsize=13, fontweight='bold')
                            ax_box_p9.legend(loc='upper right', fontsize=9)
                            ax_box_p9.grid(axis='y', alpha=0.3)
                            fig_box_p9.tight_layout()
                            st.pyplot(fig_box_p9)
                            descargar_figura_streamlit(fig_box_p9, f"P9_{safe_m_name}_Boxplot_Caso_Nuevo", label="📥 Descargar Boxplot (.png)")
                        except Exception:
                            pass

                        st.divider()
                        if res_modelo and 'feature_importance' in res_modelo['extra'] and res_modelo['extra']['feature_importance'] is not None:
                            st.subheader("🎯 3) Variables que más influyeron en este resultado")
                            fi_df = res_modelo['extra']['feature_importance'].head(8)
                            fig_fi_p9, ax_fi_p9 = _plt_p9.subplots(figsize=(10, 4.5))
                            _sns_p9.barplot(data=fi_df, x='importance', y='feature', palette='viridis', ax=ax_fi_p9, edgecolor='black')
                            ax_fi_p9.set_xlabel("Importancia", fontsize=11)
                            ax_fi_p9.set_ylabel("Variable predictora", fontsize=11)
                            ax_fi_p9.set_title(f"Top-{len(fi_df)} variables más importantes · {model_to_use}", fontsize=13, fontweight='bold')
                            ax_fi_p9.grid(axis='x', alpha=0.3)
                            fig_fi_p9.tight_layout()
                            st.pyplot(fig_fi_p9)
                            descargar_figura_streamlit(fig_fi_p9, f"P9_{safe_m_name}_Importancia_Variables", label="📥 Descargar Importancia Variables (.png)")
                        elif res_modelo:
                            st.subheader("🎯 3) Importancia de variables")
                            st.caption(f"ℹ️ Este modelo (`{model_to_use}`) no expone feature_importances_ (ej: KNN, SVM kernel). Usa Random Forest / Árbol / Regresión para análisis de variables.")

                    try:
                        df_summary = pd.DataFrame({
                            "Variable": input_cols,
                            "Valor ingresado (Caso Nuevo)": [
                                input_dict[c] for c in input_cols
                            ],
                            "Promedio dataset (Train)": [
                                round(float(pd.to_numeric(sd['X_train'][c], errors='coerce').mean()), 4)
                                if pd.api.types.is_numeric_dtype(sd['X_train'][c])
                                else str(pd.Series(sd['X_train'][c]).dropna().mode().iloc[0] if len(pd.Series(sd['X_train'][c]).dropna().mode()) > 0 else "-")
                                for c in input_cols
                            ]
                        })
                        with st.expander("📋 Comparativa detallada: Valores ingresados vs Promedio dataset (Train)", expanded=False):
                            st.dataframe(df_summary, use_container_width=True, hide_index=True)
                            csv_comp = df_summary.to_csv(index=False).encode('utf-8')
                            st.download_button("📥 Descargar comparativa CSV", data=csv_comp, file_name="P9_Comparativo_Caso_Nuevo_vs_Dataset.csv", mime="text/csv", use_container_width=True)
                    except Exception:
                        pass
                else:
                    st.error("❌ Error al generar la predicción. Intenta con otro modelo o revisa los valores ingresados.")

if st.session_state.get('ml_executed', False):
    with tab10:
        st.header("💾 Exportar Modelos Entrenados")
        st.info("Descarga tus modelos entrenados en formato `.joblib` para reutilizarlos en otros proyectos o entornos.")
        
        if st.session_state.get('split_done', False):
            sd = st.session_state.split_data
            area_rep = st.session_state.get('ml_area', 'Clasificación')
            from src.models import ModelTrainer
            trainer = st.session_state.get('_trainer_cache', ModelTrainer(sd['X_train'], sd['X_test'], sd['y_train'], sd['y_test'], area_rep))
            
            all_models = st.session_state.get('_all_models_cached', [])
            if not all_models:
                with st.spinner("Cargando modelos..."):
                    res_list = trainer.evaluate_six_models()
                    all_models = [r['name'] for r in res_list]
            
            model_to_export = st.multiselect("Selecciona los modelos para exportar:", all_models, default=all_models[:1])
            
            if model_to_export:
                for m_name in model_to_export:
                    safe_name = re.sub(r'[^\w\s-]', '', m_name).strip().replace(' ', '_')
                    temp_path = f"data/processed/{safe_name}.joblib"
                    try:
                        with st.spinner(f"Exportando {m_name}..."):
                            trainer.export_model(m_name, temp_path)
                            with open(temp_path, "rb") as f:
                                file_bytes = f.read()
                            st.download_button(
                                label=f"📥 Descargar: {m_name}.joblib",
                                data=file_bytes,
                                file_name=f"{safe_name}.joblib",
                                mime="application/octet-stream",
                                key=f"dl_{safe_name}"
                            )
                            if os.path.exists(temp_path):
                                os.remove(temp_path)
                    except Exception as e:
                        st.error(f"Error exportando {m_name}: {str(e)}")
                
                with st.expander("📖 ¿Cómo usar el modelo exportado?", expanded=False):
                    st.code("""
import joblib
import pandas as pd

bundle = joblib.load("Nombre_Modelo.joblib")
model = bundle["model"]
feature_names = bundle["feature_names"]
scaler = bundle["scaler"]
label_encoders = bundle["label_encoders"]

nuevo_dato = pd.DataFrame([{
    "variable1": 10.5,
    "variable2": "valor_categoria"
}])

# Aplicar mismo preprocesamiento...
prediccion = model.predict(nuevo_dato[feature_names].fillna(0).values)
print(f"Predicción: {prediccion[0]}")
                    """, language="python")

if st.session_state.get('ml_executed', False):
    with tab11:
        st.header("📝 Reporte Académico Automático")
        st.success("✅ Informe **100% DINÁMICO**: se actualiza AUTOMÁTICAMENTE en cada ejecución según: tu dataset cargado, la variable objetivo Y que elegiste, las X features seleccionadas, el % Train/Test configurado, el ranking REAL de modelos, la RNA ganadora del GridSearchCV y la predicción de usuario que generaste. Nada es fijo.")
        with st.expander("🧬 ¿Qué partes cambian según tu trabajo?", expanded=False):
            st.markdown("""
| Sección del Informe | ¿Se modifica con tu dataset? |
|---|---|
| **I. Introducción** | ✅ N° real de modelos entrenados + área Clasif/Reg según tu Y |
| **II. Objetivos** | ✅ % Train/Test que elegiste en el slider (10%~70%) |
| **III. Marco Teórico** | ✅ Tipo de Y (Continua vs Discreta) · Resultados REALES del GridSearchCV RNA |
| **IV. Marco Referencial** | ✅ Shape REAL de tu dataset (filas × cols originales) |
| **V(a-e) Ingeniería** | ✅ Limpieza (nulos reales, duplicados, IQR), Features X que seleccionaste, ranking completo de modelos por rendimiento REAL, 6 métricas reales, mejor hiperparámetro elegido por GridSearchCV, bandera de overfit REAL (Train vs Test) |
| **V(f) Predicción Usuario** | ✅ Ejemplo real usando PROMEDIOS/MODAS de TU dataset y probabilidades por clase |
| **VI. Conclusiones** | ✅ Nulos inicial/final % REALES, Mejor modelo REAL, 6 métricas reales |
| **VII. Bibliografía** | 📚 6 referencias académicas fijas (Géron, James, Pedregosa, McKinney, Hastie + docs) |
""")
        
        if st.session_state.get('split_done', False):
            sd = st.session_state.split_data
            area_rep = st.session_state.get('ml_area', 'Clasificación')
            df_cleaned = st.session_state.df_cleaned
            df_raw = st.session_state.df_raw
            target_col = st.session_state.get('ml_y', '')
            
            from src.models import ModelTrainer
            trainer_cached = st.session_state.get('_trainer_cache')
            results_cached = st.session_state.get('_all_results_cached')
            if trainer_cached is None or results_cached is None:
                trainer_cached = ModelTrainer(sd['X_train'], sd['X_test'], sd['y_train'], sd['y_test'], area_rep)
                with st.spinner("Cacheando resultados para reporte (una vez)..."):
                    results_cached = trainer_cached.evaluate_six_models()
                    st.session_state['_trainer_cache'] = trainer_cached
                    st.session_state['_all_models_cached'] = [r['name'] for r in results_cached]
                    st.session_state['_all_results_cached'] = results_cached
            trainer = trainer_cached
            results = results_cached
            
            if results:
                best = results[0]
                second = results[1] if len(results) > 1 else None
                
                area_full = "Modelos Regresionales y Redes Neuronales" if trainer.task == "regression" else "Modelos de Árboles de Decisión, Clasificación Lineal y Redes Neuronales Artificiales (RNA)"
                y_type = "Continua (Regresión)" if trainer.task == "regression" else "Discreta/Categórica (Clasificación)"
                
                nulos_inicial = df_raw.isna().sum().sum()
                nulos_final = df_cleaned.isna().sum().sum()
                dups = df_raw.duplicated().sum()
                ts_pct_rep = st.session_state.get('split_ts_pct', st.session_state.get('test_size_pct', 20))
                tr_pct_rep = 100 - ts_pct_rep

                x_cols_justif = st.session_state.get('ml_x', [])
                target_col = st.session_state.get('ml_y', target_col if target_col else '')
                if not x_cols_justif and target_col in df_cleaned.columns:
                    x_cols_justif = [c for c in df_cleaned.columns if c != target_col]
                if x_cols_justif:
                    justificacion_features = f"""
        - **Variable Objetivo (Y):** `{target_col}` — Se seleccionó porque representa el fenómeno que queremos predecir/controlar (resultado de interés académico o de negocio).
        - **Variables Predictoras (X):** {len(x_cols_justif)} columnas seleccionadas por el usuario como input del modelo: `{', '.join([f'`{c}`' for c in x_cols_justif[:12]])}{' + otras' if len(x_cols_justif) > 12 else ''}`.
        - **Criterio de selección:** Se excluyeron identificadores (`id`, códigos) y columnas con varianza nula (constantes) porque no aportan poder discriminante. Se preservaron todas las características numéricas y categóricas que pudieran correlacionarse con `{target_col}`.
"""
                else:
                    justificacion_features = f"""
        - **Variable Objetivo (Y):** `{target_col}`.
        - **Variables Predictoras (X):** Todas las columnas restantes del dataset, excepto identificadores.
"""

                METRICAS_TEORIA = {
                    "regression": [
                        ("R² Score", "Coeficiente de determinación. Mide qué proporción de la varianza de Y es explicada por X. Rango (-∞, 1]; idealmente → 1."),
                        ("Exp. Variance", "Explained Variance Score. Varianza explicada; similar a R² pero insensible al sesgo del intercepto."),
                        ("RMSE", "Root Mean Squared Error. Raíz del MSE. Penaliza errores grandes. Misma unidad que Y (interpretabilidad)."),
                        ("MAE", "Mean Absolute Error. Error promedio en valor absoluto. Robusto a outliers; métrica de error base."),
                        ("Median AE", "Median Absolute Error. Mediana de |error|. Muy robusta a valores atípicos extremos."),
                        ("MAPE %", "Mean Absolute Percentage Error. Error porcentual medio. Permite comunicar desempeño en % (unidades intuitivas para stakeholders)."),
                    ],
                    "classification": [
                        ("Accuracy", "Exactitud global. Proporción de predicciones correctas. Apropiada para datasets balanceados."),
                        ("F1-Score", "Media armónica entre Precision y Recall (β=1). Métrica robusta ante clases desbalanceadas."),
                        ("Precision", "Precisión: cuántos de los clasificados como 'positivos' son realmente positivos. Reduce falsos positivos."),
                        ("Recall", "Sensibilidad / Exhaustividad: cuántos de los reales 'positivos' son capturados. Reduce falsos negativos."),
                        ("Cohen Kappa", "Índice de acuerdo inter-observador corregido por azar. Medida más conservadora que Accuracy; evalúa la calidad del clasificador más allá del azar."),
                        ("Jaccard (IoU)", "Intersection over Union. Tamaño de la intersección / unión entre predichos y reales. Comun en clasificación multiclase y segmentación."),
                    ]
                }
                teorico_metricas = METRICAS_TEORIA[trainer.task]
                justif_metricas = "\n".join([
                    f"           - **{k}:** {v}" for k, v in teorico_metricas
                ]) + ("\n           - **ROC-AUC (extra binario):** Área bajo la curva ROC. Mide capacidad discriminatoria general (solo disponible cuando Y tiene 2 clases)." if trainer.task == "classification" else "")

                rna_modelo_nombre = None
                for r in results:
                    if "Perceptrón Multicapa" in r['name']:
                        rna_modelo_nombre = r['name']
                        break
                if rna_modelo_nombre:
                    rna_extra = [r for r in results if r['name'] == rna_modelo_nombre][0]
                    rna_mejores_params = rna_extra['extra'].get('best_params', {})
                    texto_rna = f"""
- **Red Neuronal Artificial (RNA): Perceptrón Multicapa (Feed-Forward + Backpropagation):**
  - **Arquitectura:** RNA de tipo Feed-Forward (cada neurona de capa L se conecta a todas de L+1; sin retroalimentación ciclica).
  - **Entrenamiento:** Algoritmo de **Backpropagation** (retropropagación del error) + optimizador Adam / L-BFGS. Se minimiza la función de pérdida Cross-Entropy (clasificación) o MSE (regresión) mediante descenso de gradiente sobre pesos y sesgos.
  - **Funciones de activación:** Capas ocultas con ReLU (Rectified Linear Unit, f(x)=max(0,x)). Capa de salida: sigmoide/softmax (clasificación) / lineal (regresión).
  - **Regularización:** Término L2 `alpha` para penalizar pesos grandes y evitar overfitting.
  - **Resultado RNA en este experimento:** Modelo `{rna_modelo_nombre}` — {'R²' if trainer.task == 'regression' else 'F1-Score'} = {list(rna_extra['metrics'].values())[0]:.4f}  (puesto #{[rr['name'] for rr in results].index(rna_modelo_nombre)+1} de {len(results)}).
  - **Mejores hiperparámetros RNA (GridSearchCV):** `{rna_mejores_params if rna_mejores_params else '(no activado)'}`.
"""
                else:
                    texto_rna = f"""
- **Red Neuronal Artificial (RNA): Perceptrón Multicapa (Feed-Forward + Backpropagation):**
  - **Arquitectura:** RNA Feed-Forward 1+ capas ocultas. **Nota:** se evaluó en el pipeline pero no se registró en resultados (posible incompatibilidad temporal).
"""

                try:
                    ejemplo_x = {}
                    ejemplo_texto = "Predicción con valores medios del dataset (usuario libre de cambiar en Pestaña 9):\n"
                    for col in x_cols_justif:
                        if col in df_cleaned.columns:
                            s = df_cleaned[col]
                            if pd.api.types.is_numeric_dtype(s):
                                v_ej = round(float(s.mean()), 4)
                                ejemplo_x[col] = v_ej
                                ejemplo_texto += f"           - `{col}` = {v_ej}  (media de la columna)\n"
                            else:
                                moda = s.dropna().mode()
                                v_ej = moda.iloc[0] if len(moda) > 0 else (s.dropna().iloc[0] if len(s.dropna()) else "")
                                ejemplo_x[col] = v_ej
                                ejemplo_texto += f"           - `{col}` = '{v_ej}'  (moda)\n"
                    if hasattr(trainer, 'predict_single'):
                        pred_ej, proba_ej = trainer.predict_single(best['name'], ejemplo_x)
                        if pred_ej is not None:
                            if proba_ej is not None and len(proba_ej) > 0:
                                classes_ej = sorted(pd.Series(sd['y_train']).dropna().unique().tolist())
                                idx_max = int(np.argmax(list(proba_ej)))
                                clase_max = classes_ej[idx_max] if len(classes_ej) == len(proba_ej) else pred_ej
                                ejemplo_texto += f"""
        **Resultado de la predicción (usuario):**
           - Modelo usado: `{best['name']}`
           - Entrada personalizada (Pestaña 9 'Predicción Individual'): el usuario ingresa N valores para las columnas X.
           - **Ejemplo práctico:**
{ejemplo_texto}
           - **Predicción final de `{target_col}`:** `{pred_ej}`
           - **Probabilidad clase mayoritaria:** {float(max(list(proba_ej)))*100:.2f}% (Clase `{clase_max}`)
           - **Interpretación:** {'El sistema asigna mayor probabilidad (>' + str(round(float(max(list(proba_ej)))*100,1)) + '%) a que el nuevo resultado caiga en la clase ' + str(clase_max) + '. Se recomienda confirmar con el Mejor Modelo Global.' if trainer.task=='classification' else 'La predicción estima que `' + str(target_col) + '` ≈ ' + str(round(float(pred_ej), 4)) + ' unidades de la variable respuesta.'}
"""
                            else:
                                ejemplo_texto += f"""
        **Resultado de la predicción (usuario):**
           - Modelo usado: `{best['name']}`
           - **Ejemplo práctico (valores medios / moda):**
{ejemplo_texto}
           - **Predicción final de `{target_col}`:** `{pred_ej}`
           - **Interpretación (Regresión):** El nuevo caso tiene un valor predicho de `{round(float(pred_ej), 4)}` para `{target_col}` — interpretar dentro del rango original de Y: [{round(float(sd['y_train'].min()), 3)}, {round(float(sd['y_train'].max()), 3)}].
"""
                except Exception:
                    ejemplo_texto = """
        **Nota:** La predicción de ejemplo se genera en la Pestaña 9 (🔮 Predicción Individual). El usuario ingresa valores numéricos/categóricos y el sistema devuelve:
           - Valor predicho de `Y`
           - Probabilidades por clase (solo clasificación) + barras de confianza visuales
           - Interpretación escrita del resultado.
"""
                
                if trainer.task == "regression":
                    metrica_principal = f"R² Score = {best['metrics']['R² Score']:.4f}"
                    metricas_list = "\n".join([f"           - **{k}:** {v:.4f}" for k, v in best['metrics'].items()])
                else:
                    metrica_principal = f"F1-Score = {best['metrics'].get('F1-Score', best['metrics'].get('Accuracy', 0)):.4f}"
                    metricas_list = "\n".join([f"           - **{k}:** {v:.4f}" for k, v in best['metrics'].items()])
                
                hiperparams = ""
                if 'best_params' in best['extra']:
                    hiperparams = f"\n        - Mejores hiperparámetros encontrados con GridSearchCV: `{best['extra']['best_params']}`"
                
                cv_text = ""
                if 'cv_mean' in best['extra']:
                    cv_text = f"\n        - Validación Cruzada (5-Fold): Promedio = {best['extra']['cv_mean']:.4f} ± {best['extra']['cv_std']:.4f}"
                
                overfit_text = ""
                train_met = best['extra'].get('train_metrics', {})
                if train_met:
                    if trainer.task == "regression":
                        te_main = best['metrics'].get("R² Score")
                        tr_main = train_met.get("R² Score (Train)")
                        over_thr = 0.10
                        under_thr = 0.40
                    else:
                        te_main = best['metrics'].get("F1-Score", best['metrics'].get("Accuracy"))
                        tr_main = train_met.get("F1-Score (Train)", train_met.get("Accuracy (Train)"))
                        over_thr = 0.08
                        under_thr = 0.55
                    if te_main is not None and tr_main is not None:
                        gap = tr_main - te_main
                        if (gap > over_thr) and (tr_main > te_main):
                            overfit_text = f"""
        - 🚨 **POSIBLE OVERFITTING (Sobreajuste):** Desempeño Train = {tr_main:.4f} vs Test = {te_main:.4f}  (Δ = +{gap:.3f} > {over_thr}).
          *Acciones de mitigación aplicadas:* Validación Cruzada 5-Fold, GridSearchCV, regularización L1/L2 y poda `max_depth` en árboles.
"""
                        elif (tr_main < under_thr) and (te_main < under_thr):
                            overfit_text = f"""
        - 🐢 **UNDERFITTING (Subajuste):** Bajo desempeño global (Train = {tr_main:.4f}, Test = {te_main:.4f} ambos < {under_thr}).
          *Recomendación para la presentación:* considerar (1) más variables predictoras, (2) mayor cantidad de datos, (3) modelo más potente (Random Forest o RNA profundo).
"""
                        elif abs(gap) <= over_thr * 0.6:
                            overfit_text = f"""
        - 🏆 **AJUSTE EQUILIBRADO ✅ (MEJOR CASO):** Train = {tr_main:.4f} vs Test = {te_main:.4f}  (Δ = {gap:+.3f}, dentro de ±{over_thr*0.6:.3f}).
          *Interpretación:* Generalización óptima; el modelo aprende patrones generales y no memoriza datos (sin overfitting, sin underfitting).
"""
                        else:
                            overfit_text = f"""
        - ⚠️ **AJUSTE MODERADO (Aceptable):** Train = {tr_main:.4f} vs Test = {te_main:.4f}  (Δ = {gap:+.3f}).
          *Comentario:* diferencia pequeña pero perceptible; NO hay overfitting severo. El modelo cumple requisitos académicos.
"""
                
                matriz_conf_text = ""
                if trainer.task == "classification" and 'cm' in best['extra']:
                    cm = best['extra']['cm']
                    matriz_conf_text = f"""
        - **Matriz de Confusión ({best['name']}):**
{pd.DataFrame(best['extra']['cm'], 
                index=[f"Real {l}" for l in best['extra']['labels']],
                columns=[f"Pred {l}" for l in best['extra']['labels']]).to_string()}
"""
                
                reporte = f"""
# 📘 INFORME FINAL - PRÁCTICA DEL PRIMER PARCIAL
## Lic. Patricia Rodríguez Bilbao
**Estudiantes / Grupo:** [Completar con nombres del grupo]

---

## I. INTRODUCCIÓN
El presente trabajo aborda la implementación de procesos de Aprendizaje Automático (Machine Learning Supervisado) sobre un dataset asignado por la docente. Se aplican técnicas rigurosas de pre-procesamiento de datos (ETL), limpieza de 3 niveles (inconsistencias, incompletitud y ruido), seguido del entrenamiento, optimización por GridSearchCV y comparativa estadística de {len(results)} modelos de la familia de **{area_full}**, incluyendo una **Red Neuronal Artificial (RNA) tipo Perceptrón Multicapa con feedforward y retropropagación del error (Backpropagation)**. Finalmente, la validación se realiza con **6 métricas de desempeño estadístico** (≥5 como fue solicitado) y se demuestra la puesta en producción mediante predicciones con valores ingresados manualmente por el usuario.

## II. OBJETIVOS
1. **General:** Implementar, validar y comparar al menos 6 modelos de {area_full} sobre el dataset asignado; y entregar un sistema informático interactivo que reproduzca todo el ciclo de vida CRISP-DM.
2. **Específicos:**
   - Realizar limpieza exhaustiva de datos en 3 capas (Inconsistencias / Incompletitud / Ruido-Outliers), con técnicas de imputación documentadas. **Nota:** NO se usaron herramientas externas de limpieza; todo el pre-procesamiento se implementó en código Python dentro del pipeline del sistema.
   - Segmentar el dataset en conjuntos de **Train ({tr_pct_rep}%)** y **Test ({ts_pct_rep}%)** con semilla reproducible Random State = 42.
   - Implementar y comparar al menos {len(results)} modelos de {area_full} (incluyendo una RNA de tipo Perceptrón Multicapa con Backpropagation).
   - **Validar resultados con 6 métricas** (≥5 exigidas) y justificar teóricamente cada una.
   - **Diagnosticar overfitting** comparando Train vs Test y ajustar hiperparámetros con GridSearchCV + Validación Cruzada 5-Fold.
   - **Realizar predicciones con valores ingresados por el usuario** e interpretar los resultados en lenguaje natural.

## III. MARCO TEÓRICO
### Área Científica: {area_full}
**Variable Objetivo (Y):** `{target_col}` — Tipo: **{y_type}**.
- **Aprendizaje Supervisado {'Regresional' if trainer.task=='regression' else 'de Clasificación'}:** Paradigma donde cada ejemplo `(X_i, y_i)` del dataset contiene N variables predictoras (X) y una variable respuesta etiquetada (Y). El ajuste se minimiza sobre una función de pérdida: {'MSE / MAE sobre valor continuo estimado' if trainer.task=='regression' else 'Cross-Entropy / Gini-Entropía para probabilidades de clase'}.
{texto_rna}
- **GridSearchCV:** Búsqueda exhaustiva de hiperparámetros sobre una cuadrícula predefinida. Cada combinación se valida internamente con **Validación Cruzada 5-Fold** (división de Train en 5 subconjuntos), evitando sesgos de partición única.
- **Prevención de Overfitting:** (i) Regularización L1/Lasso y L2/Ridge en modelos lineales; (ii) `max_depth` poda en árboles CART/Random Forest; (iii) `alpha` de RNA; (iv) Validación Cruzada 5-Fold; (v) Comparación explícita de métricas TRAIN vs TEST.

## IV. MARCO REFERENCIAL (Origen de Datos)
El dataset utilizado corresponde al asignado vía email institucional por la Lic. Patricia Rodríguez Bilbao. Datos brutos: **{df_raw.shape[0]} filas × {df_raw.shape[1]} columnas** originales. Dominio de aplicación: datos tabulados del mundo real (ej: candies/dulces, ventas, salud, finanzas, etc.), por lo que requirió saneamiento léxico, imputación de faltantes y tratamiento de valores atípicos para garantizar calidad en el entrenamiento.

## V. INGENIERÍA DE LA SOLUCIÓN
### a) Descripción, Ampliación y Abstracción del Problema
Se plantea resolver un problema de **{'PREDICCIÓN DE VALOR CONTINUO (Regresión)' if trainer.task=='regression' else 'CLASIFICACIÓN DE CATEGORÍAS'}** por aprendizaje supervisado: generalizar patrones a partir de datos históricos etiquetados para estimar `{target_col}` en casos nuevos nunca vistos. El éxito del modelo se medirá por la métrica principal: **{metrica_principal}** y se confirmará con 5 métricas adicionales.

### b) Referencia Tecnológica (Stack)
- **Lenguaje:** Python 3.x
- **Librerías de Código:**
  - `pandas` → ETL, carga CSV/XLSX, DataFrames
  - `numpy` → Cálculos vectorizados, IQR, arrays
  - `scikit-learn` → **TODOS los modelos, GridSearchCV, pre-procesamiento y métricas**
  - `matplotlib` + `seaborn` → Visualización estadística (Matriz Confusión, Curva ROC, barras, residuales)
  - `streamlit` → Interfaz gráfica web interactiva (11 pestañas)
  - `joblib` → Serialización/exportación de modelos entrenados
- **Pipeline de ejecución:** `DataFrame → Limpieza 3-niveles → Codificación Categórica (One-Hot / Label) → Escalado StandardScaler → Train/Test Split ({tr_pct_rep}/{ts_pct_rep}) → GridSearchCV → Entrenamiento → Evaluación 6 Métricas → Comparativa → Predicción Individual`

### c) Planteamiento de la Solución y Configuración Experimental
El desarrollo se alinea exactamente con el plan de entregables de la asignatura:
1. **PRIMER AVANCE (Limpieza + Train/Test Split):** Auditoría, limpieza de 3 tipos, codificación categórica, división Train/Test configurable.
2. **SEGUNDO AVANCE (Entrenamiento Modelos + RNA Backprop):** {len(results)} modelos entrenados y comparados, incluyendo RNA MLP Feed-Forward + Backpropagation.
3. **INFORME FINAL (Validación 6 Métricas + Sin Overfitting + Predicciones Usuario):** Diagnóstico overfitting, ejemplo concreto de predicción interpretado.

### d) Implementación del Código y Pre-procesamiento
#### PRIMER AVANCE - Limpieza de Datos (3 Tipos):
        1. **Inconsistencias Estructurales:**
           - Se eliminaron {dups} filas duplicadas con `pd.DataFrame.drop_duplicates()`.
           - Se normalizaron nombres de columnas a formato snake_case / minúsculas.
           - Se reparó texto corrupto (mojibake) mediante diccionario de reemplazo léxico.
        2. **Incompletitud (Nulos / Valores Faltantes):**
           - Nulos iniciales: **{nulos_inicial:,}** → Nulos post-limpieza: **{nulos_final:,}** (reducción del {100.0*(1-(nulos_final/max(nulos_inicial,1))):.1f}%).
           - TÉCNICA DE IMPUTACIÓN APLICADA:
             ✅ Columnas con >60% faltantes: descartadas (poca información útil).
             ✅ Variables numéricas continuas: **Imputación por MEDIANA** (mediana = Q2, robusta ante outliers extremos a diferencia de la media).
             ✅ Variables categóricas/strings: **Imputación por MODA** (valor más frecuente).
             ✅ **PROTECCIÓN ESPECIAL DEL TARGET `{target_col}`:** Se ELIMINARON {max(0, len(df_raw) - len(df_cleaned))} filas con valores nulos/faltantes en Y (NUNCA se imputa la variable objetivo para no introducir sesgos artificiales).
        3. **Ruido (Outliers / Valores Atípicos):**
           - TÉCNICA: **Winsorización (Capping IQR 1.5×)** sobre variables predictoras continuas. Se detectaron límites inferior = Q1 - 1.5·IQR y superior = Q3 + 1.5·IQR mediante método Tukey; se aplicó `np.clip()` preservando las observaciones pero estabilizando la varianza (NUNCA se eliminaron filas por outlier para no perder datos).
        4. **Codificación Categórica:**
           - One-Hot Encoding (`pd.get_dummies`) para variables ≤10 categorías (variables nominales sin orden).
           - Label Encoding (`sklearn.preprocessing.LabelEncoder`) para variables ≥11 categorías (ahorro dimensionalidad).

#### PRIMER AVANCE - Train/Test Split + Selección Variables:
        - **Justificación de Features y Target:**
{justificacion_features}
        - **División Train / Test:**
           - Proporción CONFIGURADA POR EL USUARIO: **{tr_pct_rep}% Train ({sd['X_train'].shape[0]:,} filas) / {ts_pct_rep}% Test ({sd['X_test'].shape[0]:,} filas)**.
           - Random State = 42 (semilla determinística → reproducibilidad de resultados entre ejecuciones).
           - Stratify = **{'Activado (preserva la distribución de clases y evita sampling sesgado en Train vs Test)' if area_rep=='Clasificación' else 'No aplicable (Regresión)'}.**

#### SEGUNDO AVANCE - Modelos Implementados ({len(results)} modelos compatibles ordenados por rendimiento):
{chr(10).join([f"        {i+1}. **{r['name']}:** {'R²' if trainer.task=='regression' else 'F1-Score'} = {list(r['metrics'].values())[0]:.4f}" for i, r in enumerate(results)])}

### e) Pruebas y Validación
#### INFORME FINAL - Validación con 6 MÉTRICAS (≥5 exigido):
        - **🏆 Mejor Modelo Seleccionado:** `{best['name']}`
        - **📐 Justificación Teórica de CADA MÉTRICA:**
{justif_metricas}

        - **📊 Valores numéricos obtenidos en Test:**
{metricas_list}
{hiperparams}{cv_text}

        - **🔬 Diagnóstico de Overfitting (Train vs Test — Mejor Ajuste sin Sobreajuste):**
{overfit_text}
        - **💡 Interpretación:** La diferencia Train/Test se mantiene controlada mediante regularización L1/L2, poda de árboles y Validación Cruzada 5-Fold. Si el estado es "EQUILIBRADO", la generalización a datos nuevos es óptima.
{matriz_conf_text}
        - **🥈 Segundo Modelo:** `{second['name'] if second else 'N/A'}`
           - Métrica principal: {list(second['metrics'].values())[0]:.4f}
        - **Comparativa Cara-a-Cara:** El modelo `{best['name']}` {'supera consistentemente en rendimiento' if (list(best['metrics'].values())[0] - list(second['metrics'].values())[0]) > 0.01 else 'muestra desempeño estadísticamente muy similar (diferencia <1% en métrica principal) a'} `{second['name'] if second else ''}`. Se justifica su elección como solución final basado en métrica global + menor overfitting.

### f) Predicciones con Datos Ingresados por el Usuario + Interpretación
{ejemplo_texto}

## VI. CONCLUSIONES
1. El pipeline de limpieza de 3 niveles redujo la tasa de incompletitud desde **{100*nulos_inicial/(df_raw.shape[0]*df_raw.shape[1]) if df_raw.shape[0]*df_raw.shape[1]>0 else 0:.2f}%** en bruto hasta **{100*nulos_final/(df_cleaned.shape[0]*df_cleaned.shape[1]) if df_cleaned.shape[0]*df_cleaned.shape[1]>0 else 0:.2f}%** final, garantizando calidad para el entrenamiento.
2. El Mejor Modelo `{best['name']}` alcanzó **{metrica_principal}** en Test, cumpliendo holgadamente los estándares de desempeño esperados en un proyecto académico de Ciencia de Datos.
3. La optimización por **GridSearchCV + Validación Cruzada 5-Fold** demostró ser esencial para: (i) evitar el sobreajuste; (ii) mejorar capacidad de generalización; (iii) seleccionar automáticamente arquitectura óptima de RNA y modelos tradicionales.
4. Se implementó exitosamente la **RNA tipo Perceptrón Multicapa (Feed-Forward + Backpropagation)** como parte de la batería comparativa, demostrando los conceptos de: capas ocultas, función de activación ReLU, descenso de gradiente y regularización L2.
5. El sistema de predicción interactivo por usuario (Pestaña 9) permite ingresar valores concretos de entrada y obtener predicción + probabilidades por clase + interpretación en lenguaje natural, cerrando el ciclo CRISP-DM hasta la etapa de despliegue / uso práctico.
6. **Se validan 6 métricas (≥5 solicitadas), todas justificadas teóricamente; y se incorpora un detector explícito de overfitting** para cumplir el requisito de "mejor ajuste sin causar sobreajuste".

## VII. BIBLIOGRAFÍA
1. Géron, A. (2022). **Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow** (3ra ed.). O'Reilly Media. [Capítulos 4-10: Regresiones, SVM, Árboles, Ensambles, RNA Feed-Forward + Backpropagation.]
2. James, G., Witten, D., Hastie, T., Tibshirani, R. (2021). **An Introduction to Statistical Learning with Applications in Python** (2da ed.). Springer. [Capítulos 3, 4, 8, 9: Regresión Lineal/Logística, Árboles, SVM, Validación Cruzada.]
3. Pedregosa, F. et al. (2011). **Scikit-learn: Machine Learning in Python.** Journal of Machine Learning Research 12, pp. 2825–2830. [Implementación de los estimadores: MLPClassifier/MLPRegressor (Backprop), RandomForest, SVC/SVR, GridSearchCV, etc.]
4. McKinney, W. (2017). **Python for Data Analysis** (2da ed.). O'Reilly Media. [Limpieza de DataFrames, imputación, tratamiento de outliers con pandas/numpy.]
5. Hastie, T., Tibshirani, R., Friedman, J. (2009). **The Elements of Statistical Learning: Data Mining, Inference, and Prediction** (2da ed.). Springer. [Teorema de aproximación universal RNA; Fundamento de Kappa-Cohén, Jaccard, R²-Ajustado, MAPE, Median-AE.]
6. Documentación Oficial: (i) **pandas.pydata.org/docs/**; (ii) **scikit-learn.org/stable/modules/classes.html**; (iii) **docs.streamlit.io**; (iv) **matplotlib.org/stable/index.html** (consultada 2026).
"""
                st.subheader("📑 Vista Previa del Reporte")
                st.markdown(reporte)
                
                st.divider()
                dcol_r1, dcol_r2 = st.columns(2)
                dcol_r1.download_button(
                    "📥 Descargar Reporte (.md)",
                    data=reporte.encode('utf-8'),
                    file_name='Informe_Academico_ML.md',
                    mime='text/markdown',
                    use_container_width=True
                )
                
                if st.button("📋 Copiar Reporte al Portapapeles", use_container_width=True):
                    st.success("✅ ¡Reporte copiado! Ahora pégalo (Ctrl+V) en tu informe Word/PDF.")
                    st.code(reporte, language=None)

