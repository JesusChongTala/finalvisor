import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import json
import re
import traceback
from scipy import stats

# Canonical property names. Different files/screenings may label the same
# physical property differently (e.g. gravimetric water content stored as
# 'gravimetry', 'ewc', 'water_content'…). Map every known variant to ONE
# canonical name so it shows up as a single entry on the Y-axis instead of
# splitting into several partial-coverage columns. Keys are normalized
# (lowercased, spaces/hyphens → '_', units/symbols stripped); values are the
# label the app displays. Extend this map as new naming variants appear.
PROPERTY_ALIASES = {
    'gravimetry': 'EWC',
    'gravimetric_water_content': 'EWC',
    'ewc': 'EWC',
    'water_content': 'EWC',
    'equilibrium_water_content': 'EWC',
    'water_uptake': 'EWC',
}


def canonical_property(name):
    """Map a raw property_name to its canonical display name via PROPERTY_ALIASES.

    Unknown names are returned stripped but otherwise unchanged (case preserved),
    so only explicitly-aliased families are unified.
    """
    raw = str(name).strip()
    norm = raw.lower()
    norm = re.sub(r'[()%]', '', norm)          # drop units/symbols like "(%)"
    norm = re.sub(r'[\s\-]+', '_', norm).strip('_')
    return PROPERTY_ALIASES.get(norm, raw)


# Properties shown by default — the rest are hidden behind a "show all" toggle.
# Uses canonical names (see PROPERTY_ALIASES), so 'EWC' rather than 'gravimetry'.
PREFERRED_PROPERTIES = ['transmittance', 'absorbance', 'EWC', 'young_modulus', 'area_growth']

# Cargar los diccionarios de propiedades
try:
    hydrophilicity = json.load(open('hydrophilicity.json', 'r'))
    lipophilicity = json.load(open('lipophilicity.json', 'r'))
except FileNotFoundError:
    st.error("Error: No se encontraron los archivos .json (hydrophilicity.json, lipophilicity.json).")
    st.stop()

# Stock solutions: mg of chemical per mL of stock. Used for chemicals dosed by
# volume of a fixed-recipe stock (e.g. MPC dissolved in isopropanol at 500 mg/mL).
# Optional file — missing = no defaults.
try:
    stock_defaults = json.load(open('stock_solutions.json', 'r'))
except FileNotFoundError:
    stock_defaults = {}

# --- Configuración de la Página ---
st.set_page_config(page_title="Experiment Viewer", layout="wide")
st.title("🔬 Experimental Properties Viewer")
st.markdown("""
This tool unifies your scattered data.
Upload your Excel files to compare properties vs. concentration across multiple screenings.
""")


# --- Helpers ---

def find_chemical_key(col, keys):
    """
    Token-aware match: returns the longest key K such that the column name
    starts with K followed by '_', ' ', or end-of-string. Avoids false
    substring matches like 'DMA' inside 'EGDMA_mL' or 'PEGDMA_mL n = 14'.
    """
    candidates = sorted(keys, key=len, reverse=True)
    for key in candidates:
        pattern = r'^' + re.escape(key) + r'(?:_|\s|$)'
        if re.match(pattern, col):
            return key
    return None


def parse_chemicals_csv(chem_file):
    """
    Parse Chemicals_calc.csv into a lookup dict {name: {g_per_mL, g_per_mol}}.
    Density may be None when only molar mass is known (then only mass columns
    can be used to compute moles).
    """
    df = pd.read_csv(chem_file)
    df['g_per_mol'] = df[['g_per_mol_LO', 'g_per_mol_HI']].mean(axis=1)
    chem_dict = {}
    for _, row in df.iterrows():
        name = str(row['name'])
        rho = row['g_per_mL'] if pd.notna(row.get('g_per_mL')) else None
        mw  = row['g_per_mol'] if pd.notna(row['g_per_mol']) else None
        if mw is not None:  # molar mass is mandatory; density optional
            chem_dict[name] = {
                'g_per_mL': float(rho) if rho is not None else None,
                'g_per_mol': float(mw),
            }
    return chem_dict


def calculate_molar_ratios(df, chem_dict, pdms_vol_cols, stock_conc=None):
    """
    Add *_ratio columns relative to PDMS for every chemical with a volumetric
    (*_mL) or mass (*_mg, *_g) column.

    Mass-based path uses only molar mass (no density needed) — preferred when
    available. Volume-based path needs density OR a stock concentration.

    stock_conc: optional dict {chem_name: mg_per_mL_of_stock}. When provided
    for a chemical, the *_mL column is interpreted as volume of stock solution
    and moles are computed as n = V * (mg_per_mL * 1e-3) / M.

    r_i = n_i / n_PDMS

    Returns (modified_df, list_of_ratio_columns, warnings).
    """
    stock_conc = stock_conc or {}
    df = df.copy()

    # --- n_PDMS per row (volume path, requires density) ---
    # Persist each PDMS component's own moles as '<chem>_mol' too, so the anchor
    # (e.g. DMS-R11, DMS-U21) also shows up in the moles composition chart.
    pdms_mol_cols = []
    n_pdms = pd.Series(np.nan, index=df.index)
    for col in pdms_vol_cols:
        chem_name = col[:-3]
        chem = chem_dict.get(chem_name, {})
        rho, mw = chem.get('g_per_mL'), chem.get('g_per_mol')
        if rho is None or mw is None:
            continue
        V = pd.to_numeric(df[col], errors='coerce')
        n_col = V * rho / mw
        df[f'{chem_name}_mol'] = n_col
        pdms_mol_cols.append(f'{chem_name}_mol')
        n_pdms = n_pdms.fillna(n_col)

    if n_pdms.isna().all():
        return df, [], [], ["PDMS anchor could not be resolved — check density values in CSV."]

    pdms_set = set(pdms_vol_cols)

    # --- Group candidate columns by chemical name with priority (mass > volume) ---
    # Priority: 1=mg, 2=g, 3=mL  (lower number = preferred)
    candidates = {}   # chem_name → list of (priority, kind, col)
    for col in df.columns:
        if col in pdms_set or col == 'Total_mL':
            continue
        if col.endswith('_mg'):
            chem_name, kind, pri = col[:-3], 'mg', 1
        elif col.endswith('_g') and not col.endswith('_mg'):
            chem_name, kind, pri = col[:-2], 'g', 2
        elif col.endswith('_mL'):
            chem_name, kind, pri = col[:-3], 'mL', 3
        else:
            continue
        candidates.setdefault(chem_name, []).append((pri, kind, col))

    ratio_cols = []
    mol_cols = list(pdms_mol_cols)
    warnings = []

    for chem_name, entries in candidates.items():
        # Resolve molar mass + density (allow per-row lookup via *_type column)
        type_col = chem_name + '_type'
        if type_col in df.columns:
            rho = pd.to_numeric(df[type_col].map(
                lambda t: (chem_dict.get(str(t)) or {}).get('g_per_mL', np.nan)),
                errors='coerce')
            mw = pd.to_numeric(df[type_col].map(
                lambda t: (chem_dict.get(str(t)) or {}).get('g_per_mol', np.nan)),
                errors='coerce')
            if mw.isna().all():
                warnings.append(f"{chem_name}: types in '{type_col}' not found in chemicals CSV")
                continue
        elif chem_name in chem_dict:
            rho = chem_dict[chem_name].get('g_per_mL')
            mw  = chem_dict[chem_name].get('g_per_mol')
            if mw is None:
                warnings.append(f"{chem_name}: missing molar mass in CSV")
                continue
        else:
            warnings.append(f"{chem_name}: not found in chemicals CSV")
            continue

        # Compute n per row, preferring mass over volume
        n_chem = pd.Series(np.nan, index=df.index)
        entries.sort(key=lambda e: e[0])

        for _, kind, col in entries:
            vals = pd.to_numeric(df[col], errors='coerce')
            if kind == 'mg':
                n_new = (vals * 1e-3) / mw
            elif kind == 'g':
                n_new = vals / mw
            else:  # 'mL'
                if chem_name in stock_conc:
                    # Stock solution: V_mL × (mg/mL × 1e-3 g/mg) / M
                    mg_per_mL = stock_conc[chem_name]
                    n_new = vals * (mg_per_mL * 1e-3) / mw
                elif rho is None or (hasattr(rho, '__len__') and pd.isna(rho).all()):
                    if vals.notna().any():
                        warnings.append(
                            f"{chem_name}: '{col}' has data but no density in CSV "
                            f"({vals.notna().sum()} rows skipped). "
                            f"Add a stock concentration in the sidebar."
                        )
                    continue
                else:
                    n_new = vals * rho / mw
            n_chem = n_chem.fillna(n_new)

        if n_chem.notna().any():
            ratio_col = f'{chem_name}_ratio'
            df[ratio_col] = np.where(n_pdms > 0, np.round(n_chem / n_pdms, 4), np.nan)
            ratio_cols.append(ratio_col)

            # Persist the absolute moles too (n = V*rho/M, or mass/M, or via stock)
            mol_col = f'{chem_name}_mol'
            df[mol_col] = n_chem
            mol_cols.append(mol_col)

    return df, ratio_cols, mol_cols, warnings


def read_properties_smart(file):
    """
    Read a properties Excel.
    Prefer the 'measurements' sheet when present (long-format master file).
    """
    try:
        xl = pd.ExcelFile(file)
        if 'measurements' in xl.sheet_names:
            return pd.read_excel(file, sheet_name='measurements')
    except Exception:
        pass
    return pd.read_excel(file)


def detect_long_properties(df):
    """Check if df is in long format with property_name + value columns."""
    return 'property_name' in df.columns and 'value' in df.columns


def pivot_long_properties(df, id_col):
    """
    Pivot long-format properties to wide. Returns (wide_df, metadata) where
    metadata maps each pivoted column to {property, state, unit, method, is_uncertainty}.
    Column names use format '{property}__{state}' (or just '{property}' if no state).
    Uncertainty columns are '{property}__{state}__unc'.
    """
    state_col = 'sample_state' if 'sample_state' in df.columns else None
    df = df.copy()

    if id_col not in df.columns:
        raise ValueError(f"Properties file is missing the ID column '{id_col}'.")

    df['property_name'] = df['property_name'].astype(str).str.strip().map(canonical_property)
    if state_col:
        df[state_col] = df[state_col].fillna('unspecified').astype(str).str.strip()
        df['_key'] = df['property_name'] + '__' + df[state_col]
    else:
        df['_key'] = df['property_name']

    wide = (df.pivot_table(index=id_col, columns='_key',
                           values='value', aggfunc='first')
              .reset_index())
    wide.columns.name = None

    metadata = {}
    for key, grp in df.groupby('_key'):
        first = grp.iloc[0]
        metadata[key] = {
            'property': first['property_name'],
            'state': first[state_col] if state_col else None,
            'unit': str(first.get('unit', '') or '').strip(),
            'method': str(first.get('method', '') or '').strip(),
            'is_uncertainty': False,
        }

    if 'uncertainty' in df.columns:
        unc = df[df['uncertainty'].notna()].copy()
        if len(unc):
            unc['_unc_key'] = unc['_key'] + '__unc'
            wide_unc = (unc.pivot_table(index=id_col, columns='_unc_key',
                                        values='uncertainty', aggfunc='first')
                          .reset_index())
            wide_unc.columns.name = None
            wide = wide.merge(wide_unc, on=id_col, how='left')
            for unc_key, grp in unc.groupby('_unc_key'):
                value_key = unc_key[:-5] if unc_key.endswith('__unc') else unc_key
                metadata[unc_key] = {
                    **metadata.get(value_key, {}),
                    'is_uncertainty': True,
                }

    return wide, metadata


def compute_derived_properties(wide_df, metadata):
    """
    Compute derived properties on top of the pivoted wide frame.
    Currently: area_growth__after_swelling = (A_wet - A_dry) / A_dry × 100
    Returns (modified_df, modified_metadata).
    """
    df = wide_df.copy()
    bs_col = 'area_pixels__before_swelling'
    as_col = 'area_pixels__after_swelling'

    if bs_col in df.columns and as_col in df.columns:
        bs = pd.to_numeric(df[bs_col], errors='coerce')
        as_ = pd.to_numeric(df[as_col], errors='coerce')
        growth = np.where(bs > 0, (as_ - bs) / bs * 100, np.nan)
        if np.isfinite(growth).any():
            col_name = 'area_growth__after_swelling'
            df[col_name] = growth
            metadata[col_name] = {
                'property': 'area_growth',
                'state': 'after_swelling',
                'unit': '%',
                'method': 'computed: (area_after − area_before) / area_before × 100',
                'is_uncertainty': False,
            }
    return df, metadata


def parse_wide_metadata(df, id_col):
    """
    Reconstruct properties metadata from a pre-pivoted wide file using '__' separator.
    Returns metadata dict or empty dict if no recognizable columns.
    """
    state_aliases = {
        'AC': 'after_crosslinking', 'AS': 'after_swelling',
        'BC': 'before_crosslinking', 'BS': 'before_swelling',
        'unspec': 'unspecified',
    }
    metadata = {}
    for col in df.columns:
        if col == id_col or '__' not in col:
            continue
        parts = col.split('__')
        if len(parts) == 2:
            prop, state = parts
            metadata[col] = {
                'property': canonical_property(prop),
                'state': state_aliases.get(state, state),
                'unit': '', 'method': '', 'is_uncertainty': False,
            }
        elif len(parts) == 3 and parts[2] == 'unc':
            prop, state, _ = parts
            value_key = f"{prop}__{state}"
            metadata[col] = {
                **metadata.get(value_key, {'property': canonical_property(prop),
                                           'state': state_aliases.get(state, state),
                                           'unit': '', 'method': ''}),
                'is_uncertainty': True,
            }
    return metadata


def correlation_analysis(x, y):
    """
    Compute linear fit + correlation stats on two arrays. Returns dict or None
    if there's not enough data (need ≥ 3 paired non-NaN values).
    """
    df = pd.DataFrame({'x': pd.to_numeric(x, errors='coerce'),
                       'y': pd.to_numeric(y, errors='coerce')}).dropna()
    n = len(df)
    if n < 3:
        return None
    xv, yv = df['x'].values, df['y'].values
    if np.allclose(xv, xv[0]) or np.allclose(yv, yv[0]):
        return None  # zero variance
    slope, intercept, r, p, _ = stats.linregress(xv, yv)
    rho, p_rho = stats.spearmanr(xv, yv)
    return {
        'n': n, 'slope': slope, 'intercept': intercept,
        'r': r, 'r2': r**2, 'p': p,
        'spearman_rho': rho, 'spearman_p': p_rho,
        'x_min': float(xv.min()), 'x_max': float(xv.max()),
    }


def fmt_value(v, sig=3):
    """Format a numeric value with adaptive precision (avoids losing tiny values)."""
    if pd.isna(v):
        return "—"
    try:
        v = float(v)
    except (TypeError, ValueError):
        return str(v)
    if v == 0:
        return "0"
    abs_v = abs(v)
    if abs_v >= 1:
        return f"{v:.{sig}f}"
    if abs_v >= 1e-3:
        return f"{v:.4f}"
    return f"{v:.2e}"


# --- Paso 1: Carga de Datos Múltiples ---
st.sidebar.header("1. Upload Data")

archivos_conc = st.sidebar.file_uploader("Upload Concentrations Excels", type=["xlsx", "xls"], accept_multiple_files=True)
archivos_prop = st.sidebar.file_uploader("Upload Properties Excels (e.g. Young, Transparency)", type=["xlsx", "xls"], accept_multiple_files=True)

if archivos_conc and archivos_prop:
    if len(archivos_conc) != len(archivos_prop):
        st.sidebar.error("⚠️ Sube la misma cantidad de archivos de Concentración y de Propiedades.")
        st.stop()

    try:
        archivos_conc = sorted(archivos_conc, key=lambda x: x.name)
        archivos_prop = sorted(archivos_prop, key=lambda x: x.name)

        df_peek = pd.read_excel(archivos_conc[0])
        df_peek.columns = df_peek.columns.astype(str).str.strip()

        # --- Paso 2: Configuración Inicial ---
        st.sidebar.header("2. Variable Configuration")
        id_col = st.sidebar.selectbox(
            "Select ID column (common in all files):",
            df_peek.columns,
            help="The unique row key shared by both concentration and properties files."
        )

        # Auto-detect grouping column when working with a single master file
        candidate_group_cols = [c for c in df_peek.columns
                                if c != id_col and c.lower() in
                                ('screening', 'screening_id', 'experiment', 'batch')]
        group_col = None
        if len(archivos_conc) == 1 and candidate_group_cols:
            group_col = st.sidebar.selectbox(
                "Group/Color column:",
                ["(filename)"] + candidate_group_cols,
                index=1,
                help="Used to color points and aggregate the composition bar chart."
            )
            if group_col == "(filename)":
                group_col = None

        # --- Screening Filter ---
        st.sidebar.markdown("---")
        st.sidebar.subheader("🔍 Filter")

        # Determine filter source: group column (single-file) or filenames (multi-file)
        if group_col and group_col in df_peek.columns:
            filter_source_label = group_col
            available_groups_raw = df_peek[group_col].dropna().unique().tolist()
            try:
                available_groups = sorted(available_groups_raw, key=lambda x: (isinstance(x, str), x))
            except TypeError:
                available_groups = sorted([str(g) for g in available_groups_raw])
        elif len(archivos_conc) > 1:
            filter_source_label = "file"
            available_groups = sorted([
                f.name.replace('.xlsx', '').replace('.xls', '') for f in archivos_conc
            ])
        else:
            filter_source_label = None
            available_groups = []

        selected_groups = available_groups  # default = all

        if available_groups:
            available_str = [str(g) for g in available_groups]
            session_key = f'filter_{filter_source_label}'

            # Initialize / reconcile session state with current options
            if session_key not in st.session_state:
                st.session_state[session_key] = list(available_str)
            else:
                st.session_state[session_key] = [
                    v for v in st.session_state[session_key] if v in available_str
                ]
                if not st.session_state[session_key]:
                    st.session_state[session_key] = list(available_str)

            def _set_filter(key, values):
                st.session_state[key] = list(values)

            c1, c2, c3 = st.sidebar.columns(3)
            c1.button("All", key=f'btn_all_{filter_source_label}',
                      on_click=_set_filter, args=(session_key, available_str),
                      use_container_width=True)
            c2.button("None", key=f'btn_none_{filter_source_label}',
                      on_click=_set_filter, args=(session_key, []),
                      use_container_width=True)
            c3.button("Last 5", key=f'btn_last_{filter_source_label}',
                      on_click=_set_filter, args=(session_key, available_str[-5:]),
                      use_container_width=True)

            selected_str = st.sidebar.multiselect(
                f"Show {filter_source_label}(s):",
                options=available_str,
                key=session_key,
                help=f"Select which {filter_source_label}(s) to include in the analysis. "
                     "Use the buttons above for quick presets."
            )

            # Map back to original types (numeric vs string)
            str_to_orig = {str(g): g for g in available_groups}
            selected_groups = [str_to_orig[s] for s in selected_str]

            n_sel = len(selected_groups)
            n_tot = len(available_groups)
            if n_sel == 0:
                st.sidebar.error(f"⚠️ No {filter_source_label}s selected — pick at least one.")
                st.stop()
            elif n_sel == n_tot:
                st.sidebar.caption(f"📋 All {n_tot} {filter_source_label}(s) included.")
            else:
                st.sidebar.caption(f"📋 {n_sel}/{n_tot} {filter_source_label}(s) selected.")

        # --- Paso 3: Calculadora de Composición ---
        st.sidebar.markdown("---")
        st.sidebar.subheader("🧮 Group Calculator")
        usar_calculadora = st.sidebar.checkbox("Enable calculation (Hydrophilic/Hydrophobic)")

        # --- Paso 4: Calculadora de Ratios Molares ---
        st.sidebar.markdown("---")
        st.sidebar.subheader("⚗️ Molar Ratio Calculator")
        usar_ratios = st.sidebar.checkbox("Calculate molar ratios (relative to PDMS)")

        pdms_vol_cols         = []
        chem_dict             = {}
        ratio_cols_global     = []
        mol_cols_global       = []
        ratio_warnings_global = []
        stock_conc_dict       = {}

        if usar_ratios:
            chem_upload = st.sidebar.file_uploader("Upload Chemicals_calc.csv", type=["csv"])
            if chem_upload:
                chem_dict = parse_chemicals_csv(chem_upload)

                ml_cols = [c for c in df_peek.columns
                           if c.endswith('_mL') and c != 'Total_mL']

                if not ml_cols:
                    st.sidebar.warning("No *_mL columns found in the first concentration file.")
                else:
                    pdms_auto = [c for c in ml_cols if c[:-3] in chem_dict
                                 and (c[:-3].startswith('DMS') or 'PDMS' in c)]
                    pdms_vol_cols = st.sidebar.multiselect(
                        "PDMS column(s) — select all if the type varies per row:",
                        options=ml_cols,
                        default=pdms_auto,
                        help="For each row the first non-empty PDMS column is used as anchor."
                    )
                    if pdms_vol_cols:
                        detected = [c for c in pdms_vol_cols if c[:-3] in chem_dict]
                        unknown  = [c for c in pdms_vol_cols if c[:-3] not in chem_dict]
                        if detected:
                            st.sidebar.info("✅ Auto-resolved: " +
                                            ", ".join(f"**{c[:-3]}**" for c in detected))
                        if unknown:
                            st.sidebar.warning("⚠️ Not found in CSV: " + ", ".join(unknown))

                # Auto-detect chemicals that have a _mL column with data but no density
                # in the CSV — these are candidates for stock-solution configuration.
                stock_candidates = []
                for c in ml_cols:
                    chem_name = c[:-3]
                    if c in pdms_vol_cols:
                        continue
                    if chem_name in chem_dict and chem_dict[chem_name].get('g_per_mL') is None:
                        vals = pd.to_numeric(df_peek[c], errors='coerce')
                        if vals.notna().any():
                            stock_candidates.append((chem_name, c, int(vals.notna().sum())))

                if stock_candidates:
                    # Apply defaults from stock_solutions.json silently
                    for chem_name, _, _ in stock_candidates:
                        if chem_name in stock_defaults:
                            stock_conc_dict[chem_name] = float(stock_defaults[chem_name])

                    auto_count = sum(1 for cn, _, _ in stock_candidates
                                     if cn in stock_defaults)
                    n_total = len(stock_candidates)
                    label = f"📦 Stock solutions ({auto_count}/{n_total} preset)"
                    with st.sidebar.expander(label, expanded=(auto_count < n_total)):
                        st.caption(
                            "Volume columns of chemicals without density in CSV. "
                            "Defaults loaded from `stock_solutions.json` — override "
                            "here for ad-hoc cases."
                        )
                        for chem_name, col, n_rows in stock_candidates:
                            default = float(stock_defaults.get(chem_name, 0.0))
                            mg_per_ml = st.number_input(
                                f"**{chem_name}** ({col}, {n_rows} rows) — mg/mL of stock:",
                                min_value=0.0, value=default, step=10.0,
                                key=f"stock_{chem_name}",
                                help="Leave at 0 to skip. "
                                     "Persistent defaults live in `stock_solutions.json`."
                            )
                            if mg_per_ml > 0:
                                stock_conc_dict[chem_name] = mg_per_ml
                            elif chem_name in stock_conc_dict:
                                del stock_conc_dict[chem_name]
            else:
                st.sidebar.info("Upload Chemicals_calc.csv to enable ratio calculation.")

        # --- Paso 5: Procesamiento de Múltiples Archivos ---
        lista_dfs_unidos = []
        concentration_columns = set()
        properties_metadata = {}        # column → {property, state, unit, ...}
        merge_stats = []

        for conc_file, prop_file in zip(archivos_conc, archivos_prop):
            df_c = pd.read_excel(conc_file)
            df_p_raw = read_properties_smart(prop_file)

            df_c.columns   = df_c.columns.astype(str).str.strip()
            df_p_raw.columns = df_p_raw.columns.astype(str).str.strip()

            # Detect long-format properties → pivot internally
            if detect_long_properties(df_p_raw):
                if id_col not in df_p_raw.columns:
                    st.error(f"Long-format properties file '{prop_file.name}' is missing "
                             f"the ID column '{id_col}'. Available columns: "
                             f"{', '.join(df_p_raw.columns)}")
                    st.stop()
                df_p, meta = pivot_long_properties(df_p_raw, id_col)
                properties_metadata.update(meta)
            else:
                df_p = df_p_raw
                wide_meta = parse_wide_metadata(df_p, id_col)
                if wide_meta:
                    properties_metadata.update(wide_meta)

            # Compute derived properties (area_growth from before/after_swelling pixels)
            df_p, properties_metadata = compute_derived_properties(df_p, properties_metadata)

            # Hydrophilic / Hydrophobic group calculator (token-aware, _mL only)
            if usar_calculadora:
                df_c['Total_Hydrophilic'] = 0.0
                df_c['Total_Hydrophobic'] = 0.0

                for col in df_c.columns:
                    if col == id_col:
                        continue
                    # Only volumetric columns contribute (avoids MPC_mg + MPC_mL double count)
                    if not col.endswith('_mL') or col == 'Total_mL':
                        continue

                    key_h = find_chemical_key(col, hydrophilicity.keys())
                    if key_h:
                        val = pd.to_numeric(df_c[col], errors='coerce').fillna(0)
                        df_c['Total_Hydrophilic'] += val * hydrophilicity[key_h]

                    key_l = find_chemical_key(col, lipophilicity.keys())
                    if key_l:
                        val = pd.to_numeric(df_c[col], errors='coerce').fillna(0)
                        df_c['Total_Hydrophobic'] += val * lipophilicity[key_l]

                df_c['Ratio_Philic_Phobic'] = np.where(
                    df_c['Total_Hydrophobic'] == 0, np.nan,
                    df_c['Total_Hydrophilic'] / df_c['Total_Hydrophobic']
                )

            if usar_ratios and chem_dict and pdms_vol_cols:
                active_pdms = [c for c in pdms_vol_cols if c in df_c.columns]
                if active_pdms:
                    df_c, ratio_cols, mol_cols, ratio_warnings = calculate_molar_ratios(
                        df_c, chem_dict, active_pdms, stock_conc=stock_conc_dict)
                    for rc in ratio_cols:
                        if rc not in ratio_cols_global:
                            ratio_cols_global.append(rc)
                    for mc in mol_cols:
                        if mc not in mol_cols_global:
                            mol_cols_global.append(mc)
                    if ratio_warnings:
                        ratio_warnings_global.extend(ratio_warnings)

            n_conc, n_prop = len(df_c), len(df_p)
            df_temp_merged = pd.merge(df_c, df_p, on=id_col, how='inner')
            n_merged = len(df_temp_merged)
            merge_stats.append((conc_file.name, n_conc, n_prop, n_merged))

            nombre_limpio = conc_file.name.replace('.xlsx', '').replace('.xls', '')
            df_temp_merged['Screening_File'] = nombre_limpio

            concentration_columns.update(df_c.columns)
            lista_dfs_unidos.append(df_temp_merged)

        df_merged = pd.concat(lista_dfs_unidos, ignore_index=True)
        df_merged_unfiltered = df_merged.copy()  # for coverage analysis
        n_total_before_filter = len(df_merged)

        # Apply screening filter
        if available_groups and len(selected_groups) < len(available_groups):
            if group_col and group_col in df_merged.columns:
                df_merged = df_merged[df_merged[group_col].isin(selected_groups)].reset_index(drop=True)
            elif filter_source_label == "file":
                df_merged = df_merged[df_merged['Screening_File'].isin(selected_groups)].reset_index(drop=True)

        n_after_filter = len(df_merged)

        if n_after_filter == 0:
            st.warning(f"⚠️ No samples match the current filter. "
                       f"Try selecting more {filter_source_label}(s) in the sidebar.")
            st.stop()

        # Active filter banner in main panel
        if available_groups and len(selected_groups) < len(available_groups):
            preview = ", ".join(str(g) for g in selected_groups[:8])
            if len(selected_groups) > 8:
                preview += f", … (+{len(selected_groups)-8} more)"
            st.info(
                f"🔍 **Filter active** — showing **{n_after_filter}** of {n_total_before_filter} samples "
                f"from {len(selected_groups)} {filter_source_label}(s): {preview}"
            )

        # Merge diagnostics
        with st.sidebar.expander("ℹ️ Merge diagnostics", expanded=False):
            for name, nc, np_, nm in merge_stats:
                drop_c = nc - nm
                drop_p = np_ - nm
                msg = f"**{name}**  \nconc={nc}, prop={np_}, merged={nm}"
                if drop_c > 0 or drop_p > 0:
                    msg += f"  \n⚠️ dropped: {drop_c} from conc, {drop_p} from prop"
                st.markdown(msg)

        if usar_calculadora:
            st.sidebar.success("✅ Multi-file Calculation Complete!")
        if usar_ratios and ratio_cols_global:
            st.sidebar.success(f"✅ {len(ratio_cols_global)} molar ratio column(s) added.")
        if usar_ratios and ratio_warnings_global:
            # Deduplicate warnings while preserving order
            seen = set()
            unique_warnings = [w for w in ratio_warnings_global
                               if not (w in seen or seen.add(w))]
            with st.sidebar.expander(f"⚠️ {len(unique_warnings)} ratio warning(s)", expanded=False):
                for w in unique_warnings:
                    st.markdown(f"- {w}")

        # --- Paso 6: Configuración de Visualización Avanzada ---
        st.sidebar.markdown("---")
        st.sidebar.header("3. Advanced Visualization")

        # Identify column families
        all_cols        = list(df_merged.columns)
        properties_cols = [c for c in all_cols
                           if c not in concentration_columns and c != 'Screening_File']

        # Build property → list of (state, column, meta, n_samples) map
        # Sorted by sample count descending so the state with most data is default
        props_by_name = {}
        for col, meta in properties_metadata.items():
            if meta.get('is_uncertainty'):
                continue
            if col in df_merged.columns:
                n_samples = int(pd.to_numeric(df_merged[col], errors='coerce').notna().sum())
                props_by_name.setdefault(meta['property'], []).append(
                    (meta.get('state'), col, meta, n_samples))
        for prop, entries in props_by_name.items():
            entries.sort(key=lambda e: -e[3])  # most-samples-first

        all_property_names = sorted(props_by_name.keys())
        # Default view: only the 4 properties used for analysis
        preferred_present = [p for p in PREFERRED_PROPERTIES if p in all_property_names]
        if preferred_present:
            show_all_props = st.sidebar.checkbox(
                f"Show all properties ({len(all_property_names)} total)",
                value=False,
                help=f"By default only {len(preferred_present)} key properties are shown: "
                     + ", ".join(preferred_present)
            )
            property_names = all_property_names if show_all_props else preferred_present
        else:
            property_names = all_property_names

        has_properties_metadata = bool(property_names)

        # Concentration / ratio / derived columns available for axes
        non_property_cols = [c for c in all_cols
                             if c not in {id_col, 'Screening_File'}
                             and c not in properties_metadata
                             and c not in properties_cols]

        def _pick_property(prefix, default_idx=0):
            """Hierarchical Property → State picker. Returns (col_name, label, unc_col)."""
            prop = st.sidebar.selectbox(f"{prefix} Property:", property_names,
                                        index=min(default_idx, len(property_names) - 1),
                                        key=f"{prefix}_property")
            states = props_by_name[prop]  # list of (state, col, meta, n_samples)
            if len(states) > 1:
                # State labels include sample count for transparency about coverage
                state_options = [f"{(s or '(no state)')}  ·  {n} samples"
                                 for s, _, _, n in states]
                state_choice = st.sidebar.selectbox(f"{prefix} State:", state_options,
                                                    key=f"{prefix}_state",
                                                    help="States are sorted by sample count "
                                                         "(most data first).")
                idx = state_options.index(state_choice)
                state, col, meta, _ = states[idx]
                state_label = state or '(no state)'
            else:
                state, col, meta, _ = states[0]
                state_label = state or '(no state)'
            unit = meta.get('unit', '')
            unc_col = f"{col}__unc" if f"{col}__unc" in df_merged.columns else None
            label = f"{prop} [{state_label}]" + (f" ({unit})" if unit else "")
            return col, label, unc_col

        x_col_label = y_col_label = None
        parametro_error_x = parametro_error_y = None

        if has_properties_metadata:
            # --- Y-axis: always a property (the natural use case) ---
            st.sidebar.markdown("**Y-axis**")
            default_y = next((i for i, p in enumerate(property_names)
                              if 'young' in p.lower()), 0)
            y_col, y_col_label, parametro_error_y = _pick_property("Y", default_idx=default_y)

            # --- X-axis: choose between concentration/ratio or a property ---
            st.sidebar.markdown("**X-axis**")
            x_mode = st.sidebar.radio(
                "X-axis source:", ["Concentration / Ratio", "Property"],
                horizontal=True, key="x_mode"
            )
            if x_mode == "Property":
                x_col, x_col_label, parametro_error_x = _pick_property("X", default_idx=1)
            else:
                if not non_property_cols:
                    st.sidebar.error("No concentration columns available for X-axis.")
                    st.stop()
                x_col = st.sidebar.selectbox("X column:", non_property_cols, index=0,
                                             key="x_conc_col")
                x_col_label = x_col
        else:
            # Legacy fallback: flat dropdowns when no property metadata is detected
            plot_axis_options = [c for c in all_cols if c not in {id_col, 'Screening_File'}]
            x_col = st.sidebar.selectbox("X Axis:", plot_axis_options, index=0)
            y_col = st.sidebar.selectbox("Y Axis:", plot_axis_options,
                                         index=min(1, len(plot_axis_options) - 1))
            x_col_label = x_col
            y_col_label = y_col

            unc_candidates = [c for c in plot_axis_options
                              if 'unc' in c.lower() or 'std' in c.lower() or 'sd' in c.lower()]
            if 'young' in x_col.lower() or 'modulus' in x_col.lower():
                e = st.sidebar.selectbox(f"SD for X ({x_col}):",
                                         ["None"] + (unc_candidates or plot_axis_options))
                if e != "None": parametro_error_x = e
            if 'young' in y_col.lower() or 'modulus' in y_col.lower():
                e = st.sidebar.selectbox(f"SD for Y ({y_col}):",
                                         ["None"] + (unc_candidates or plot_axis_options))
                if e != "None": parametro_error_y = e

        # Allow user to override auto-detected uncertainty
        if has_properties_metadata and (parametro_error_x or parametro_error_y):
            with st.sidebar.expander("Error bars (auto-detected)", expanded=False):
                if parametro_error_y:
                    st.caption(f"Y: {parametro_error_y}")
                    if st.checkbox("Disable Y error bars", key="disable_y_err"):
                        parametro_error_y = None
                if parametro_error_x:
                    st.caption(f"X: {parametro_error_x}")
                    if st.checkbox("Disable X error bars", key="disable_x_err"):
                        parametro_error_x = None

        st.sidebar.markdown("#### 🎨 Multi-Component Analysis")

        # Component candidates: numeric, came from conc files, exclude IDs/totals/group/ratios/derived
        numeric_cols    = df_merged.select_dtypes(include=np.number).columns.tolist()
        derived_exclude = {'Total_mL', 'Total_Hydrophilic', 'Total_Hydrophobic',
                           'Ratio_Philic_Phobic'} | set(ratio_cols_global)
        component_excl  = ({id_col, x_col, y_col, 'Screening_File'}
                           | derived_exclude
                           | (set(candidate_group_cols) if group_col else set())
                           | set(properties_cols))
        component_candidates = [c for c in numeric_cols
                                if c in concentration_columns
                                and c not in component_excl
                                and (c.endswith('_mL') or c.endswith('_mg'))]

        multi_color_cols = st.sidebar.multiselect(
            "Select components for composition (volumes only):",
            options=component_candidates,
            help="Only volumetric columns from the concentration file are eligible."
        )

        # Decide the effective grouping column for color/aggregation
        effective_group = group_col if group_col else 'Screening_File'
        # Ensure stable string type for plotting
        df_merged[effective_group] = df_merged[effective_group].astype(str)
        unique_groups = df_merged[effective_group].nunique()

        # Bar chart aggregation toggle
        bar_aggregate = False
        bar_display = "Stacked"
        if multi_color_cols:
            bar_aggregate = st.sidebar.checkbox(
                "Aggregate bar chart by group (mean composition)",
                value=(len(df_merged) > 50),
                help="Recommended for >50 samples — otherwise the per-sample bars become unreadable."
            )
            bar_display = st.sidebar.radio(
                "Bar chart display",
                ["Stacked", "Grouped (log scale)", "100% stacked"],
                index=0,
                help=("Grouped (log scale) puts each component in its own bar on a "
                      "log axis so tiny components (µmol/nmol) stay visible next to "
                      "large ones. 100% stacked normalizes each bar to its share."),
            )

        st.write("---")

        # --- Tabla de Ratios Molares ---
        if usar_ratios and ratio_cols_global:
            ratio_cols_present = [c for c in ratio_cols_global if c in df_merged.columns]
            if ratio_cols_present:
                with st.expander("⚗️ Molar Ratios (relative to PDMS)", expanded=False):
                    st.markdown(
                        "Each value is the molar ratio of the component relative to PDMS:  "
                        r"$r_i = \dfrac{V_i \cdot \rho_i / M_i}{V_\text{PDMS} \cdot \rho_\text{PDMS} / M_\text{PDMS}}$"
                    )
                    display_ratio_cols = [c for c in [id_col, effective_group] + ratio_cols_present
                                          if c in df_merged.columns]
                    st.dataframe(
                        df_merged[display_ratio_cols].style.format(
                            {c: "{:.3f}" for c in ratio_cols_present}, na_rep="—"
                        ),
                        use_container_width=True
                    )

                    ratio_means = (
                        df_merged.groupby(effective_group)[ratio_cols_present]
                        .mean()
                        .reset_index()
                        .melt(id_vars=effective_group, var_name='Component', value_name='Mean Ratio')
                    )
                    ratio_means['Component'] = ratio_means['Component'].str.replace('_ratio', '', regex=False)
                    fig_ratios = px.bar(
                        ratio_means,
                        x='Component', y='Mean Ratio',
                        color=effective_group,
                        barmode='group',
                        title=f'Mean Molar Ratios per {effective_group} (relative to PDMS)',
                        labels={'Mean Ratio': 'Molar ratio (mol/mol PDMS)'}
                    )
                    st.plotly_chart(fig_ratios, use_container_width=True)

        # --- Property coverage banner ---
        # Compute which screenings have data for the chosen axes (uses UNfiltered data)
        if group_col and group_col in df_merged_unfiltered.columns:
            coverage_group = group_col
        else:
            coverage_group = 'Screening_File'

        def _screenings_with_data(col):
            if col not in df_merged_unfiltered.columns:
                return None
            sub = df_merged_unfiltered[[coverage_group, col]].copy()
            sub[col] = pd.to_numeric(sub[col], errors='coerce')
            counts = sub.dropna(subset=[col]).groupby(coverage_group).size()
            return counts.sort_index()

        cov_y = _screenings_with_data(y_col)
        cov_x = _screenings_with_data(x_col) if x_col in properties_metadata else None

        if cov_y is not None and len(cov_y) > 0:
            all_groups_unf = df_merged_unfiltered[coverage_group].dropna().unique()
            n_total_groups = len(all_groups_unf)

            # Intersection if both axes are properties
            if cov_x is not None and len(cov_x) > 0:
                groups_with_data = sorted(set(cov_y.index) & set(cov_x.index))
                label_what = f"**{y_col_label or y_col}**  AND  **{x_col_label or x_col}**"
            else:
                groups_with_data = list(cov_y.index)
                label_what = f"**{y_col_label or y_col}**"

            n_with = len(groups_with_data)
            preview = ", ".join(str(g) for g in groups_with_data[:12])
            if n_with > 12:
                preview += f", … (+{n_with - 12})"

            banner_col1, banner_col2 = st.columns([4, 1])
            with banner_col1:
                if n_with == 0:
                    st.warning(f"⚠️ No screening has data for {label_what}.")
                else:
                    st.info(
                        f"📊 **{n_with}** of {n_total_groups} {coverage_group}s have data "
                        f"for {label_what}: {preview}"
                    )
            with banner_col2:
                if n_with > 0 and n_with < n_total_groups:
                    if available_groups:
                        session_key_filter = f'filter_{filter_source_label}'

                        def _apply_data_filter(key, values):
                            st.session_state[key] = [str(v) for v in values]

                        st.button(
                            f"🎯 Filter to {n_with}",
                            key=f"btn_filter_data_{y_col}",
                            on_click=_apply_data_filter,
                            args=(session_key_filter, groups_with_data),
                            use_container_width=True,
                            help=f"Show only the {n_with} {coverage_group}s "
                                 f"with data for the selected propert(y/ies)."
                        )

        # --- Lógica de Renderizado de Gráficos ---
        # Build a working frame (no mutation of df_merged with hover/dominant/bar_id)
        df_view = df_merged.copy()

        x_axis_title = x_col_label or x_col
        y_axis_title = y_col_label or y_col

        if not multi_color_cols:
            st.subheader(f"Chart: {y_axis_title}  vs  {x_axis_title}")
            fig = px.scatter(
                df_view,
                x=x_col,
                y=y_col,
                error_x=parametro_error_x,
                error_y=parametro_error_y,
                color=effective_group if unique_groups > 1 else None,
                hover_data=[id_col],
                text=id_col,
                title=f"Relationship: {y_axis_title} vs {x_axis_title}",
                labels={x_col: x_axis_title, y_col: y_axis_title}
            )
            fig.update_traces(
                textposition='top center',
                marker=dict(size=14, line=dict(width=1, color='DarkSlateGrey'))
            )

            # Correlation analysis (only when both axes are numeric)
            corr = correlation_analysis(df_view[x_col], df_view[y_col])
            if corr:
                x_line = np.linspace(corr['x_min'], corr['x_max'], 50)
                y_line = corr['slope'] * x_line + corr['intercept']
                fig.add_scatter(
                    x=x_line, y=y_line, mode='lines',
                    line=dict(color='white', dash='dash', width=2),
                    name=f"Linear fit (R²={corr['r2']:.3f})",
                    hoverinfo='skip',
                )

            st.plotly_chart(fig, use_container_width=True)

            if corr:
                sign = "+" if corr['intercept'] >= 0 else "−"
                eq = (f"`{y_axis_title.split(' (')[0]}` = "
                      f"**{corr['slope']:.3f}** × `{x_axis_title.split(' (')[0]}` "
                      f"{sign} **{abs(corr['intercept']):.3f}**")
                col_a, col_b, col_c, col_d = st.columns(4)
                col_a.metric("N (paired)", f"{corr['n']}")
                col_b.metric("R²", f"{corr['r2']:.3f}")
                col_c.metric("Pearson r", f"{corr['r']:.3f}",
                             help=f"p = {corr['p']:.2e}")
                col_d.metric("Spearman ρ", f"{corr['spearman_rho']:.3f}",
                             help=f"p = {corr['spearman_p']:.2e}")
                st.markdown(f"**Linear fit:**  {eq}")

        else:
            st.subheader(f"Composition view: {y_axis_title}  vs  {x_axis_title}")

            palette = px.colors.qualitative.Plotly + px.colors.qualitative.D3 + px.colors.qualitative.Set3
            color_map = {c: palette[i % len(palette)] for i, c in enumerate(multi_color_cols)}

            df_view['Dominant_Component'] = df_view[multi_color_cols].fillna(0).idxmax(axis=1)

            # Vectorized hover text builder
            hover_lines = [f"<b>Sample: {sid}</b><br><b>Group: {grp}</b><br><br><b>Composition:</b>"
                           for sid, grp in zip(df_view[id_col].astype(str), df_view[effective_group])]
            for col in multi_color_cols:
                vals = [fmt_value(v) for v in df_view[col]]
                hover_lines = [h + f"<br>- {col}: {v}" for h, v in zip(hover_lines, vals)]

            ratio_cols_present = [c for c in ratio_cols_global if c in df_view.columns]
            if ratio_cols_present:
                hover_lines = [h + "<br><br><b>Molar ratios:</b>" for h in hover_lines]
                for rc in ratio_cols_present:
                    label = rc.replace('_ratio', '')
                    vals = [fmt_value(v) for v in df_view[rc]]
                    hover_lines = [h + f"<br>- {label}: {v}" for h, v in zip(hover_lines, vals)]

            df_view['Composition_Hover'] = hover_lines

            fig_scatter = px.scatter(
                df_view,
                x=x_col,
                y=y_col,
                error_x=parametro_error_x,
                error_y=parametro_error_y,
                color='Dominant_Component',
                symbol=effective_group if unique_groups > 1 else None,
                color_discrete_map=color_map,
                custom_data=['Composition_Hover'],
                text=id_col,
                title=f"{y_axis_title} vs {x_axis_title} (Color = Dominant component | Symbol = Group)",
                labels={x_col: x_axis_title, y_col: y_axis_title}
            )
            fig_scatter.update_traces(
                textposition='top center',
                marker=dict(size=14, line=dict(width=1, color='DarkSlateGrey')),
                hovertemplate="%{customdata[0]}<br><br><b>X:</b> %{x}<br><b>Y:</b> %{y}<extra></extra>"
            )

            if bar_aggregate and unique_groups > 1:
                # Plot moles (n) instead of concentration when mole columns are
                # available. Each selected '<chem>_mL/_mg/_g' has a '<chem>_mol'
                # counterpart produced by calculate_molar_ratios.
                def _mol_col(col):
                    for suf in ('_mL', '_mg', '_g'):
                        if col.endswith(suf):
                            return col[:-len(suf)] + '_mol'
                    return None

                mol_pairs = [(_mol_col(c), c) for c in multi_color_cols]
                bar_cols = [m for m, _ in mol_pairs if m and m in df_view.columns]

                if bar_cols:
                    y_label = 'Moles (mol)'
                    is_moles = True
                    # Map each '<chem>_mol' to the same color its '<chem>_mL'
                    # counterpart uses in the scatter, so both plots match.
                    bar_color_map = {m: color_map[c] for m, c in mol_pairs
                                     if m in bar_cols and c in color_map}
                else:  # fallback: no moles available (enable the molar calculator)
                    bar_cols = multi_color_cols
                    y_label = 'Mean concentration'
                    is_moles = False
                    bar_color_map = color_map

                df_bar = (df_view.groupby(effective_group)[bar_cols]
                                 .mean()
                                 .reset_index())

                if bar_display == "100% stacked":
                    # Normalize each group to its composition share so every
                    # component is visible regardless of absolute magnitude.
                    totals = df_bar[bar_cols].fillna(0).sum(axis=1)
                    df_bar[bar_cols] = (df_bar[bar_cols]
                                        .div(totals.where(totals != 0, np.nan),
                                             axis=0) * 100)
                    y_label = 'Composition (%)'
                elif is_moles:
                    # Pick a molar unit (mol, mmol, µmol, nmol, pmol, fmol) from the
                    # tallest stacked bar so the scale lives in the axis title
                    # instead of Plotly prefixing every tick (e.g. "600µ").
                    peak = df_bar[bar_cols].fillna(0).sum(axis=1).max()
                    units = [(1, 'mol'), (1e3, 'mmol'), (1e6, 'µmol'),
                             (1e9, 'nmol'), (1e12, 'pmol'), (1e15, 'fmol')]
                    factor, unit = units[0]
                    if peak and np.isfinite(peak) and peak > 0:
                        for f, u in units:
                            factor, unit = f, u
                            if peak * f >= 1:
                                break
                    df_bar[bar_cols] = df_bar[bar_cols] * factor
                    y_label = f'Moles ({unit})'

                fig_bars = px.bar(
                    df_bar,
                    x=effective_group,
                    y=bar_cols,
                    color_discrete_map=bar_color_map,
                    title=f"Mean Composition by {effective_group}",
                    labels={'value': y_label, 'variable': 'Component'}
                )
                if bar_display == "Grouped (log scale)":
                    # Each component gets its own bar on a log axis, so tiny
                    # components stay visible next to the dominant ones.
                    fig_bars.update_layout(barmode='group')
                    fig_bars.update_yaxes(type='log')
            else:
                df_sorted = df_view.sort_values(by=[effective_group, id_col]).copy()
                df_sorted['Bar_ID'] = (df_sorted[effective_group].astype(str) + " | "
                                       + df_sorted[id_col].astype(str))
                # Cap to a manageable count for readability
                if len(df_sorted) > 80:
                    st.info(f"⚠️ Showing first 80 of {len(df_sorted)} samples in the bar chart. "
                            f"Enable 'Aggregate bar chart by group' for the full view.")
                    df_sorted = df_sorted.head(80)
                fig_bars = px.bar(
                    df_sorted,
                    x='Bar_ID',
                    y=multi_color_cols,
                    color_discrete_map=color_map,
                    title="Composition Breakdown",
                    labels={'value': 'Concentration', 'variable': 'Component'}
                )
                fig_bars.update_layout(xaxis_title=f"{effective_group} | Sample")

            grouped = (bar_aggregate and unique_groups > 1
                       and bar_display == "Grouped (log scale)")
            if not grouped:
                fig_bars.update_layout(barmode='stack')
            if bar_aggregate and unique_groups > 1 and is_moles and not grouped:
                # Keep the unit prefix out of the tick labels (no "600µ"): the
                # scale is already expressed in the y-axis title.
                fig_bars.update_yaxes(exponentformat='none', ticksuffix='')

            col1, col2 = st.columns((3, 2))
            with col1:
                st.plotly_chart(fig_scatter, use_container_width=True)
            with col2:
                st.plotly_chart(fig_bars, use_container_width=True)

        # --- Tabla de Datos Final ---
        with st.expander("View Unified Multi-Screening Data Table"):
            # Drop hover/dominant helper columns for the user-facing table
            user_view = df_merged.copy()
            st.dataframe(user_view)

    except Exception as e:
        st.error(f"An error occurred while processing: {e}")
        with st.expander("Show traceback"):
            st.code(traceback.format_exc())
        st.write("Hint: Asegúrate de que los archivos compartan la columna ID seleccionada y que las columnas numéricas sean válidas.")

else:
    st.info("Please upload your Excel files in the sidebar to begin. You can select multiple files at once.")
