# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 20:42:42 2025

@author: AMokdad
"""

import streamlit as st
import pandas as pd
import math
import os
import pickle  # Pour sauvegarder/charger l'état
from io import BytesIO

import plotly.express as px  # Pour la visualisation (optionnel)
import plotly.graph_objects as go

# -----------------------------
# 0) Configurer la page
# -----------------------------
st.set_page_config(
    page_title="PPA-valuator",
    page_icon="🦖",
    layout="wide"
)

# -----------------------------
# 1) CSS personnalisé (thème d'origine)
# -----------------------------
st.markdown(
    """
    <style>
    .page-title {
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 0.2rem;
    }
    .yellow-bar {
        width: 60px;
        height: 4px;
        background-color: #FFCC00;
        margin-bottom: 1rem;
    }
    [data-testid="stButton"] > button {
        color: white !important;
        background-color: #333333 !important;
        border: 1px solid #333333 !important;
    }
    [data-testid="stButton"] > button:hover {
        background-color: #FFA500 !important;
        border: 1px solid #FFA500 !important;
        color: white !important;
    }
    [data-testid="stButton"] > button:focus,
    [data-testid="stButton"] > button:active {
        background-color: #FF8C00 !important;
        border: 1px solid #FF8C00 !important;
        color: white !important;
        outline: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    [data-testid="stSidebar"] [data-testid="stSelectbox"] > div > button {
        color: white !important;
        border: 1px solid #333333 !important;
    }
    [data-testid="stSidebar"] [data-testid="stSelectbox"] > div > button:hover {
        border: 1px solid #ffffff !important;
        color: white !important;
    }
    [data-testid="stSidebar"] [data-testid="stSelectbox"] > div > button:focus,
    [data-testid="stSidebar"] [data-testid="stSelectbox"] > div > button:active {
        outline: none !important;
        border: 1px solid #ffffff !important;
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Fonction de formatage des nombres
# -----------------------------
def format_number(val, decimals=0):
    """
    Formate un nombre en insérant un espace comme séparateur des milliers.
    Pour les nombres décimaux, le point est remplacé par une virgule.
    """
    if decimals:
        formatted = f"{val:,.{decimals}f}"
        return formatted.replace(",", " ").replace(".", ",")
    else:
        return f"{val:,.0f}".replace(",", " ")

# -----------------------------
# Fonction d'affichage du titre avec barre jaune
# -----------------------------
def page_title_and_bar(title_text):
    """Affiche un titre + la barre jaune sous forme HTML."""
    st.markdown(f"<div class='page-title'>{title_text}</div>", unsafe_allow_html=True)
    st.markdown("<div class='yellow-bar'></div>", unsafe_allow_html=True)

# -----------------------------
# (A) Fonction style DataFrame avec formatage des nombres
# -----------------------------
def style_dataframe(df, use_container_width=True):
    """
    Affiche le DataFrame sans l'index en réinitialisant l'index et en formatant
    les colonnes numériques pour insérer un espace comme séparateur des milliers.
    """
    df_reset = df.reset_index(drop=True)
    format_dict = {
        col: (
            lambda x: format_number(x, 0)
            if pd.notnull(x) and float(x).is_integer()
            else format_number(x, 2)
        )
        for col in df_reset.select_dtypes(include=["number"]).columns
    }
    return st.dataframe(df_reset.style.format(format_dict), use_container_width=use_container_width)

# -----------------------------
# 2) Variables de session
# -----------------------------
# Pour le Goodwill
if "adjustments" not in st.session_state:
    st.session_state["adjustments"] = []
if "tax_rate" not in st.session_state:
    st.session_state["tax_rate"] = 0.3
if "initial_anr" not in st.session_state:
    st.session_state["initial_anr"] = 0
if "ownership_share" not in st.session_state:
    st.session_state["ownership_share"] = 1.0
if "bilan_data" not in st.session_state:
    st.session_state["bilan_data"] = None

if "acquisition_price" not in st.session_state:
    st.session_state["acquisition_price"] = 0.0
if "additional_price" not in st.session_state:
    st.session_state["additional_price"] = 0.0

if "parameters_locked" not in st.session_state:
    st.session_state["parameters_locked"] = False

if "bilan_df" not in st.session_state:
    st.session_state["bilan_df"] = None

if "dummy_key" not in st.session_state:
    st.session_state["dummy_key"] = 0

# Pour le module CSM
if "csm_df" not in st.session_state:
    st.session_state["csm_df"] = None

# Pour stocker passifs techniques, part réassureur, etc.
if "passif_technique" not in st.session_state:
    st.session_state["passif_technique"] = 0.0
if "part_reassureurs" not in st.session_state:
    st.session_state["part_reassureurs"] = 0.0
if "be_net" not in st.session_state:
    st.session_state["be_net"] = 0.0
if "ra_net" not in st.session_state:
    st.session_state["ra_net"] = 0.0
if "csm_net" not in st.session_state:
    st.session_state["csm_net"] = 0.0

# -----------------------------
# (B) Fonctions Sauvegarde/Chargement (pickle)
# -----------------------------
def save_session_state(filepath):
    """Sauvegarde toutes les variables du session_state dans un fichier pickle."""
    try:
        with open(filepath, "wb") as f:
            pickle.dump(dict(st.session_state), f)
        return True, None
    except Exception as e:
        return False, str(e)

def load_session_state(filepath):
    """Charge un dictionnaire pickle et met à jour st.session_state."""
    try:
        with open(filepath, "rb") as f:
            loaded_data = pickle.load(f)
        for k, v in loaded_data.items():
            st.session_state[k] = v
        return True, None
    except Exception as e:
        return False, str(e)

# -----------------------------
# 3) Menu latéral
# -----------------------------
st.sidebar.title("Menu")
page = st.sidebar.selectbox(
    "Choisissez une page",
    [
        "Accueil",
        "Module Goodwill",
        "Module CSM",
        "Sauvegarder"
    ]
)

# ------------------------------------------------------------------
# Page "Accueil"
# ------------------------------------------------------------------
if page == "Accueil":
    page_title_and_bar("Accueil")
    st.markdown("""
    Bienvenue dans l'application de calcul **PPA-valuator🦖**.  

    Utilisez le menu sur la gauche pour naviguer entre les différentes pages.
    """)

# ------------------------------------------------------------------
# Page "Module Goodwill"
# ------------------------------------------------------------------
elif page == "Module Goodwill":
    page_title_and_bar("Module Goodwill")
    st.markdown("Choisissez l'un des onglets ci-dessous pour naviguer dans le module.")

    tab_import, tab_params, tab_ppa, tab_calcul, tab_export = st.tabs([
        "Import du bilan initial",
        "Paramètres",
        "Ajustements PPA",
        "Calcul du goodwill",
        "Export"
    ])

    # ========== Onglet 1 : Import du bilan initial ==========
    with tab_import:
        st.subheader("Import du bilan initial")

        uploaded_file = st.file_uploader("Importer un fichier Excel", type=["xlsx"])
        if uploaded_file:
            try:
                df = pd.read_excel(uploaded_file)
                df.columns = [col.strip() for col in df.columns]

                required_columns = ["Actif/Passif", "Agrégat", "Montant en €"]
                if not all(col in df.columns for col in required_columns):
                    st.error(f"Le fichier doit contenir : {', '.join(required_columns)}.")
                else:
                    # Convertir la colonne Montant en numérique
                    df["Montant en €"] = pd.to_numeric(df["Montant en €"], errors="coerce").fillna(0).abs()

                    # On sépare Actif / Passif
                    actifs = df[df["Actif/Passif"].str.lower() == "actif"][["Agrégat", "Montant en €"]]
                    passifs = df[df["Actif/Passif"].str.lower() == "passif"][["Agrégat", "Montant en €"]]

                    total_actifs = actifs["Montant en €"].sum()
                    total_passifs = passifs["Montant en €"].sum()

                    # Construction d'un "Bilan" simplifié
                    max_rows = max(len(actifs), len(passifs))
                    tableau = []
                    for i in range(max_rows):
                        agg_a = actifs.iloc[i]["Agrégat"] if i < len(actifs) else ""
                        mnt_a = ""
                        if i < len(actifs):
                            mnt_a = format_number(actifs.iloc[i]['Montant en €'], 0)
                        agg_p = passifs.iloc[i]["Agrégat"] if i < len(passifs) else ""
                        mnt_p = ""
                        if i < len(passifs):
                            mnt_p = format_number(passifs.iloc[i]['Montant en €'], 0)
                        tableau.append([agg_a, mnt_a, agg_p, mnt_p])

                    # Ligne total
                    tableau.append([
                        "Total Actifs",
                        format_number(total_actifs, 0),
                        "Total Passifs",
                        format_number(total_passifs, 0)
                    ])

                    bilan_df = pd.DataFrame(
                        tableau,
                        columns=["Actifs", "Montants actifs (€)", "Passifs", "Montants passifs (€)"]
                    )
                    st.session_state["bilan_df"] = bilan_df

                    # Extraction "Passifs techniques"
                    passif_tech = df[
                        (df["Agrégat"].str.lower() == "passifs techniques") &
                        (df["Actif/Passif"].str.lower() == "passif")
                    ]["Montant en €"].sum()
                    st.session_state["passif_technique"] = passif_tech

                    # Extraction "Part des réassureurs"
                    part_reass = df[
                        (df["Agrégat"].str.lower() == "part des reassureurs") &
                        (df["Actif/Passif"].str.lower() == "actif")
                    ]["Montant en €"].sum()
                    st.session_state["part_reassureurs"] = part_reass

                    # Contrôle équilibre
                    if math.isclose(total_actifs, total_passifs, abs_tol=1e-6):
                        st.success("Le bilan est équilibré : Actif = Passif.")
                    else:
                        st.error(
                            f"Bilan non équilibré : Actif ({format_number(total_actifs,0)} €) "
                            f"≠ Passif ({format_number(total_passifs,0)} €)."
                        )

                    # Recherche du capital => ANR initial
                    capital_row = df[
                        (df["Agrégat"].str.lower() == "capitaux propres") &
                        (df["Actif/Passif"].str.lower() == "passif")
                    ]
                    if not capital_row.empty:
                        initial_anr = capital_row["Montant en €"].sum()
                        st.session_state["initial_anr"] = initial_anr
                        st.info("ANR initial calculé : " + format_number(initial_anr, 0) + " €")
                    else:
                        st.warning("Aucune ligne 'Capitaux propres' trouvée dans les passifs.")

                    # Contrôle passifs techniques
                    if passif_tech == 0.0:
                        st.warning(
                            "Impossible de détecter un montant non-nul pour 'Passifs techniques'. "
                            "Vérifiez le libellé dans Excel."
                        )
                    else:
                        st.info("Passifs techniques = " + format_number(passif_tech, 2) + " €")

                    # Contrôle part des réassureurs
                    if part_reass == 0.0:
                        st.warning(
                            "Impossible de détecter un montant non-nul pour 'Part des reassureurs'. "
                            "Vérifiez le libellé dans Excel."
                        )
                    else:
                        st.info("Part des reassureurs = " + format_number(part_reass, 2) + " €")

            except Exception as e:
                st.error("Erreur lors de la lecture : " + str(e))

        if st.session_state["bilan_df"] is not None:
            st.subheader("Bilan importé :")
            style_dataframe(st.session_state["bilan_df"])

    # ========== Onglet 2 : Paramètres ==========
    with tab_params:
        st.subheader("Paramètres")
        st.markdown("Définissez les paramètres :")

        if st.session_state["parameters_locked"]:
            st.write("Taux d'imposition : " + format_number(st.session_state['tax_rate']*100, 1) + " %")
            st.write("Quote-part de détention : " + format_number(st.session_state['ownership_share']*100, 1) + " %")
            st.write("Prix d'acquisition : " + format_number(st.session_state['acquisition_price'], 2) + " €")
            st.write("Complément de prix : " + format_number(st.session_state['additional_price'], 2) + " €")

            if st.button("Déverrouiller"):
                st.session_state["parameters_locked"] = False
        else:
            tax_rate = st.number_input(
                "Taux d'imposition (%)",
                min_value=0.0, max_value=100.0,
                value=st.session_state["tax_rate"]*100, step=0.1
            )
            st.session_state["tax_rate"] = tax_rate / 100

            ownership_share = st.number_input(
                "Quote-part de détention (%)",
                min_value=0.0, max_value=100.0,
                value=st.session_state["ownership_share"]*100, step=0.1
            )
            st.session_state["ownership_share"] = ownership_share / 100

            acquisition_price = st.number_input(
                "Prix d'acquisition (€)",
                min_value=0.0,
                value=st.session_state["acquisition_price"],
                step=1000.0
            )
            st.session_state["acquisition_price"] = acquisition_price

            additional_price = st.number_input(
                "Complément de prix (€)",
                min_value=0.0,
                value=st.session_state["additional_price"],
                step=1000.0
            )
            st.session_state["additional_price"] = additional_price

            if st.button("Verrouiller"):
                st.session_state["parameters_locked"] = True

    # ========== Onglet 3 : Ajustements PPA ==========
    with tab_ppa:
        st.subheader("Ajustements de PPA")
        st.markdown("Ajoutez des ajustements de PPA...")

        # Saisie manuelle
        st.subheader("Saisie manuelle")
        with st.form("add_adjustment_form"):
            label = st.text_input("Intitulé")
            value = st.number_input("Montant (€)", step=1000.0)
            submitted = st.form_submit_button("Ajouter")

            if submitted:
                if label and value != 0:
                    already_exists = any(
                        (adj["label"] == label.strip() and adj["value"] == round(value))
                        for adj in st.session_state["adjustments"]
                    )
                    if already_exists:
                        st.warning("Cet ajustement existe déjà (même intitulé et même montant).")
                    else:
                        st.session_state["adjustments"].append({
                            "label": label.strip(),
                            "value": round(value)
                        })
                        st.success("Ajustement '" + label + "' ajouté : " + format_number(value, 0) + " €")
                else:
                    st.error("Veuillez renseigner un intitulé et un montant valide.")

        # Import Excel
        st.subheader("Importer un fichier Excel")
        uploaded_ppa = st.file_uploader("Fichier .xlsx", type=["xlsx"])
        if uploaded_ppa:
            try:
                df_ppa = pd.read_excel(uploaded_ppa)
                df_ppa.columns = [c.strip().lower() for c in df_ppa.columns]

                possible_label_cols = ["intitulé", "label"]
                possible_value_cols = ["montant", "value"]

                label_col, value_col = None, None
                for c in possible_label_cols:
                    if c in df_ppa.columns:
                        label_col = c
                        break
                for c in possible_value_cols:
                    if c in df_ppa.columns:
                        value_col = c
                        break

                if label_col and value_col:
                    df_ppa[value_col] = pd.to_numeric(df_ppa[value_col], errors="coerce").fillna(0).round()
                    new_adjustments = []
                    for _, row in df_ppa.iterrows():
                        lbl = str(row[label_col]).strip()
                        val = int(row[value_col])
                        if lbl and val != 0:
                            already_exists = any(
                                (adj["label"] == lbl and adj["value"] == val)
                                for adj in st.session_state["adjustments"]
                            )
                            if not already_exists:
                                new_adjustments.append({"label": lbl, "value": val})
                    st.session_state["adjustments"].extend(new_adjustments)
                    st.success(str(len(new_adjustments)) + " ajustement(s) importé(s).")
                else:
                    st.error("Colonnes 'intitulé/label' et 'montant/value' introuvables dans ce fichier.")
            except Exception as e:
                st.error("Erreur lecture du fichier : " + str(e))

        # Ajustement automatique IFRS17
        st.subheader("Ajustement automatique IFRS 17")
        st.markdown("""
        Cet ajustement correspond à la différence entre l'évaluation des actifs et passifs d'assurance en norme française et selon les principes de la norme IFRS 17 :  
        **Passifs techniques** - **Part des réassureurs** - **BE IFRS 17 net** - **RA IFRS 17 net** - **CSM net**.
        """)

        if st.button("Calculer l'ajustement IFRS 17"):
            passif_tech = st.session_state.get("passif_technique", 0.0)
            part_reass = st.session_state.get("part_reassureurs", 0.0)
            be_net = st.session_state.get("be_net", 0.0)
            ra_net = st.session_state.get("ra_net", 0.0)
            csm_net = st.session_state.get("csm_net", 0.0)

            val_auto = passif_tech - part_reass - be_net - ra_net - csm_net
            label_auto = "Ajustement IFRS 17"

            already_exists = any(
                (adj["label"] == label_auto and adj["value"] == round(val_auto))
                for adj in st.session_state["adjustments"]
            )
            if already_exists:
                st.warning("L'ajustement automatique (même montant) existe déjà.")
            else:
                st.session_state["adjustments"].append({
                    "label": label_auto,
                    "value": round(val_auto)
                })
                st.success("Ajustement automatique : " + label_auto + " = " + format_number(val_auto, 0) + " € ajouté.")

        # Liste ajustements
        st.subheader("Ajustements enregistrés")
        if st.session_state["adjustments"]:
            for i, adj in enumerate(st.session_state["adjustments"]):
                c1, c2, c3 = st.columns([0.5, 0.3, 0.2])
                with c1:
                    st.write(adj['label'])
                with c2:
                    st.write(format_number(adj['value'], 2) + " €")
                with c3:
                    if st.button("Supprimer", key=f"delete_{i}"):
                        st.session_state["adjustments"].pop(i)
                        st.session_state["dummy_key"] += 1
                        # Remplacement de st.experimental_rerun() par st.stop()
                        st.stop()
        else:
            st.info("Aucun ajustement pour le moment.")

    # ========== Onglet 4 : Calcul du Goodwill ==========
    with tab_calcul:
        st.subheader("Calcul du Goodwill")

        initial_anr = st.session_state.get("initial_anr", 0.0)
        adjustments = st.session_state.get("adjustments", [])
        tax_rate = st.session_state.get("tax_rate", 0.3)
        ownership_share = st.session_state.get("ownership_share", 1.0)
        acquisition_price = st.session_state.get("acquisition_price", 0.0)
        additional_price = st.session_state.get("additional_price", 0.0)

        total_adjustments = sum(item["value"] for item in adjustments)
        deferred_taxes = -(tax_rate * total_adjustments)
        final_anr = initial_anr + total_adjustments + deferred_taxes
        anr_at_share = final_anr * ownership_share
        total_investment = acquisition_price + additional_price
        diff = anr_at_share - total_investment

        if diff < 0:
            st.success("Goodwill : " + format_number(abs(diff), 2) + " €")
        elif diff > 0:
            st.warning("Badwill : " + format_number(diff, 2) + " €")
        else:
            st.info("Équilibre exact (Ni Goodwill ni Badwill).")

        st.write("---")
        st.write("### Détail du calcul :")
        st.write("- **ANR initial** : " + format_number(initial_anr, 2) + " €")
        st.write("- **Somme des ajustements PPA** : " + format_number(total_adjustments, 2) + " €")
        st.write("- **Impôts différés** (taux=" + format_number(tax_rate*100, 1) + " %) : " + format_number(deferred_taxes, 2) + " €")
        st.write("- **ANR final** = ANR initial + Ajustements + Impôts différés = " + format_number(final_anr, 2) + " €")
        st.write("- **Quote-part** (" + format_number(ownership_share*100, 1) + " %) => " + format_number(anr_at_share, 2) + " €")
        st.write("- **Prix d'acquisition** : " + format_number(acquisition_price, 2) + " €")
        st.write("- **Complément de prix** : " + format_number(additional_price, 2) + " €")
        st.write("- **Contrepartie payée** : " + format_number(total_investment, 2) + " €")
        st.write("- **Différence** = " + format_number(diff, 2) + " €")

        if adjustments:
            st.write("#### Ajustements de PPA inclus :")
            for adj in adjustments:
                st.write("- " + adj['label'] + " : " + format_number(adj['value'], 2) + " €")
        else:
            st.write("Aucun ajustement saisi.")

    # ========== Onglet 5 : Export (Excel) ==========
    with tab_export:
        st.subheader("Export complet en Excel")

        initial_anr = st.session_state.get("initial_anr", 0.0)
        adjustments = st.session_state.get("adjustments", [])
        tax_rate = st.session_state.get("tax_rate", 0.3)
        ownership_share = st.session_state.get("ownership_share", 1.0)
        acquisition_price = st.session_state.get("acquisition_price", 0.0)
        additional_price = st.session_state.get("additional_price", 0.0)

        total_adjustments = sum(item["value"] for item in adjustments)
        deferred_taxes = -(tax_rate * total_adjustments)
        final_anr = initial_anr + total_adjustments + deferred_taxes
        anr_at_share = final_anr * ownership_share
        total_investment = acquisition_price + additional_price
        diff = anr_at_share - total_investment

        if diff < 0:
            type_gwd = "Goodwill"
            montant_gwd = abs(diff)
        elif diff > 0:
            type_gwd = "Badwill"
            montant_gwd = diff
        else:
            type_gwd = "Équilibre"
            montant_gwd = 0.0

        st.write("Cliquez ci-dessous pour générer l'Excel (Synthèse, Ajustements_PPA, Paramètres).")

        summary_data = [
            ("ANR initial", initial_anr),
            ("Ajustements de PPA (total)", sum(item["value"] for item in adjustments)),
            ("Impôts différés", deferred_taxes),
            ("ANR final", final_anr),
            ("Quote-part (%)", ownership_share * 100),
            ("ANR à la quote-part", anr_at_share),
            ("Prix d'acquisition", acquisition_price),
            ("Complément de prix", additional_price),
            ("Contrepartie payée", acquisition_price + additional_price),
            ("Type d'écart", type_gwd),
            ("Montant d'écart", montant_gwd),
        ]
        df_synth = pd.DataFrame(summary_data, columns=["Description", "Valeur"])

        def convert_val(row):
            if row["Description"] == "Type d'écart":
                return str(row["Valeur"])
            else:
                return float(row["Valeur"])

        df_synth["Valeur"] = df_synth.apply(convert_val, axis=1)

        if adjustments:
            df_ppa = pd.DataFrame(adjustments)
            df_ppa["value"] = df_ppa["value"].astype(float)
        else:
            df_ppa = pd.DataFrame(columns=["label", "value"])

        param_data = [
            ("Taux d'imposition (%)", tax_rate * 100),
            ("Quote-part de détention (%)", ownership_share * 100),
            ("Prix d'acquisition (€)", acquisition_price),
            ("Complément de prix (€)", additional_price),
        ]
        df_params = pd.DataFrame(param_data, columns=["Paramètre", "Valeur"])
        df_params["Valeur"] = df_params["Valeur"].astype(float)

        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df_synth.to_excel(writer, index=False, sheet_name="Synthese")
            df_ppa.to_excel(writer, index=False, sheet_name="Ajustements_PPA")
            df_params.to_excel(writer, index=False, sheet_name="Paramètres")

            workbook = writer.book
            ws_synth = writer.sheets["Synthese"]
            ws_ppa = writer.sheets["Ajustements_PPA"]
            ws_params = writer.sheets["Paramètres"]

            ws_synth.hide_gridlines(2)
            ws_ppa.hide_gridlines(2)
            ws_params.hide_gridlines(2)

            header_format = workbook.add_format({
                'bold': True,
                'bg_color': '#4F81BD',
                'font_color': 'white',
                'align': 'center'
            })
            euro_format = workbook.add_format({'num_format': '# ##0,00'})
            percent_format = workbook.add_format({'num_format': '0,00%'})

            for col_num, col_name in enumerate(df_synth.columns):
                ws_synth.write(0, col_num, col_name, header_format)
            ws_synth.set_column(0, 0, 35)
            ws_synth.set_column(1, 1, 15)

            for col_num, col_name in enumerate(df_ppa.columns):
                ws_ppa.write(0, col_num, col_name, header_format)
            ws_ppa.set_column(0, 0, 35)
            ws_ppa.set_column(1, 1, 15)

            for col_num, col_name in enumerate(df_params.columns):
                ws_params.write(0, col_num, col_name, header_format)
            ws_params.set_column(0, 0, 40)
            ws_params.set_column(1, 1, 15)

            # Mise en forme des valeurs (Synthese)
            for row_num in range(len(df_synth)):
                descr = str(df_synth.iloc[row_num]["Description"]).lower()
                val_raw = df_synth.iloc[row_num]["Valeur"]
                if descr == "type d'écart":
                    ws_synth.write_string(row_num+1, 1, val_raw)
                else:
                    val_float = float(val_raw)
                    if "(%)" in descr:
                        ws_synth.write_number(row_num+1, 1, val_float/100.0, percent_format)
                    else:
                        ws_synth.write_number(row_num+1, 1, val_float, euro_format)

            # Mise en forme des valeurs (Ajustements_PPA)
            for row_num in range(len(df_ppa)):
                if not df_ppa.empty:
                    val_ppa = df_ppa.iloc[row_num]["value"]
                    ws_ppa.write_number(row_num+1, 1, val_ppa, euro_format)

            # Mise en forme des valeurs (Paramètres)
            for row_num in range(len(df_params)):
                param_text = str(df_params.iloc[row_num]["Paramètre"]).lower()
                val_param = df_params.iloc[row_num]["Valeur"]
                if "(%)" in param_text:
                    ws_params.write_number(row_num+1, 1, val_param/100.0, percent_format)
                else:
                    ws_params.write_number(row_num+1, 1, val_param, euro_format)

        st.download_button(
            label="Télécharger l'Excel formaté",
            data=output.getvalue(),
            file_name="goodwill_synthese.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

# ------------------------------------------------------------------
# Page "Module CSM"
# ------------------------------------------------------------------
elif page == "Module CSM":
    page_title_and_bar("Module CSM")
    st.markdown("Module dédié au calcul de la CSM.")

    tab_import_csm, tab_calcul_csm, tab_controle_csm, tab_resultats_csm, tab_export_csm = st.tabs([
        "Import portefeuille",
        "Calcul de la CSM",
        "Contrôle",
        "Résultats",
        "Export"
    ])

    # ---- 1) "Import portefeuille" ----
    with tab_import_csm:
        st.subheader("Import d'un portefeuille IFRS17")
        st.markdown("""
        Format attendu :  
        - Portefeuille IFRS 17 (texte)
        - Juste valeur (nombre)
        - BE IFRS 17 (nombre)
        - RA IFRS 17 (nombre)
        - Type portefeuille (brut ou cédé)
        """)

        csm_file = st.file_uploader("Charger le portefeuille IFRS 17", type=["xlsx"])
        if csm_file:
            try:
                df_csm = pd.read_excel(csm_file)
                df_csm.columns = [c.strip() for c in df_csm.columns]

                required_cols = ["Portefeuille IFRS 17", "Juste valeur", "BE IFRS 17", "RA IFRS 17", "Type portefeuille"]
                if not all(rc in df_csm.columns for rc in required_cols):
                    st.error(f"Le fichier doit contenir : {', '.join(required_cols)}.")
                else:
                    for col in ["Juste valeur", "BE IFRS 17", "RA IFRS 17"]:
                        df_csm[col] = pd.to_numeric(df_csm[col], errors="coerce").fillna(0.0)

                    st.session_state["csm_df"] = df_csm
                    st.success("Portefeuille IFRS17 importé avec succès.")
            except Exception as e:
                st.error("Erreur : " + str(e))

        if st.session_state["csm_df"] is not None:
            style_dataframe(st.session_state["csm_df"], use_container_width=True)

    # ---- 2) "Calcul de la CSM" ----
    with tab_calcul_csm:
        st.subheader("Calcul de la CSM")

        if st.session_state["csm_df"] is None:
            st.warning("Veuillez d'abord importer un portefeuille dans l'onglet précédent.")
        else:
            df_calc = st.session_state["csm_df"].copy()

            df_calc["csm_value"] = df_calc["Juste valeur"] - df_calc["BE IFRS 17"] - df_calc["RA IFRS 17"]

            def csm_logic(row):
                csm_val = row["csm_value"]
                type_op = str(row["Type portefeuille"]).strip().lower()
                if type_op == "brut":
                    if csm_val > 0:
                        return csm_val, None
                    else:
                        return None, csm_val
                elif type_op in ["cédé", "cede"]:
                    return csm_val, None
                else:
                    if csm_val > 0:
                        return csm_val, None
                    else:
                        return None, csm_val

            df_calc["CSM"], df_calc["LC"] = zip(*df_calc.apply(csm_logic, axis=1))

            st.write("**Table après calcul (avec CSM, LC)**")
            style_dataframe(
                df_calc[[
                    "Portefeuille IFRS 17", "Juste valeur", "BE IFRS 17",
                    "RA IFRS 17", "Type portefeuille", "CSM", "LC"
                ]]
            )

            st.session_state["csm_df"] = df_calc

    # ---- 3) "Contrôle" : JV = BE + RA + CSM - LC ?
    with tab_controle_csm:
        st.subheader("Contrôle : Juste valeur - BE IFRS 17 - RA IFRS 17 - CSM - Composante de perte = 0")

        if st.session_state["csm_df"] is None:
            st.warning("Veuillez importer et calculer la CSM d'abord.")
        else:
            df_check = st.session_state["csm_df"].copy()
            df_check["diff"] = df_check["Juste valeur"] - (
                df_check["BE IFRS 17"] + df_check["RA IFRS 17"] + df_check["CSM"].fillna(0) + df_check["LC"].fillna(0)
            )
            epsilon = 1e-6
            df_check["check"] = df_check["diff"].apply(lambda x: "OK" if abs(x) < epsilon else "KO")

            style_dataframe(
                df_check[[
                    "Portefeuille IFRS 17", "Type portefeuille",
                    "Juste valeur", "BE IFRS 17", "RA IFRS 17", "CSM", "LC",
                    "diff", "check"
                ]]
            )

            nb_total = len(df_check)
            nb_ok = (df_check["check"] == "OK").sum()
            nb_ko = nb_total - nb_ok
            if nb_ko == 0:
                st.success("Contrôle validé pour les " + str(nb_total) + " lignes (toutes 'OK').")
            else:
                st.error("Contrôle non validé pour " + str(nb_ko) + " ligne(s) / " + str(nb_total) + ".")

    # ---- 4) "Résultats"
    with tab_resultats_csm:
        st.subheader("Résultats de la CSM")

        if st.session_state["csm_df"] is None:
            st.warning("Veuillez importer un portefeuille et effectuer le calcul de la CSM.")
        else:
            df_res = st.session_state["csm_df"].copy()

            df_brut = df_res[df_res["Type portefeuille"].str.lower() == "brut"].copy()
            df_cede = df_res[df_res["Type portefeuille"].str.lower().isin(["cédé", "cede"])].copy()

            be_brut = df_brut["BE IFRS 17"].sum()
            ra_brut = df_brut["RA IFRS 17"].sum()
            csm_brut = df_brut["CSM"].sum()
            lc_brut = df_brut["LC"].sum() if "LC" in df_brut.columns else 0.0

            be_cede = df_cede["BE IFRS 17"].sum()
            ra_cede = df_cede["RA IFRS 17"].sum()
            csm_cede = df_cede["CSM"].sum()
            lc_cede = df_cede["LC"].sum() if "LC" in df_cede.columns else 0.0

            be_net = be_brut - be_cede
            ra_net = ra_brut - ra_cede
            csm_net = csm_brut - csm_cede
            lc_net = lc_brut - lc_cede

            st.session_state["be_net"] = be_net
            st.session_state["ra_net"] = ra_net
            st.session_state["csm_net"] = csm_net

            st.write("**BE Net** = " + format_number(be_net, 2))
            st.write("**RA Net** = " + format_number(ra_net, 2))
            st.write("**CSM Net** = " + format_number(csm_net, 2))
            st.write("**LC Net** = " + format_number(lc_net, 2))

            st.write("### Tableau détaillé avec sous-totaux et Net")
            cols = ["Type portefeuille", "Portefeuille IFRS 17", "BE IFRS 17", "RA IFRS 17", "CSM", "LC"]
            rows = []

            for idx, row in df_brut.iterrows():
                rows.append({
                    "Type portefeuille": row["Type portefeuille"],
                    "Portefeuille IFRS 17": row["Portefeuille IFRS 17"],
                    "BE IFRS 17": row["BE IFRS 17"],
                    "RA IFRS 17": row["RA IFRS 17"],
                    "CSM": row["CSM"],
                    "LC": row["LC"]
                })
            rows.append({
                "Type portefeuille": "SOUS-TOTAL BRUT",
                "Portefeuille IFRS 17": "",
                "BE IFRS 17": be_brut,
                "RA IFRS 17": ra_brut,
                "CSM": csm_brut,
                "LC": lc_brut
            })

            for idx, row in df_cede.iterrows():
                rows.append({
                    "Type portefeuille": row["Type portefeuille"],
                    "Portefeuille IFRS 17": row["Portefeuille IFRS 17"],
                    "BE IFRS 17": row["BE IFRS 17"],
                    "RA IFRS 17": row["RA IFRS 17"],
                    "CSM": row["CSM"],
                    "LC": row["LC"]
                })
            rows.append({
                "Type portefeuille": "SOUS-TOTAL CÉDÉ",
                "Portefeuille IFRS 17": "",
                "BE IFRS 17": be_cede,
                "RA IFRS 17": ra_cede,
                "CSM": csm_cede,
                "LC": lc_cede
            })

            rows.append({
                "Type portefeuille": "NET (BRUT - CÉDÉ)",
                "Portefeuille IFRS 17": "",
                "BE IFRS 17": be_net,
                "RA IFRS 17": ra_net,
                "CSM": csm_net,
                "LC": lc_net
            })

            df_final = pd.DataFrame(rows, columns=cols)
            style_dataframe(df_final)

    # ---- 5) "Export CSM"
    with tab_export_csm:
        st.subheader("Export")
        if st.session_state["csm_df"] is None:
            st.warning("Rien à exporter, veuillez importer et calculer la CSM d'abord.")
        else:
            df_export_csm = st.session_state["csm_df"].copy()
            output_csm = BytesIO()
            with pd.ExcelWriter(output_csm, engine='xlsxwriter') as writer:
                df_export_csm.to_excel(writer, index=False, sheet_name="CSM_Calcul")

                workbook = writer.book
                ws = writer.sheets["CSM_Calcul"]
                ws.hide_gridlines(2)

                header_fmt = workbook.add_format({
                    'bold': True,
                    'bg_color': '#4F81BD',
                    'font_color': 'white',
                    'align': 'center'
                })
                for col_num, col_name in enumerate(df_export_csm.columns):
                    ws.write(0, col_num, col_name, header_fmt)

                ws.set_column(0, 0, 30)
                ws.set_column(1, 1, 15)
                ws.set_column(2, 2, 15)
                ws.set_column(3, 3, 15)
                ws.set_column(4, 4, 18)
                ws.set_column(5, 5, 15)
                ws.set_column(6, 6, 15)

            st.download_button(
                label="Télécharger le tableau CSM",
                data=output_csm.getvalue(),
                file_name="csm_calcul.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

# ------------------------------------------------------------------
# Page "Sauvegarder"
# ------------------------------------------------------------------
elif page == "Sauvegarder":
    page_title_and_bar("Sauvegarder / Charger l'état")
    st.markdown("""
    Ici, vous pouvez sauvegarder votre avancement
    en téléchargeant un fichier de sauvegarde, puis en le rechargeant ultérieurement.
    """)

    # Bouton pour télécharger l'état de la session
    session_bytes = pickle.dumps(dict(st.session_state))
    st.download_button(
         label="Télécharger la sauvegarde de session",
         data=session_bytes,
         file_name="session_state.pkl",
         mime="application/octet-stream"
    )

    # Téléversement d'une sauvegarde
    uploaded_state = st.file_uploader("Charger une sauvegarde de session", type=["pkl"])
    if uploaded_state is not None:
         try:
              loaded_data = pickle.load(uploaded_state)
              for k, v in loaded_data.items():
                   st.session_state[k] = v
              st.success("Session chargée avec succès!")
         except Exception as e:
              st.error("Erreur lors du chargement de la session : " + str(e))
