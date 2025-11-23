# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Cycle Time Predictor – Aluminium Turning",
    layout="wide"
)

@st.cache_resource
def load_model():
    data = joblib.load("model.pkl")
    return data["model"], data["feature_names"]

model, feature_names = load_model()

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 0.95rem;
        color: #555;
        margin-bottom: 1rem;
    }
    .tagline {
        font-size: 0.85rem;
        color: #777;
        font-style: italic;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        padding: 1.2rem 1.5rem;
        border-radius: 0.75rem;
        border: 1px solid #eee;
        background-color: #fafafa;
    }
    .metric-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        color: #666;
        letter-spacing: 0.08em;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 0.3rem;
    }
    .metric-unit {
        font-size: 0.9rem;
        color: #777;
        margin-left: 0.25rem;
    }
    .chips {
        display: flex;
        flex-wrap: wrap;
        gap: 0.4rem;
        margin-top: 0.4rem;
    }
    .chip {
        font-size: 0.75rem;
        padding: 0.25rem 0.6rem;
        border-radius: 999px;
        border: 1px solid #ddd;
        background-color: #f7f7f7;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.sidebar.title("⚙️ Process Inputs")

st.sidebar.markdown(
    "Ajustează parametrii de strunjire pentru piese din **aliaj de aluminiu** "
    "și vezi cum se modifică timpul de ciclu estimat."
)

part_complexity = st.sidebar.slider(
    "Part complexity [1–5]",
    min_value=1,
    max_value=5,
    value=3,
    help=(
        "1 = piesă simplă, puține operații și toleranțe lejere\n"
        "5 = piesă complexă, multiple suprafețe funcționale și toleranțe strânse."
    )
)

cutting_speed = st.sidebar.slider(
    "Cutting speed vc [m/min]",
    min_value=150,
    max_value=600,
    value=300,
    step=10,
    help="Viteza de așchiere tipică pentru strunjirea aliajelor de aluminiu."
)

feed = st.sidebar.slider(
    "Feed f [mm/rev]",
    min_value=0.05,
    max_value=0.35,
    value=0.18,
    step=0.01,
    help="Avans per rotație în operația de strunjire longitudinală."
)

depth_of_cut = st.sidebar.slider(
    "Depth of cut ap [mm]",
    min_value=0.2,
    max_value=4.0,
    value=1.5,
    step=0.1,
    help="Adâncimea efectivă de așchiere."
)

st.sidebar.markdown("---")

part_diameter = st.sidebar.slider(
    "Part diameter [mm]",
    min_value=20,
    max_value=150,
    value=60,
    step=5
)

part_length = st.sidebar.slider(
    "Machined length [mm]",
    min_value=10,
    max_value=200,
    value=80,
    step=5
)

material_hardness = st.sidebar.slider(
    "Material hardness [HRC]",
    min_value=20,
    max_value=32,
    value=24,
    step=1,
    help="Duritatea echivalentă a aliajului de aluminiu."
)

tool_wear = st.sidebar.slider(
    "Tool wear VB [mm]",
    min_value=0.05,
    max_value=0.30,
    value=0.10,
    step=0.01,
    help="Lățimea craterului de uzură VB conform ISO 3685."
)

coolant_flow = st.sidebar.slider(
    "Coolant flow [L/min]",
    min_value=2,
    max_value=12,
    value=6,
    step=1,
    help="Debit de lichid de răcire în zona de așchiere."
)

predict_button = st.sidebar.button("🔮 Predict cycle time")


st.markdown('<div class="main-title">Cycle Time Predictor – Aluminium Turning</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">'
    'ML demonstrator for estimating cycle time in CNC turning of aluminium parts, '
    'combining analytical machining relations with data-driven modelling.'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="tagline">'
    'Hybrid physical–statistical approach: classical turning time equations used to generate '
    'synthetic data, then approximated via Random Forest regression.'
    '</div>',
    unsafe_allow_html=True
)

tabs = st.tabs(["📈 Prediction", "📊 Model & Methodology"])

# Helper: theoretical time calculation (simplified)
def compute_theoretical_time(
    cutting_speed_m_min: float,
    feed_mm_rev: float,
    depth_of_cut_mm: float,
    part_diameter_mm: float,
    part_length_mm: float,
) -> float:
    """
    T_mach [sec] ≈ 60 * (L * pi * d) / (1000 * vc * f)
    (strunjire longitudinală simplificată)
    """
    pi = np.pi
    vc = max(cutting_speed_m_min, 1e-3)
    f = max(feed_mm_rev, 1e-3)
    L = part_length_mm
    d = part_diameter_mm

    T_mach_sec = 60.0 * (L * pi * d) / (1000.0 * vc * f)
    return T_mach_sec


with tabs[0]:
    col_left, col_right = st.columns([1.1, 1.2])

    with col_left:
        st.markdown("### 🔍 Prediction summary")

        if predict_button:
            # vectorul de intrare, cu aceleași denumiri ca la antrenare
            input_df = pd.DataFrame([{
                "part_complexity": part_complexity,
                "cutting_speed_m_min": cutting_speed,
                "feed_mm_rev": feed,
                "depth_of_cut_mm": depth_of_cut,
                "part_diameter_mm": part_diameter,
                "part_length_mm": part_length,
                "material_hardness_HRC": material_hardness,
                "tool_wear_VB_mm": tool_wear,
                "coolant_flow_l_min": coolant_flow,
            }])

            pred = float(model.predict(input_df)[0])

            # timp teoretic simplificat (doar așchiere)
            T_theoretical = compute_theoretical_time(
                cutting_speed_m_min=cutting_speed,
                feed_mm_rev=feed,
                depth_of_cut_mm=depth_of_cut,
                part_diameter_mm=part_diameter,
                part_length_mm=part_length,
            )

            # small card-style metric display
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Predicted cycle time</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="metric-value">{pred:.1f}'
                '<span class="metric-unit">sec</span></div>',
                unsafe_allow_html=True
            )
            st.markdown(
                f'<div class="chips">'
                f'<div class="chip">Theoretical machining time ≈ {T_theoretical:.1f} s</div>'
                f'<div class="chip">Δ (ML − theory) ≈ {pred - T_theoretical:+.1f} s</div>'
                f'</div>',
                unsafe_allow_html=True
            )
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("#### Interpretation")
            st.write(
                "- Modelul ML include nu doar timpul teoretic de așchiere, ci și efecte "
                "asociate **complexității piesei**, **durității materialului**, "
                "**uzurii sculei** și **condițiilor de răcire**.\n"
                "- Diferența față de timpul teoretic simplificat reflectă acești timpi "
                "auxiliari și deviații de proces."
            )

        else:
            st.info("Introdu parametrii în sidebar și apasă **“🔮 Predict cycle time”** pentru a vedea rezultatele.")

    with col_right:
        st.markdown("### 📊 Feature importance")
        # importanțele globale din Random Forest
        importances = model.feature_importances_
        fi_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importances
        }).sort_values("importance", ascending=False)

        st.bar_chart(fi_df.set_index("feature"))

        with st.expander("Descriere scurtă a parametrilor"):
            st.markdown(
                """
                - **cutting_speed_m_min** – viteza de așchiere, controlată din CNC (m/min)  
                - **feed_mm_rev** – avans pe rotație (mm/rev)  
                - **depth_of_cut_mm** – adâncimea de așchiere (mm)  
                - **part_diameter_mm / part_length_mm** – geometria zonei prelucrate  
                - **material_hardness_HRC** – duritatea echivalentă a aliajului de aluminiu  
                - **tool_wear_VB_mm** – uzura sculei (wide of wear land VB)  
                - **coolant_flow_l_min** – debitul de răcire în zona de așchiere  
                - **part_complexity** – scor calitativ 1–5 care agregă numărul de operații, setup-uri și toleranțe strânse.
                """
            )

with tabs[1]:
    st.markdown("### 🧠 Model & Methodology")

    st.markdown(
        """
        Acest demonstrator implementează un **model hibrid fizico–statistic** pentru
        estimarea timpilor de ciclu la strunjirea aliajelor de aluminiu:

        1. **Generare de date sintetice**  
           - Se pornește de la ecuația clasică a timpului de așchiere la strunjire
             pentru o operație longitudinală:  
             \n
             \\[
             T_{mach} = 60 \\cdot \\frac{L \\cdot \\pi d}{1000 \\cdot v_c \\cdot f}
             \\]
             \n
             unde \\(L\\) este lungimea prelucrată, \\(d\\) diametrul, \\(v_c\\) viteza de așchiere,
             iar \\(f\\) avansul.

        2. **Modelarea timpilor auxiliari și a efectelor de proces**  
           - Se adaugă termeni pentru **complexitatea piesei**, **duritatea materialului**,
             **uzura sculei** și **condițiile de răcire**, precum și zgomot statistic,
             pentru a aproxima variațiile reale de producție.

        3. **Învățare automată**  
           - Pe setul de date astfel generat este antrenat un model de regresie
             **Random Forest**, care aproximează relația intrare–ieșire și permite
             obținerea unei estimări robuste a timpului de ciclu, inclusiv pentru
             combinații noi de parametri.

        4. **Interpretabilitate**  
           - Importanța globală a caracteristicilor (feature importance) este utilizată
             pentru a evidenția parametrii cu impact major asupra timpului de ciclu,
             oferind suport decizional inginerilor de proces.
        """
    )

    st.markdown(
        """
        În context academic, acest tip de abordare poate fi integrat într-o
        **metodologie de optimizare a proceselor de prelucrare** bazată pe:
        - modele analitice clasice (deterministe),
        - augmentare cu date și zgomot controlat,
        - învățare automată pentru captarea efectelor neliniare și a interacțiunilor
          dintre parametri tehnologici și de proces.
        """
    )
