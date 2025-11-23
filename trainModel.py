import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

np.random.seed(42)
n_samples = 3000

# 1. Generăm date sintetice pentru strunjire aluminiu
# Parametri inspirați din domenii reale de lucru pentru Aluminiu
data = pd.DataFrame({
    # rating intern 1–5: număr operații, setup-uri, suprafețe critice, toleranțe strânse
    "part_complexity": np.random.randint(1, 6, size=n_samples),

    # viteza de așchiere [m/min] – valori uzuale pentru Al (viteze relativ mari)
    "cutting_speed_m_min": np.random.uniform(150, 600, size=n_samples),

    # avans per rotație [mm/rev]
    "feed_mm_rev": np.random.uniform(0.05, 0.35, size=n_samples),

    # adâncimea de așchiere [mm]
    "depth_of_cut_mm": np.random.uniform(0.2, 4.0, size=n_samples),

    # geometria piesei
    "part_diameter_mm": np.random.uniform(20, 150, size=n_samples),
    "part_length_mm": np.random.uniform(10, 200, size=n_samples),

    # duritatea materialului (aliaj de aluminiu) [HRC]
    "material_hardness_HRC": np.random.uniform(20, 32, size=n_samples),

    # uzura sculei – lățimea craterului de uzură VB [mm]
    "tool_wear_VB_mm": np.random.uniform(0.05, 0.30, size=n_samples),

    # debit lichid de răcire [L/min]
    "coolant_flow_l_min": np.random.uniform(2, 12, size=n_samples),
})

# 2. Model fizic simplificat pentru timpul de așchiere
# T_mach [sec] = 60 * (L * pi * d) / (1000 * v_c * f)
pi = np.pi
L = data["part_length_mm"]
d = data["part_diameter_mm"]
v_c = data["cutting_speed_m_min"]
f = data["feed_mm_rev"]

# evităm împărțirea la zero (teoretic nu ar trebui să apară)
v_c = np.clip(v_c, 1e-3, None)
f = np.clip(f, 1e-3, None)

T_mach_sec = 60.0 * (L * pi * d) / (1000.0 * v_c * f)

# 3. Termeni suplimentari pentru realism
# T_total = T0 + T_mach + efecte ale complexității, durității, uzurii, răcirii
T0 = 15.0  # timpi auxiliari (prindere piesă, repoziționare, etc.) [sec]

# efectele sunt puse ca termeni liniari:
complexity_term = (data["part_complexity"] - 1) * 12.0          # fiecare treaptă + ~12s
hardness_term = (data["material_hardness_HRC"] - 24.0) * 1.5    # Al mai dur → timp puțin mai mare
wear_term = (data["tool_wear_VB_mm"] - 0.05) * 80.0             # uzură mai mare → timp mai mare
coolant_term = - (data["coolant_flow_l_min"] - 2.0) * 1.2       # răcire mai mare → timp ușor mai mic

# 4. Timpul teoretic + zgomot statistic (simulare variații reale)
cycle_time_sec = (
    T0
    + T_mach_sec
    + complexity_term
    + hardness_term
    + wear_term
    + coolant_term
)

# zgomot gaussian pentru a simula variații de proces, operator, mașină
noise = np.random.normal(0, 6, size=n_samples)
cycle_time_sec = cycle_time_sec + noise

# nu vrem valori negative sau absurde
cycle_time_sec = np.clip(cycle_time_sec, 5, None)

data["cycle_time_sec"] = cycle_time_sec

# 5. Pregătim datele pentru ML
X = data.drop(columns=["cycle_time_sec"])
y = data["cycle_time_sec"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Model ML – Random Forest Regression
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=10,
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train, y_train)

# 7. Evaluare simplă
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("===== MODEL PERFORMANCE (TURNING ALUMINIUM) =====")
print(f"MAE: {mae:.2f} sec")
print(f"R² : {r2:.3f}")
print("=================================================")

joblib.dump(
    {"model": model, "feature_names": list(X.columns)},
    "model.pkl"
)

print("Model salvat în model.pkl")
