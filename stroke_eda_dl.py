# %% [markdown]
# # EDA and Visualization of the Stroke Prediction Dataset + Deep Learning (MLP)
# Course: U21ADP05 Exploratory Data Analysis and Visualization - Assignment II
#
# Visualization pipeline (Fry's 7 stages): Acquire -> Parse -> Filter -> Mine -> Represent -> Refine -> Interact

# %%
# ---------- 0. Imports & global settings ----------
import os, json, random, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve,
                             average_precision_score, accuracy_score, precision_score,
                             recall_score, f1_score, ConfusionMatrixDisplay)
from sklearn.utils.class_weight import compute_class_weight
warnings.filterwarnings("ignore")
SEED = 42
random.seed(SEED); np.random.seed(SEED)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
keras.utils.set_random_seed(SEED)

FIG = "figures"; os.makedirs(FIG, exist_ok=True)
sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight"})
PAL = {0: "#4C72B0", 1: "#C44E52"}
results = {}

def save(name):
    plt.savefig(f"{FIG}/{name}.png"); plt.close()

# %%
# ---------- STAGE 1: ACQUIRE ----------
df_raw = pd.read_csv("data/healthcare-dataset-stroke-data.csv")
print("Shape:", df_raw.shape)
print(df_raw.dtypes)
results["raw_shape"] = list(df_raw.shape)

# ---------- STAGE 2: PARSE / UNDERSTAND ----------
print(df_raw.describe(include="all").T)
print("Missing values:\n", df_raw.isna().sum())
print("Duplicate rows:", df_raw.duplicated().sum(), "| duplicate ids:", df_raw["id"].duplicated().sum())
results["missing_bmi"] = int(df_raw["bmi"].isna().sum())
results["duplicates"] = int(df_raw.duplicated().sum())
results["class_counts"] = {int(k): int(v) for k, v in df_raw["stroke"].value_counts().items()}

# %%
# ---------- STAGE 3: FILTER (cleaning) ----------
df = df_raw.copy()
df = df.drop_duplicates().drop(columns="id")            # id is an identifier, not a feature
df = df[df["gender"] != "Other"].reset_index(drop=True)  # single record -> cannot be learned / stratified

def iqr_bounds(s):
    q1, q3 = s.quantile([.25, .75]); i = q3 - q1
    return q1 - 1.5 * i, q3 + 1.5 * i

out_counts = {}
for c in ["age", "avg_glucose_level", "bmi"]:
    lo, hi = iqr_bounds(df[c].dropna())
    out_counts[c] = int(((df[c] < lo) | (df[c] > hi)).sum())
print("IQR outliers:", out_counts)
results["outliers_iqr"] = out_counts

# Feature engineering -> brings predictors from 10 to 15
def add_features(d):
    d = d.copy()
    d["age_group"] = pd.cut(d["age"], [-1, 17, 39, 59, 200], labels=["child", "young_adult", "middle_aged", "senior"]).astype(str)
    d["bmi_category"] = pd.cut(d["bmi"], [0, 18.5, 25, 30, 200], labels=["underweight", "normal", "overweight", "obese"]).astype(object)
    d["bmi_category"] = d["bmi_category"].where(d["bmi"].notna(), "missing")
    d["glucose_category"] = pd.cut(d["avg_glucose_level"], [0, 100, 125, 1000], labels=["normal", "prediabetic", "diabetic"]).astype(str)
    d["comorbidity_count"] = d["hypertension"] + d["heart_disease"]
    d["is_senior"] = (d["age"] >= 60).astype(int)
    return d

df = add_features(df)
print(df.head())
results["clean_shape"] = list(df.shape)

# %%
# ---------- STAGE 4: MINE (statistics) ----------
num = ["age", "avg_glucose_level", "bmi"]
cat = ["gender", "hypertension", "heart_disease", "ever_married", "work_type", "Residence_type", "smoking_status"]
summ = df[num].describe().T
summ["skew"] = df[num].skew(); summ["kurtosis"] = df[num].kurt()
print(summ)
results["numeric_summary"] = summ.round(3).to_dict()

stat_tests = {}
for c in num:   # Mann-Whitney U (non-normal) between stroke / no stroke
    a, b = df.loc[df.stroke == 1, c].dropna(), df.loc[df.stroke == 0, c].dropna()
    u, p = stats.mannwhitneyu(a, b)
    stat_tests[f"MWU_{c}"] = {"median_stroke": float(a.median()), "median_nostroke": float(b.median()), "p": float(p)}
for c in cat + ["age_group", "bmi_category", "glucose_category"]:
    ct = pd.crosstab(df[c], df["stroke"]); chi, p, dof, _ = stats.chi2_contingency(ct)
    n = ct.values.sum(); v = np.sqrt(chi / (n * (min(ct.shape) - 1)))
    stat_tests[f"chi2_{c}"] = {"chi2": float(chi), "p": float(p), "cramers_v": float(v)}
for c in num:
    d = df[[c, "stroke"]].dropna(); r, p = stats.pointbiserialr(d["stroke"], d[c])
    stat_tests[f"pointbiserial_{c}"] = {"r": float(r), "p": float(p)}
results["stat_tests"] = stat_tests
for k, v in stat_tests.items(): print(k, {a: round(b, 4) for a, b in v.items()})

# Distribution fitting (CO4): normality + candidate continuous distributions
fits = {}
for c in num:
    x = df[c].dropna().values
    sh_p = stats.normaltest(x).pvalue
    ks_norm = stats.kstest(x, "norm", args=(x.mean(), x.std()))
    ks_lnorm = stats.kstest(x, "lognorm", args=stats.lognorm.fit(x, floc=0))
    fits[c] = {"dagostino_p": float(sh_p), "KS_normal": float(ks_norm.statistic), "KS_lognormal": float(ks_lnorm.statistic)}
# Discrete: stroke counts follow Binomial(n, p)
p_hat = df["stroke"].mean(); n_tot = len(df)
fits["stroke_binomial"] = {"n": n_tot, "p_hat": float(p_hat), "mean": float(n_tot * p_hat), "sd": float(np.sqrt(n_tot * p_hat * (1 - p_hat)))}
fits["comorbidity_distribution"] = {int(k): float(v) for k, v in df["comorbidity_count"].value_counts(normalize=True).sort_index().round(4).items()}
results["distribution_fits"] = fits
print(json.dumps(fits, indent=1, default=str))

# %%
# ---------- STAGE 5: REPRESENT (visualisations) ----------
# V1: Target class distribution (+ stroke rate by age group)
fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
cnt = df["stroke"].value_counts().sort_index()
bars = ax[0].bar(["No stroke (0)", "Stroke (1)"], cnt.values, color=[PAL[0], PAL[1]])
for b, v in zip(bars, cnt.values):
    ax[0].text(b.get_x() + b.get_width() / 2, v + 40, f"{v}\n({v/len(df):.1%})", ha="center", fontsize=9)
ax[0].set(title="(a) Target class distribution", ylabel="Patients"); ax[0].set_ylim(0, cnt.max() * 1.18)
order = ["child", "young_adult", "middle_aged", "senior"]
rate = df.groupby("age_group")["stroke"].mean().reindex(order) * 100
sns.barplot(x=rate.index, y=rate.values, color="#C44E52", ax=ax[1])
for i, v in enumerate(rate.values): ax[1].text(i, v + .2, f"{v:.1f}%", ha="center", fontsize=9)
ax[1].set(title="(b) Stroke rate by age group", ylabel="Stroke rate (%)", xlabel="")
plt.tight_layout(); save("fig01_class_distribution")

# V2: Histograms + KDE of numeric features by class
fig, ax = plt.subplots(1, 3, figsize=(11, 3.4))
for a, c, t in zip(ax, num, ["Age (years)", "Avg. glucose (mg/dL)", "BMI (kg/m$^2$)"]):
    sns.histplot(data=df, x=c, hue="stroke", kde=True, stat="density", common_norm=False, palette=PAL, ax=a, bins=30)
    a.set(xlabel=t, title=f"Distribution of {c}")
plt.tight_layout(); save("fig02_histograms")

# V3: Boxplots (outliers)
fig, ax = plt.subplots(1, 3, figsize=(10, 3.4))
for a, c in zip(ax, num):
    sns.boxplot(data=df, x=df["stroke"].astype(str), y=c, hue=df["stroke"].astype(str), palette={"0": PAL[0], "1": PAL[1]}, legend=False, ax=a, flierprops=dict(marker=".", markersize=3))
    a.set(xlabel="Stroke", title=f"{c} vs stroke")
plt.tight_layout(); save("fig03_boxplots")

# V4: Correlation heatmap
cm_df = df[["age", "avg_glucose_level", "bmi", "hypertension", "heart_disease", "comorbidity_count", "is_senior", "stroke"]].copy()
cm_df["ever_married"] = (df["ever_married"] == "Yes").astype(int)
corr = cm_df.corr(method="spearman")
plt.figure(figsize=(6.2, 5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, linewidths=.5, cbar_kws={"label": "Spearman rho"}, annot_kws={"size": 8})
plt.title("Spearman correlation heatmap"); save("fig04_correlation_heatmap")
results["corr_with_stroke"] = corr["stroke"].drop("stroke").round(3).to_dict()

# V5: Stroke rate across categorical features
fig, ax = plt.subplots(2, 4, figsize=(13, 6)); ax = ax.ravel()
for a, c in zip(ax, cat + ["glucose_category"]):
    r = df.groupby(c)["stroke"].mean() * 100
    sns.barplot(x=r.index.astype(str), y=r.values, ax=a, color="#4C72B0")
    a.set(title=c, xlabel="", ylabel="Stroke rate (%)"); a.tick_params(axis="x", rotation=25)
plt.tight_layout(); save("fig05_categorical_stroke_rate")

# V6: Scatter age vs glucose coloured by stroke (bmi as size)
plt.figure(figsize=(6.2, 4.4))
sns.scatterplot(data=df.sort_values("stroke"), x="age", y="avg_glucose_level", hue="stroke", palette=PAL, size="bmi", sizes=(8, 70), alpha=.55)
plt.title("Age vs glucose (point size = BMI)"); plt.xlabel("Age (years)"); plt.ylabel("Avg. glucose (mg/dL)")
plt.legend(fontsize=7, loc="upper left", ncol=2); save("fig06_scatter_age_glucose")

# V7: Distribution fitting (Normal vs Log-normal for BMI & glucose, Q-Q plot)
fig, ax = plt.subplots(2, 2, figsize=(9, 6.4))
for j, c in enumerate(["bmi", "avg_glucose_level"]):
    x = df[c].dropna().values; xs = np.linspace(x.min(), x.max(), 300)
    ax[0, j].hist(x, bins=40, density=True, color="#9ab", alpha=.7, label="Data")
    ax[0, j].plot(xs, stats.norm.pdf(xs, x.mean(), x.std()), "r-", label="Normal")
    ax[0, j].plot(xs, stats.lognorm.pdf(xs, *stats.lognorm.fit(x, floc=0)), "g--", label="Log-normal")
    ax[0, j].set(title=f"PDF fit: {c}", xlabel=c); ax[0, j].legend(fontsize=8)
    stats.probplot(np.log(x), dist="norm", plot=ax[1, j]); ax[1, j].set_title(f"Q-Q plot of log({c})")
plt.tight_layout(); save("fig07_distribution_fits")

# V8: Discrete distributions: Binomial for stroke, comorbidity PMF
fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
sd = fits["stroke_binomial"]["sd"]
k = np.arange(int(n_tot * p_hat - 4 * sd), int(n_tot * p_hat + 4 * sd))
ax[0].bar(k, stats.binom.pmf(k, n_tot, p_hat), color="#C44E52", width=1)
ax[0].axvline(df["stroke"].sum(), color="k", ls="--", label=f"Observed = {df['stroke'].sum()}")
ax[0].set(title=f"Binomial(n={n_tot}, p={p_hat:.3f}) for stroke count", xlabel="Number of strokes", ylabel="PMF"); ax[0].legend()
cc = df["comorbidity_count"].value_counts(normalize=True).sort_index()
sns.barplot(x=cc.index, y=cc.values, ax=ax[1], color="#4C72B0")
ax[1].set(title="PMF of comorbidity count", xlabel="Comorbidities (hypertension + heart disease)", ylabel="Proportion")
plt.tight_layout(); save("fig08_discrete_distributions")

# V9: Pairplot (sample)
pp = sns.pairplot(df[num + ["stroke"]].dropna().sample(1500, random_state=SEED), hue="stroke", palette=PAL,
                  corner=True, plot_kws=dict(s=12, alpha=.5), height=2)
pp.fig.suptitle("Pairplot of numeric features", y=1.01); pp.savefig(f"{FIG}/fig09_pairplot.png"); plt.close("all")

# %%
# ---------- STAGE 6: REFINE (preprocessing + split) ----------
y = df["stroke"].values
X_all = df.drop(columns="stroke")
X_tr_, X_tmp, y_tr, y_tmp = train_test_split(X_all, y, test_size=0.30, stratify=y, random_state=SEED)
X_va_, X_te_, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=SEED)
print("Split sizes train/val/test:", len(X_tr_), len(X_va_), len(X_te_), "| stroke rate:", y_tr.mean().round(4), y_va.mean().round(4), y_te.mean().round(4))
results["split"] = {"train": len(X_tr_), "val": len(X_va_), "test": len(X_te_),
                    "stroke_train": int(y_tr.sum()), "stroke_val": int(y_va.sum()), "stroke_test": int(y_te.sum())}

# --- fit all transformations on TRAIN only (no leakage) ---
bmi_median = X_tr_.groupby("age_group")["bmi"].median().to_dict()                  # group-wise median imputation
bounds = {c: iqr_bounds(X_tr_[c].dropna()) for c in ["avg_glucose_level", "bmi"]}  # winsorise glucose & bmi

def transform_basic(d):
    d = d.copy()
    d["bmi"] = d.apply(lambda r: bmi_median[r["age_group"]] if pd.isna(r["bmi"]) else r["bmi"], axis=1)
    for c, (lo, hi) in bounds.items(): d[c] = d[c].clip(lo, hi)
    d["bmi_category"] = pd.cut(d["bmi"], [0, 18.5, 25, 30, 200], labels=["underweight", "normal", "overweight", "obese"]).astype(str)
    return d

X_tr_, X_va_, X_te_ = transform_basic(X_tr_), transform_basic(X_va_), transform_basic(X_te_)

num_cols = ["age", "avg_glucose_level", "bmi", "comorbidity_count"]
cat_cols = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status", "age_group", "bmi_category", "glucose_category"]
bin_cols = ["hypertension", "heart_disease", "is_senior"]

def encode(d, cols_ref=None):
    out = pd.get_dummies(d[cat_cols + bin_cols + num_cols], columns=cat_cols, dtype=float)
    if cols_ref is not None: out = out.reindex(columns=cols_ref, fill_value=0.0)
    return out

E_tr = encode(X_tr_); cols = list(E_tr.columns)
E_va, E_te = encode(X_va_, cols), encode(X_te_, cols)
scaler = StandardScaler().fit(E_tr[num_cols])

def finish(E):
    E = E.copy().astype("float64"); E[num_cols] = scaler.transform(E[num_cols]); return E.values.astype("float32")

Xtr, Xva, Xte = finish(E_tr), finish(E_va), finish(E_te)
print("Final model input dimension:", Xtr.shape[1])
results["n_input_features"] = int(Xtr.shape[1])

# %%
# ---------- STAGE 7: DEEP LEARNING MODEL (MLP) ----------
cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_tr); cw = {0: float(cw[0]), 1: float(cw[1])}

def build_mlp(in_dim, units=(64, 32), dropout=0.3, lr=1e-3, l2=1e-4):
    reg = keras.regularizers.l2(l2)
    m = keras.Sequential([layers.Input((in_dim,))])
    for u in units:
        m.add(layers.Dense(u, activation="relu", kernel_regularizer=reg)); m.add(layers.BatchNormalization()); m.add(layers.Dropout(dropout))
    m.add(layers.Dense(1, activation="sigmoid"))
    m.compile(optimizer=keras.optimizers.Adam(lr), loss="binary_crossentropy",
              metrics=["accuracy", keras.metrics.AUC(name="auc")])
    return m

def fit_model(params, epochs=150, verbose=0):
    keras.utils.set_random_seed(SEED)
    m = build_mlp(Xtr.shape[1], params["units"], params["dropout"], params["lr"], params["l2"])
    es = keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=15, restore_best_weights=True)
    h = m.fit(Xtr, y_tr, validation_data=(Xva, y_va), epochs=epochs, batch_size=params["batch"],
              class_weight=cw, callbacks=[es], verbose=verbose)
    return m, h

# basic hyper-parameter tuning (grid search on validation AUC)
grid = list(ParameterGrid({"units": [(32,), (64, 32), (128, 64)], "dropout": [0.2, 0.4], "lr": [1e-3, 3e-3], "batch": [64], "l2": [1e-4]}))
tune = []
for p in grid:
    m, h = fit_model(p)
    va_auc = roc_auc_score(y_va, m.predict(Xva, verbose=0).ravel())
    tune.append({"units": str(p["units"]), "dropout": p["dropout"], "lr": p["lr"], "batch": p["batch"], "val_auc": va_auc, "epochs": len(h.history["loss"])})
    print(tune[-1])
tune_df = pd.DataFrame(tune)
best = grid[int(tune_df["val_auc"].idxmax())]
tune_df = tune_df.sort_values("val_auc", ascending=False); print(tune_df)
tune_df.to_csv("hyperparameter_tuning_results.csv", index=False)
print("Best hyper-parameters:", best)
results["best_params"] = {k: str(v) for k, v in best.items()}
results["tuning"] = tune_df.round(4).to_dict("records")

model, hist = fit_model(best)
model.summary()
results["n_params"] = int(model.count_params()); results["epochs_trained"] = len(hist.history["loss"])

# %%
# ---------- MODEL EVALUATION ----------
pv = model.predict(Xva, verbose=0).ravel(); pt = model.predict(Xte, verbose=0).ravel()
fpr, tpr, thr = roc_curve(y_va, pv); thr_opt = float(thr[np.argmax(tpr - fpr)])   # Youden's J on validation
yhat = (pt >= thr_opt).astype(int)

def metrics(y_true, p, yh):
    return {"accuracy": accuracy_score(y_true, yh), "precision": precision_score(y_true, yh, zero_division=0),
            "recall": recall_score(y_true, yh), "f1": f1_score(y_true, yh), "roc_auc": roc_auc_score(y_true, p),
            "pr_auc": average_precision_score(y_true, p)}

m_mlp = metrics(y_te, pt, yhat)
# baseline: logistic regression
lr_m = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xtr, y_tr)
plr = lr_m.predict_proba(Xte)[:, 1]
f_, t_, th_ = roc_curve(y_va, lr_m.predict_proba(Xva)[:, 1]); tl = th_[np.argmax(t_ - f_)]
m_lr = metrics(y_te, plr, (plr >= tl).astype(int))
results["threshold"] = thr_opt; results["test_mlp"] = m_mlp; results["test_logreg"] = m_lr
results["confusion_matrix"] = confusion_matrix(y_te, yhat).tolist()
print("MLP:", {k: round(v, 3) for k, v in m_mlp.items()}); print("LogReg:", {k: round(v, 3) for k, v in m_lr.items()})

# R1/R2/R3: Loss, accuracy, AUC vs epoch
H = hist.history
fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
for a, k, t in zip(ax, ["loss", "accuracy", "auc"], ["Loss vs epoch", "Accuracy vs epoch", "AUC vs epoch"]):
    a.plot(H[k], label="Train"); a.plot(H["val_" + k], label="Validation"); a.set(title=t, xlabel="Epoch", ylabel=k); a.legend()
plt.tight_layout(); save("fig10_training_curves")

# Confusion matrix
fig, ax = plt.subplots(1, 2, figsize=(8, 3.6))
ConfusionMatrixDisplay(confusion_matrix(y_te, yhat), display_labels=["No stroke", "Stroke"]).plot(ax=ax[0], cmap="Blues", colorbar=False)
ax[0].set_title("Counts")
ConfusionMatrixDisplay(confusion_matrix(y_te, yhat, normalize="true"), display_labels=["No stroke", "Stroke"]).plot(ax=ax[1], cmap="Blues", colorbar=False, values_format=".2f")
ax[1].set_title("Row-normalised"); plt.tight_layout(); save("fig11_confusion_matrix")

# ROC and PR curves (test)
fig, ax = plt.subplots(1, 2, figsize=(9, 3.8))
for p_, n_, c_ in [(pt, "MLP", "#C44E52"), (plr, "Logistic Reg.", "#4C72B0")]:
    f, t2, _ = roc_curve(y_te, p_); ax[0].plot(f, t2, color=c_, label=f"{n_} (AUC={roc_auc_score(y_te, p_):.3f})")
    pr, rc, _ = precision_recall_curve(y_te, p_); ax[1].plot(rc, pr, color=c_, label=f"{n_} (AP={average_precision_score(y_te, p_):.3f})")
ax[0].plot([0, 1], [0, 1], "k--", lw=1); ax[0].set(title="ROC curve (test set)", xlabel="False positive rate", ylabel="True positive rate"); ax[0].legend()
ax[1].axhline(y_te.mean(), color="k", ls="--", lw=1); ax[1].set(title="Precision-recall curve (test set)", xlabel="Recall", ylabel="Precision"); ax[1].legend()
plt.tight_layout(); save("fig12_roc_pr")

# Predicted probability distribution by true class (error analysis for classification)
plt.figure(figsize=(6, 3.6))
sns.histplot(x=pt, hue=y_te, palette=PAL, bins=30, stat="density", common_norm=False, element="step")
plt.axvline(thr_opt, color="k", ls="--", label=f"Threshold = {thr_opt:.2f}")
plt.title("Predicted stroke probability by true class"); plt.xlabel("Predicted probability"); plt.legend(); save("fig13_probability_distribution")

# Permutation feature importance (test AUC drop)
rng = np.random.default_rng(SEED); base = roc_auc_score(y_te, pt); fi = {}
groups = {c: [c] for c in ["age", "avg_glucose_level", "bmi", "hypertension", "heart_disease", "comorbidity_count", "is_senior"]}
for c in cat_cols: groups[c] = [i for i in cols if i.startswith(c + "_")]
for g, cs in groups.items():
    idx = [cols.index(c) for c in cs]; drops = []
    for _ in range(20):
        Xp = Xte.copy(); perm = rng.permutation(len(Xp)); Xp[:, idx] = Xte[perm][:, idx]
        drops.append(base - roc_auc_score(y_te, model.predict(Xp, verbose=0).ravel()))
    fi[g] = (float(np.mean(drops)), float(np.std(drops)))
fi_s = pd.Series({k: v[0] for k, v in fi.items()}).sort_values(); err = [fi[k][1] for k in fi_s.index]
plt.figure(figsize=(6, 4.6)); plt.barh(fi_s.index, fi_s.values, xerr=err, color="#55A868")
plt.xlabel("Mean drop in test ROC-AUC"); plt.title("Permutation feature importance (MLP)"); save("fig14_feature_importance")
results["feature_importance"] = {k: round(v[0], 4) for k, v in sorted(fi.items(), key=lambda kv: -kv[1][0])}

json.dump(results, open("results.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print("DONE")
