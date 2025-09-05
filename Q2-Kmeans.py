# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ========== 1) 读取 Excel ==========
excel_path = "附件.xlsx"
sheets = pd.read_excel(excel_path, sheet_name=None)

male_df = None
for name, _df in sheets.items():
    if "男胎" in name:
        male_df = _df.copy()
        break
assert male_df is not None, "未找到男胎检测数据"

# ========== 2) 列名模糊匹配 ==========
def find_col(cols, keywords):
    cols_clean = [str(c).strip() for c in cols]
    for kw in keywords:
        for c in cols_clean:
            if kw in c:
                return c
    return None

col_week = find_col(male_df.columns, ["检测孕周", "孕周"])
col_bmi  = find_col(male_df.columns, ["孕妇BMI", "BMI"])
col_yconc = find_col(male_df.columns, ["Y染色体浓度"])

assert col_week and col_bmi and col_yconc, "关键列未找到"

# ========== 3) 孕周字符串 -> 数值 ==========
wk_re = re.compile(r"^\s*(\d+)\s*(?:w|周)?\s*\+\s*(\d+)\s*(?:d|天)?\s*$", re.IGNORECASE)

def parse_weeks(x):
    if pd.isna(x): return np.nan
    s = str(x).strip()
    m = wk_re.match(s)
    if m:
        return int(m.group(1)) + int(m.group(2)) / 7
    num = re.findall(r"\d+\.?\d*", s)
    if len(num) == 1:
        try: return float(num[0])
        except: return np.nan
    return np.nan

male_df["孕周_周"] = male_df[col_week].apply(parse_weeks)

# ========== 4) 计算达标时间 ==========
male_df = male_df[[col_bmi, "孕周_周", col_yconc, "孕妇代码"]].dropna()
male_df = male_df.sort_values(["孕妇代码", "孕周_周"])

达标时间 = []
for pid, group in male_df.groupby("孕妇代码"):
    g = group.sort_values("孕周_周")
    hit = g[g[col_yconc] >= 0.04]
    if not hit.empty:
        first_week = hit["孕周_周"].iloc[0]
        bmi_val = g[col_bmi].iloc[0]
        达标时间.append((pid, bmi_val, first_week))

达标_df = pd.DataFrame(达标时间, columns=["孕妇代码", "BMI", "达标孕周"])

# ========== 5) K-Means 聚类得到 BMI 分组 ==========
X = 达标_df[["BMI"]].values

inertias = []
K_range = range(2, 10)  # 2~9 类
for k in K_range:
    km = KMeans(n_clusters=k, random_state=0, n_init=10)
    km.fit(X)
    inertias.append(km.inertia_)

# 绘制肘点图
plt.figure(figsize=(6,4))
plt.plot(K_range, inertias, marker="o")
plt.xlabel("聚类数 K")
plt.ylabel("组内平方和 (Inertia)")
plt.title("K-Means 肘点法选择 K")
plt.grid(True)
plt.tight_layout()
plt.savefig("Q2_elbow.png", dpi=300)
plt.show()

# 选定 K 值（例如 5，可根据肘点法结果调整）
best_k = 5
kmeans = KMeans(n_clusters=best_k, random_state=0, n_init=10)
labels = kmeans.fit_predict(X)
达标_df["BMI组"] = labels

# 按簇号 → 转换为区间
cluster_centers = sorted(kmeans.cluster_centers_.flatten())
cluster_bounds = [min(X)[0]] + [(a+b)/2 for a,b in zip(cluster_centers[:-1], cluster_centers[1:])] + [max(X)[0]]
print("\nK-Means 聚类得到的 BMI 区间：", cluster_bounds)

# ========== 6) 风险函数 ==========
def risk_function(sub, t, alpha=1.0, beta=1.0):
    n = len(sub)
    if n == 0: return np.inf
    early_fail = (sub["达标孕周"] > t).sum() / n
    delay = sub[sub["达标孕周"] < t]
    if len(delay) > 0:
        delay_rate = ((t - delay["达标孕周"]) / t).mean()
    else:
        delay_rate = 0
    return alpha * early_fail + beta * delay_rate

# ========== 7) 每组最佳时点 ==========
result = []
candidate = np.arange(10, 20.1, 0.1)

plt.figure(figsize=(8,6))
for g, sub in 达标_df.groupby("BMI组"):
    if len(sub) == 0:
        continue
    risks = [risk_function(sub, t) for t in candidate]
    best_idx = int(np.argmin(risks))
    best_week = candidate[best_idx]
    mean_week = sub["达标孕周"].mean()
    result.append((g, len(sub), round(mean_week,2), round(best_week,2), round(risks[best_idx],3)))

    plt.plot(candidate, risks, label=f"BMI组 {g} (最佳={best_week:.1f}周)")
    plt.scatter(best_week, risks[best_idx], marker="o")

plt.xlabel("检测孕周 t (周)")
plt.ylabel("风险值")
plt.title("风险函数曲线 (t vs 风险值)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("Q2_risk_curve.png", dpi=300)
plt.show()

# ========== 8) 输出结果 ==========
result_df = pd.DataFrame(result, columns=["BMI组","人数","平均达标孕周","最佳检测时点","最小风险值"])
print("=== 各BMI组的最佳检测时点（基于风险函数 + K-Means分组）===")
print(result_df.to_string(index=False))

# 保存结果
result_df.to_excel("Q2_result_kmeans.xlsx", index=False)
print("\n结果已保存到 Q2_result_kmeans.xlsx, Q2_risk_curve.png 和 Q2_elbow.png")




