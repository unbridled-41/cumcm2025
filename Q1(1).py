# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import pearsonr, spearmanr


# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ---- 1) 读取 Excel（两个子表） ----
excel_path = "附件.xlsx"  # 确保与脚本在同一路径下
sheets = pd.read_excel(excel_path, sheet_name=None)  # 读全部子表

# 尝试获取“男胎检测数据”与“女胎检测数据”
male_df = None
female_df = None
for name, _df in sheets.items():
    if "男胎" in name:
        male_df = _df.copy()
    if "女胎" in name:
        female_df = _df.copy()

assert male_df is not None, "未找到包含'男胎'的子表，请检查Excel表名。"

# ---- 2) 模糊匹配列名工具 ----
def find_col(cols, keywords):
    cols_clean = [str(c).strip() for c in cols]
    for kw in keywords:
        for c in cols_clean:
            if kw in c:
                return c
    return None

# 目标字段：孕周、BMI、Y染色体浓度
col_week = find_col(male_df.columns, ["检测孕周", "孕周"])
col_bmi  = find_col(male_df.columns, ["孕妇BMI", "BMI"])
col_yconc = find_col(male_df.columns, ["Y染色体浓度", "Y 浓度", "Y染色体 游离", "Y浓度"])

missing = [("检测孕周", col_week), ("BMI", col_bmi), ("Y染色体浓度", col_yconc)]
missing_msg = [name for name, val in missing if val is None]
assert len(missing_msg) == 0, f"找不到关键列：{', '.join(missing_msg)}；请检查Excel表头。"

# ---- 3) 清洗：只保留Y浓度非空（男胎） ----
male_df = male_df[male_df[col_yconc].notna()].copy()

# ---- 4) 孕周字符串 -> 数值(周) ----
wk_re = re.compile(r"^\s*(\d+)\s*(?:w|周)?\s*\+\s*(\d+)\s*(?:d|天)?\s*$", re.IGNORECASE)

def parse_weeks(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    m = wk_re.match(s)
    if m:
        w = int(m.group(1))
        d = int(m.group(2))
        return w + d/7.0
    num = re.findall(r"\d+\.?\d*", s)
    if len(num) == 1:
        try:
            return float(num[0])
        except:
            return np.nan
    return np.nan

male_df["孕周_周"] = male_df[col_week].apply(parse_weeks)
male_df = male_df[male_df["孕周_周"].notna()].copy()

# ---- 5) 散点图 ----
plt.figure()
plt.scatter(male_df["孕周_周"], male_df[col_yconc], alpha=0.6)
plt.xlabel("孕周（周）")
plt.ylabel("Y染色体浓度（比例）")
plt.title("孕周 vs Y染色体浓度")
plt.tight_layout()
plt.show()

plt.figure()
plt.scatter(male_df[col_bmi], male_df[col_yconc], alpha=0.6)
plt.xlabel("BMI")
plt.ylabel("Y染色体浓度（比例）")
plt.title("BMI vs Y染色体浓度")
plt.tight_layout()
plt.show()

# ---- 6) 相关性分析 ----
def pairwise_dropna(a, b):
    m = pd.notna(a) & pd.notna(b)
    return a[m].astype(float), b[m].astype(float)

x1, y1 = pairwise_dropna(male_df["孕周_周"], male_df[col_yconc])
pearson_week = pearsonr(x1, y1) if len(x1) > 2 else (np.nan, np.nan)
spearman_week = spearmanr(x1, y1) if len(x1) > 2 else (np.nan, np.nan)

x2, y2 = pairwise_dropna(male_df[col_bmi], male_df[col_yconc])
pearson_bmi = pearsonr(x2, y2) if len(x2) > 2 else (np.nan, np.nan)
spearman_bmi = spearmanr(x2, y2) if len(x2) > 2 else (np.nan, np.nan)

print("=== 相关性分析（男胎）===")
print(f"孕周 vs Y浓度  Pearson r={pearson_week[0]:.4f}, p={pearson_week[1]:.3e}")
print(f"孕周 vs Y浓度  Spearman ρ={spearman_week[0]:.4f}, p={spearman_week[1]:.3e}")
print(f"BMI  vs Y浓度  Pearson r={pearson_bmi[0]:.4f}, p={pearson_bmi[1]:.3e}")
print(f"BMI  vs Y浓度  Spearman ρ={spearman_bmi[0]:.4f}, p={spearman_bmi[1]:.3e}")

# ---- 7) 多元线性回归 OLS ----
reg_df = male_df[["孕周_周", col_bmi, col_yconc]].dropna().copy()
reg_df["交互项"] = reg_df["孕周_周"] * reg_df[col_bmi]

features = ["孕周_周", col_bmi, "交互项"]
X = reg_df[features]
y = reg_df[col_yconc].astype(float)
X = sm.add_constant(X)
ols_model = sm.OLS(y, X).fit()

print("\\n=== 多元线性回归（含交互项）===")
print(ols_model.summary())
