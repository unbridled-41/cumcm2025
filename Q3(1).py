import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib
import re

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 读取数据
df_male = pd.read_excel("附件.xlsx", sheet_name="男胎检测数据")

# 转换孕周格式为数值
def convert_week_to_numeric(week_str):
    if pd.isna(week_str):
        return np.nan
    if isinstance(week_str, (int, float)):
        return week_str

    week_str = str(week_str).lower().replace('周', '').replace('w', '')
    match = re.search(r'(\d+)(?:\+(\d+))?', week_str)
    if match:
        weeks = int(match.group(1))
        days = int(match.group(2)) if match.group(2) else 0
        return weeks + days / 7
    return np.nan

df_male['检测孕周数值'] = df_male['检测孕周'].apply(convert_week_to_numeric)

# 关键修正：对每个孕妇找到最早达标时间
print("处理每个孕妇的最早达标时间...")

# 标记达标样本
df_male['达标'] = df_male['Y染色体浓度'] >= 0.04

# 对每个孕妇，找到最早达标的检测时间
earliest_pass_df = df_male[df_male['达标']].groupby('孕妇代码').agg({
    '检测孕周数值': 'min',
    'Y染色体浓度': 'first',
    '孕妇BMI': 'first',
    '年龄': 'first',
    '身高': 'first',
    '体重': 'first'
}).reset_index()

earliest_pass_df.rename(columns={'检测孕周数值': '最早达标孕周'}, inplace=True)

print(f"总孕妇数: {df_male['孕妇代码'].nunique()}")
print(f"有达标记录的孕妇数: {len(earliest_pass_df)}")
print(f"达标孕妇比例: {len(earliest_pass_df) / df_male['孕妇代码'].nunique() * 100:.1f}%")

# 合并回原始数据，为每个孕妇添加最早达标时间
df_male = df_male.merge(earliest_pass_df[['孕妇代码', '最早达标孕周']], on='孕妇代码', how='left')

# 对于没有达标记录的孕妇，设置最早达标孕周为NaN
df_male.loc[df_male['最早达标孕周'].isna(), '最早达标孕周'] = np.nan

# 1. 特征选择
feature_rename = {
    '年龄': '年龄',
    '身高': '身高',
    '体重': '体重',
    '孕妇BMI': 'BMI',
    'GC含量': '总GC含量',
    '13号染色体的GC含量': 'GC13',
    '18号染色体的GC含量': 'GC18',
    '21号染色体的GC含量': 'GC21'
}

features = list(feature_rename.keys())

# 使用每个孕妇的第一条记录进行分析（避免重复）
unique_patients = df_male.drop_duplicates('孕妇代码')

# 计算相关系数
plot_data = unique_patients[features + ['Y染色体浓度']].copy()
plot_data.rename(columns=feature_rename, inplace=True)

corr = plot_data.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f',
            annot_kws={'size': 10}, cbar_kws={'shrink': 0.8})
plt.title("特征与Y染色体浓度的相关系数热力图", fontsize=14, pad=20)
plt.tight_layout()
plt.savefig('Q3-相关系数热力图.png', dpi=300, bbox_inches='tight')
plt.close()

# 随机森林特征重要性
X = unique_patients[features].fillna(0)
y = unique_patients['Y染色体浓度'].fillna(0)
model = RandomForestRegressor(random_state=42)
model.fit(X, y)

importance_df = pd.DataFrame({
    '特征': [feature_rename.get(f, f) for f in features],
    '重要性': model.feature_importances_
}).sort_values('重要性', ascending=False)

print("随机森林特征重要性:")
print(importance_df.to_string(index=False))
print()

# 3. 层次聚类BMI分组（基于唯一孕妇）
bmi_data = unique_patients[['孕妇BMI']].dropna().values

plt.figure(figsize=(12, 6))
Z = linkage(bmi_data, method='ward')
dendrogram(Z, truncate_mode='level', p=5)
plt.title('BMI层次聚类树状图')
plt.xlabel('样本索引')
plt.ylabel('距离')
plt.axhline(y=15, color='r', linestyle='--')
plt.savefig('Q3-BMI层次聚类树状图.png', dpi=300, bbox_inches='tight')
plt.close()

n_clusters = 4
hierarchical = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', linkage='ward')
unique_patients['BMI分组'] = hierarchical.fit_predict(bmi_data)

# 将分组结果映射回原始数据
bmi_group_mapping = unique_patients[['孕妇代码', 'BMI分组']]
df_male = df_male.merge(bmi_group_mapping, on='孕妇代码', how='left')

# 输出每个分组的BMI区间
group_stats = unique_patients.groupby('BMI分组')['孕妇BMI'].agg(['min', 'max', 'mean', 'count'])
print("各BMI分组的统计信息:")
print(group_stats)
print()

# 4. 风险函数
def risk(week):
    if pd.isna(week):
        return 3  # 从未达标，最高风险
    if week <= 12:
        return 0  # 早期，风险低
    elif week <= 27:
        return 1  # 中期，风险高
    else:
        return 2  # 晚期，风险极高

# 计算每个孕妇的风险值
unique_patients['风险值'] = unique_patients['最早达标孕周'].apply(risk)

# 5. 最佳时点选择
groups = unique_patients.groupby('BMI分组')
best_weeks = {}

print("各BMI分组的最佳NIPT时点:")
print("-" * 70)
for group_id, group_df in groups:
    if len(group_df) > 0:
        # 计算该组中达标孕妇的最早达标孕周分布
        passed_patients = group_df[~group_df['最早达标孕周'].isna()]

        if len(passed_patients) > 0:
            # 找到能够覆盖一定比例孕妇的孕周
            sorted_weeks = passed_patients['最早达标孕周'].sort_values()
            # 选择第75百分位的孕周作为推荐时点（覆盖75%的达标孕妇）
            recommended_week = sorted_weeks.quantile(0.75)

            # 计算在该孕周检测的风险
            risk_at_week = risk(recommended_week)

            best_weeks[group_id] = (recommended_week, risk_at_week)

            bmi_min = group_df['孕妇BMI'].min()
            bmi_max = group_df['孕妇BMI'].max()

            print(f"分组 {group_id + 1} (BMI范围: {bmi_min:.1f}-{bmi_max:.1f}):")
            print(f"  推荐检测孕周: {recommended_week:.2f} 周")
            print(f"  在该孕周检测的风险等级: {risk_at_week}")

            # 统计信息
            total_patients = len(group_df)
            passed_patients_count = len(passed_patients)
            pass_rate = (passed_patients_count / total_patients) * 100

            print(f"  孕妇总数: {total_patients}")
            print(f"  达标孕妇数: {passed_patients_count}")
            print(f"  达标率: {pass_rate:.1f}%")
            print(f"  平均最早达标孕周: {passed_patients['最早达标孕周'].mean():.2f}周")
            print("-" * 70)
        else:
            print(f"分组 {group_id + 1}: 没有孕妇达到4%的Y染色体浓度")
            print("-" * 70)

# 可视化结果 - 每个图单独保存

# 图1: 各分组最早达标孕周分布
plt.figure(figsize=(10, 6))
for group_id in range(n_clusters):
    group_data = unique_patients[unique_patients['BMI分组'] == group_id]
    passed_data = group_data[~group_data['最早达标孕周'].isna()]

    if len(passed_data) > 0:
        plt.scatter([group_id + 1] * len(passed_data), passed_data['最早达标孕周'],
                    alpha=0.6, label=f'分组{group_id + 1}' if group_id == 0 else "")

        # 标注中位数
        median_week = passed_data['最早达标孕周'].median()
        plt.plot(group_id + 1, median_week, 'ro', markersize=8)

plt.axhline(y=12, color='g', linestyle='--', alpha=0.7, label='12周（低风险截止）')
plt.axhline(y=27, color='orange', linestyle='--', alpha=0.7, label='27周（高风险截止）')
plt.xlabel('BMI分组')
plt.ylabel('最早达标孕周')
plt.title('各分组孕妇的最早达标孕周分布')
plt.xticks(range(1, n_clusters + 1), [f'分组{i}' for i in range(1, n_clusters + 1)])
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Q3-各分组最早达标孕周分布.png', dpi=300, bbox_inches='tight')
plt.close()

# 图2: 各分组达标率
plt.figure(figsize=(10, 6))
pass_rates = []
for group_id in range(n_clusters):
    group_data = unique_patients[unique_patients['BMI分组'] == group_id]
    pass_rate = (len(group_data[~group_data['最早达标孕周'].isna()]) / len(group_data)) * 100
    pass_rates.append(pass_rate)

plt.bar(range(1, n_clusters + 1), pass_rates, color='skyblue')
plt.xlabel('BMI分组')
plt.ylabel('达标率 (%)')
plt.title('各分组的孕妇达标率')
plt.xticks(range(1, n_clusters + 1), [f'分组{i}' for i in range(1, n_clusters + 1)])
for i, v in enumerate(pass_rates):
    plt.text(i + 1, v + 1, f'{v:.1f}%', ha='center')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Q3-各分组达标率.png', dpi=300, bbox_inches='tight')
plt.close()

# 图3: 风险分布
plt.figure(figsize=(10, 6))
risk_counts = unique_patients.groupby(['BMI分组', '风险值']).size().unstack(fill_value=0)
risk_counts.plot(kind='bar', stacked=True, colormap='RdYlGn_r')
plt.xlabel('BMI分组')
plt.ylabel('孕妇数量')
plt.title('各分组的风险分布')
plt.xticks(rotation=0)
plt.legend(['低风险(0)', '中风险(1)', '高风险(2)', '极高风险(3)'])
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Q3-风险分布.png', dpi=300, bbox_inches='tight')
plt.close()

# 图4: BMI与最早达标孕周的关系
plt.figure(figsize=(10, 6))
passed_patients = unique_patients[~unique_patients['最早达标孕周'].isna()]
scatter = plt.scatter(passed_patients['孕妇BMI'], passed_patients['最早达标孕周'],
            c=passed_patients['BMI分组'], cmap='viridis', alpha=0.6)
plt.colorbar(scatter, label='BMI分组')
plt.xlabel('孕妇BMI')
plt.ylabel('最早达标孕周')
plt.title('BMI与最早达标孕周的关系')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Q3-BMI与达标孕周关系.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n🎯 最终NIPT时点建议（基于孕妇级别分析）:")
print("=" * 80)
for group_id in range(n_clusters):
    if group_id in best_weeks:
        group_data = unique_patients[unique_patients['BMI分组'] == group_id]
        bmi_min = group_data['孕妇BMI'].min()
        bmi_max = group_data['孕妇BMI'].max()

        print(f"分组 {group_id + 1} (BMI: {bmi_min:.1f}-{bmi_max:.1f}):")
        print(f"  📅 推荐检测时间: 孕{best_weeks[group_id][0]:.1f}周")
        print(f"  ⚠️  预期风险等级: {best_weeks[group_id][1]}")
        print(f"  📊 覆盖比例: 约75%的达标孕妇在此孕周前已达标")
        print()

print("所有图表已保存为PNG文件，文件名以'Q3-'为前缀")