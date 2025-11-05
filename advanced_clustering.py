import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import f_oneway
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')
np.random.seed(42)

# Enhanced styling
sns.set_theme(style="whitegrid", palette="Set2")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300

print("="*100)
print("ADVANCED CUSTOMER SEGMENTATION ANALYSIS SYSTEM".center(100))
print("="*100)
print()

# ==================== DATA GENERATION ====================
print("📊 GENERATING ENHANCED CUSTOMER DATA...")
print("-"*100)

n_samples = 500

# Enhanced segments with more realistic attributes
segment1 = {  # Premium Customers
    'Annual_Income': np.random.normal(80, 10, 125),
    'Spending_Score': np.random.normal(75, 10, 125),
    'Age': np.random.normal(35, 5, 125),
    'Num_Purchases': np.random.poisson(50, 125),
    'Avg_Transaction': np.random.normal(150, 20, 125),
    'Days_Since_Last_Purchase': np.random.poisson(15, 125),
    'Customer_Tenure_Months': np.random.normal(36, 12, 125),
    'Return_Rate': np.random.uniform(0.02, 0.08, 125)
}

segment2 = {  # Conservative High Earners
    'Annual_Income': np.random.normal(75, 10, 125),
    'Spending_Score': np.random.normal(25, 10, 125),
    'Age': np.random.normal(45, 5, 125),
    'Num_Purchases': np.random.poisson(15, 125),
    'Avg_Transaction': np.random.normal(80, 15, 125),
    'Days_Since_Last_Purchase': np.random.poisson(45, 125),
    'Customer_Tenure_Months': np.random.normal(48, 15, 125),
    'Return_Rate': np.random.uniform(0.01, 0.05, 125)
}

segment3 = {  # Young Spenders
    'Annual_Income': np.random.normal(30, 8, 125),
    'Spending_Score': np.random.normal(70, 10, 125),
    'Age': np.random.normal(25, 4, 125),
    'Num_Purchases': np.random.poisson(40, 125),
    'Avg_Transaction': np.random.normal(60, 10, 125),
    'Days_Since_Last_Purchase': np.random.poisson(20, 125),
    'Customer_Tenure_Months': np.random.normal(18, 8, 125),
    'Return_Rate': np.random.uniform(0.05, 0.12, 125)
}

segment4 = {  # Budget Conscious
    'Annual_Income': np.random.normal(35, 8, 125),
    'Spending_Score': np.random.normal(30, 10, 125),
    'Age': np.random.normal(50, 6, 125),
    'Num_Purchases': np.random.poisson(10, 125),
    'Avg_Transaction': np.random.normal(40, 8, 125),
    'Days_Since_Last_Purchase': np.random.poisson(60, 125),
    'Customer_Tenure_Months': np.random.normal(24, 10, 125),
    'Return_Rate': np.random.uniform(0.03, 0.09, 125)
}

# Combine segments
df_list = [pd.DataFrame(seg) for seg in [segment1, segment2, segment3, segment4]]
df = pd.concat(df_list, ignore_index=True)
df['Customer_ID'] = [f'CUST{str(i).zfill(4)}' for i in range(1, len(df)+1)]

# Calculate derived features
df['Customer_Lifetime_Value'] = (df['Num_Purchases'] * df['Avg_Transaction'] * 
                                  df['Customer_Tenure_Months'] / 12).round(2)
df['Purchase_Frequency'] = (df['Num_Purchases'] / df['Customer_Tenure_Months'] * 30).round(2)
df['Recency_Score'] = (100 - df['Days_Since_Last_Purchase']).clip(0, 100)

# Reorder columns
cols = ['Customer_ID', 'Age', 'Annual_Income', 'Spending_Score', 'Num_Purchases', 
        'Avg_Transaction', 'Days_Since_Last_Purchase', 'Customer_Tenure_Months',
        'Return_Rate', 'Customer_Lifetime_Value', 'Purchase_Frequency', 'Recency_Score']
df = df[cols].round(2)

print(f"✓ Generated {len(df)} customer records with {len(df.columns)-1} features")
print(f"✓ Date range: Last {df['Customer_Tenure_Months'].max():.0f} months")
print()

# ==================== FEATURE ENGINEERING ====================
print("🔧 FEATURE ENGINEERING & PREPROCESSING...")
print("-"*100)

features = ['Annual_Income', 'Spending_Score', 'Age', 'Num_Purchases', 
            'Avg_Transaction', 'Customer_Lifetime_Value', 'Purchase_Frequency',
            'Days_Since_Last_Purchase', 'Recency_Score']
X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"✓ Selected {len(features)} features for clustering")
print(f"✓ Applied StandardScaler normalization")
print()

# ==================== OPTIMAL CLUSTER DETERMINATION ====================
print("🎯 DETERMINING OPTIMAL NUMBER OF CLUSTERS...")
print("-"*100)

metrics = {
    'k': [],
    'inertia': [],
    'silhouette': [],
    'davies_bouldin': [],
    'calinski_harabasz': []
}

K_range = range(2, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    metrics['k'].append(k)
    metrics['inertia'].append(kmeans.inertia_)
    metrics['silhouette'].append(silhouette_score(X_scaled, labels))
    metrics['davies_bouldin'].append(davies_bouldin_score(X_scaled, labels))
    metrics['calinski_harabasz'].append(calinski_harabasz_score(X_scaled, labels))

metrics_df = pd.DataFrame(metrics)

# Determine optimal k (highest silhouette score)
optimal_k = metrics_df.loc[metrics_df['silhouette'].idxmax(), 'k']
print(f"✓ Optimal K determined: {int(optimal_k)} clusters")
print(f"✓ Best Silhouette Score: {metrics_df['silhouette'].max():.3f}")
print()

# Visualization of metrics
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Cluster Optimization Metrics', fontsize=16, fontweight='bold')

axes[0, 0].plot(metrics_df['k'], metrics_df['inertia'], marker='o', linewidth=2, markersize=8)
axes[0, 0].set_xlabel('Number of Clusters (K)', fontsize=11)
axes[0, 0].set_ylabel('Inertia', fontsize=11)
axes[0, 0].set_title('Elbow Method', fontsize=12, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].axvline(optimal_k, color='red', linestyle='--', alpha=0.7, label=f'Optimal K={int(optimal_k)}')
axes[0, 0].legend()

axes[0, 1].plot(metrics_df['k'], metrics_df['silhouette'], marker='o', linewidth=2, markersize=8, color='green')
axes[0, 1].set_xlabel('Number of Clusters (K)', fontsize=11)
axes[0, 1].set_ylabel('Silhouette Score', fontsize=11)
axes[0, 1].set_title('Silhouette Analysis', fontsize=12, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].axvline(optimal_k, color='red', linestyle='--', alpha=0.7, label=f'Optimal K={int(optimal_k)}')
axes[0, 1].legend()

axes[1, 0].plot(metrics_df['k'], metrics_df['davies_bouldin'], marker='o', linewidth=2, markersize=8, color='orange')
axes[1, 0].set_xlabel('Number of Clusters (K)', fontsize=11)
axes[1, 0].set_ylabel('Davies-Bouldin Index', fontsize=11)
axes[1, 0].set_title('Davies-Bouldin Index (Lower is Better)', fontsize=12, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(metrics_df['k'], metrics_df['calinski_harabasz'], marker='o', linewidth=2, markersize=8, color='purple')
axes[1, 1].set_xlabel('Number of Clusters (K)', fontsize=11)
axes[1, 1].set_ylabel('Calinski-Harabasz Score', fontsize=11)
axes[1, 1].set_title('Calinski-Harabasz Score (Higher is Better)', fontsize=12, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ==================== CLUSTERING WITH MULTIPLE ALGORITHMS ====================
print("🤖 COMPARING CLUSTERING ALGORITHMS...")
print("-"*100)

# KMeans
kmeans = KMeans(n_clusters=int(optimal_k), random_state=42, n_init=10)
df['Cluster_KMeans'] = kmeans.fit_predict(X_scaled)

# Hierarchical Clustering
hierarchical = AgglomerativeClustering(n_clusters=int(optimal_k))
df['Cluster_Hierarchical'] = hierarchical.fit_predict(X_scaled)

# Gaussian Mixture
gmm = GaussianMixture(n_components=int(optimal_k), random_state=42)
df['Cluster_GMM'] = gmm.fit_predict(X_scaled)

# Compare algorithms
algo_scores = {
    'Algorithm': ['K-Means', 'Hierarchical', 'Gaussian Mixture'],
    'Silhouette Score': [
        silhouette_score(X_scaled, df['Cluster_KMeans']),
        silhouette_score(X_scaled, df['Cluster_Hierarchical']),
        silhouette_score(X_scaled, df['Cluster_GMM'])
    ]
}
algo_comparison = pd.DataFrame(algo_scores)
print(algo_comparison.to_string(index=False))
print()

# Use best performing algorithm
best_algo = algo_comparison.loc[algo_comparison['Silhouette Score'].idxmax(), 'Algorithm']
print(f"✓ Best performing algorithm: {best_algo}")

if 'K-Means' in best_algo:
    df['Cluster'] = df['Cluster_KMeans']
elif 'Hierarchical' in best_algo:
    df['Cluster'] = df['Cluster_Hierarchical']
else:
    df['Cluster'] = df['Cluster_GMM']

print()

# ==================== CLUSTER PROFILING ====================
print("📈 DETAILED CLUSTER PROFILING...")
print("-"*100)

cluster_profiles = df.groupby('Cluster').agg({
    'Age': ['mean', 'std'],
    'Annual_Income': ['mean', 'std'],
    'Spending_Score': ['mean', 'std'],
    'Num_Purchases': ['mean', 'std'],
    'Avg_Transaction': ['mean', 'std'],
    'Customer_Lifetime_Value': ['mean', 'std'],
    'Purchase_Frequency': ['mean', 'std'],
    'Days_Since_Last_Purchase': ['mean', 'std'],
    'Customer_ID': 'count'
}).round(2)

cluster_profiles.columns = ['_'.join(col).strip() for col in cluster_profiles.columns.values]
cluster_profiles.rename(columns={'Customer_ID_count': 'Customer_Count'}, inplace=True)

print(cluster_profiles)
print()

# Statistical significance testing (ANOVA)
print("📊 STATISTICAL SIGNIFICANCE TESTING (ANOVA)...")
print("-"*100)

for feature in ['Annual_Income', 'Spending_Score', 'Customer_Lifetime_Value']:
    groups = [df[df['Cluster'] == i][feature].values for i in df['Cluster'].unique()]
    f_stat, p_value = f_oneway(*groups)
    significance = "✓ Significant" if p_value < 0.05 else "✗ Not Significant"
    print(f"{feature:30s} | F-statistic: {f_stat:8.2f} | p-value: {p_value:.4f} | {significance}")

print()

# ==================== CLUSTER PERSONAS ====================
print("👥 CUSTOMER SEGMENT PERSONAS...")
print("-"*100)

personas = {
    0: {
        'name': 'Premium Loyalists',
        'icon': '💎',
        'description': 'High-value customers with strong purchase frequency and large transaction sizes',
        'strategy': 'VIP treatment, exclusive offers, loyalty rewards program'
    },
    1: {
        'name': 'Potential Champions',
        'icon': '🌟',
        'description': 'Growing customers with good engagement and moderate spending',
        'strategy': 'Personalized recommendations, upselling opportunities, engagement campaigns'
    },
    2: {
        'name': 'Budget Seekers',
        'icon': '🎯',
        'description': 'Price-sensitive customers looking for value and deals',
        'strategy': 'Discount promotions, bundle offers, clearance sales communications'
    },
    3: {
        'name': 'At-Risk Customers',
        'icon': '⚠️',
        'description': 'Low engagement with infrequent purchases and high recency',
        'strategy': 'Win-back campaigns, special incentives, survey feedback collection'
    }
}

for cluster_id in sorted(df['Cluster'].unique()):
    if cluster_id in personas:
        persona = personas[cluster_id]
        size = len(df[df['Cluster'] == cluster_id])
        pct = (size / len(df)) * 100
        avg_clv = df[df['Cluster'] == cluster_id]['Customer_Lifetime_Value'].mean()
        
        print(f"\n{persona['icon']} CLUSTER {cluster_id}: {persona['name'].upper()}")
        print(f"   Size: {size} customers ({pct:.1f}%)")
        print(f"   Avg CLV: ${avg_clv:,.2f}")
        print(f"   Profile: {persona['description']}")
        print(f"   Strategy: {persona['strategy']}")

print()

# ==================== ADVANCED VISUALIZATIONS ====================
print("🎨 GENERATING ADVANCED VISUALIZATIONS...")
print("-"*100)

# 1. PCA 3D Visualization
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

colors = plt.cm.Set2(np.linspace(0, 1, int(optimal_k)))
for i, cluster_id in enumerate(sorted(df['Cluster'].unique())):
    cluster_data = X_pca[df['Cluster'] == cluster_id]
    ax.scatter(cluster_data[:, 0], cluster_data[:, 1], cluster_data[:, 2],
               c=[colors[i]], label=f'Cluster {cluster_id}', s=50, alpha=0.6, edgecolors='w')

ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=11)
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=11)
ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%})', fontsize=11)
ax.set_title('3D PCA Cluster Visualization', fontsize=14, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.show()

# 2. Cluster Characteristics Heatmap
cluster_means = df.groupby('Cluster')[features].mean()
plt.figure(figsize=(12, 6))
sns.heatmap(cluster_means.T, annot=True, fmt='.2f', cmap='YlGnBu', cbar_kws={'label': 'Normalized Value'})
plt.title('Cluster Characteristics Heatmap', fontsize=14, fontweight='bold')
plt.xlabel('Cluster', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.tight_layout()
plt.show()

# 3. CLV Distribution by Cluster
plt.figure(figsize=(12, 6))
sns.violinplot(data=df, x='Cluster', y='Customer_Lifetime_Value', palette='Set2')
plt.title('Customer Lifetime Value Distribution by Cluster', fontsize=14, fontweight='bold')
plt.xlabel('Cluster', fontsize=12)
plt.ylabel('CLV ($)', fontsize=12)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# 4. Cluster Size and Revenue Contribution
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

cluster_sizes = df['Cluster'].value_counts().sort_index()
cluster_revenue = df.groupby('Cluster')['Customer_Lifetime_Value'].sum().sort_index()

ax1.pie(cluster_sizes, labels=[f'Cluster {i}' for i in cluster_sizes.index],
        autopct='%1.1f%%', startangle=90, colors=colors)
ax1.set_title('Customer Distribution', fontsize=12, fontweight='bold')

ax2.pie(cluster_revenue, labels=[f'Cluster {i}' for i in cluster_revenue.index],
        autopct='%1.1f%%', startangle=90, colors=colors)
ax2.set_title('Revenue Contribution', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()

# 5. Radar Chart for Cluster Comparison
from math import pi

categories = ['Income', 'Spending', 'Purchases', 'CLV', 'Frequency']
fig, axes = plt.subplots(2, 2, figsize=(14, 12), subplot_kw=dict(projection='polar'))
axes = axes.flatten()

for idx, cluster_id in enumerate(sorted(df['Cluster'].unique())):
    cluster_data = df[df['Cluster'] == cluster_id]
    
    values = [
        cluster_data['Annual_Income'].mean() / df['Annual_Income'].max(),
        cluster_data['Spending_Score'].mean() / 100,
        cluster_data['Num_Purchases'].mean() / df['Num_Purchases'].max(),
        cluster_data['Customer_Lifetime_Value'].mean() / df['Customer_Lifetime_Value'].max(),
        cluster_data['Purchase_Frequency'].mean() / df['Purchase_Frequency'].max()
    ]
    
    angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))]
    values += values[:1]
    angles += angles[:1]
    
    ax = axes[idx]
    ax.plot(angles, values, 'o-', linewidth=2, color=colors[idx])
    ax.fill(angles, values, alpha=0.25, color=colors[idx])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)
    ax.set_title(f'Cluster {cluster_id} Profile', fontsize=12, fontweight='bold', pad=20)
    ax.grid(True)

plt.tight_layout()
plt.show()

print("✓ All visualizations generated successfully!")
print()

# ==================== BUSINESS RECOMMENDATIONS ====================
print("💼 ACTIONABLE BUSINESS RECOMMENDATIONS...")
print("="*100)

total_revenue = df['Customer_Lifetime_Value'].sum()

for cluster_id in sorted(df['Cluster'].unique()):
    cluster_data = df[df['Cluster'] == cluster_id]
    cluster_revenue = cluster_data['Customer_Lifetime_Value'].sum()
    revenue_pct = (cluster_revenue / total_revenue) * 100
    
    if cluster_id in personas:
        persona = personas[cluster_id]
        print(f"\n{persona['icon']} {persona['name'].upper()} (Cluster {cluster_id})")
        print(f"   Revenue Impact: ${cluster_revenue:,.2f} ({revenue_pct:.1f}% of total)")
        print(f"   Action Items:")
        print(f"   • {persona['strategy']}")
        
        # Specific recommendations
        avg_recency = cluster_data['Days_Since_Last_Purchase'].mean()
        if avg_recency > 45:
            print(f"   • URGENT: Average {avg_recency:.0f} days since last purchase - activate retention campaign")
        
        avg_return_rate = cluster_data['Return_Rate'].mean()
        if avg_return_rate > 0.08:
            print(f"   • HIGH RETURN RATE ({avg_return_rate:.1%}) - investigate product quality/fit issues")

print()
print("="*100)
print("ANALYSIS COMPLETE - Export df to CSV for further use: df.to_csv('customer_segments.csv')")
print("="*100)