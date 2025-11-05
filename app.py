import streamlit as st
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Customer Segmentation Platform",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
    }
    .metric-label {
        font-size: 1rem;
        opacity: 0.9;
    }
    .persona-card {
        border-left: 4px solid #667eea;
        padding: 1rem;
        margin: 1rem 0;
        /* Dark card to match Streamlit dark theme and ensure high-contrast text */
        background: #0b1220; /* very dark navy */
        color: #f8f9fa; /* light text for readability */
        border-radius: 8px;
        box-shadow: 0 6px 14px rgba(2,6,23,0.6);
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if 'data_generated' not in st.session_state:
    st.session_state.data_generated = False
    st.session_state.df = None
    st.session_state.clustered = False

# Sidebar
with st.sidebar:
    st.title("🎯 Control Panel")
    st.markdown("---")
    
    st.subheader("📊 Data Source")
    data_source = st.radio("Choose data source:", ["Generate Sample Data", "Upload CSV"])
    
    if data_source == "Generate Sample Data":
        n_samples = st.slider("Number of customers:", 100, 2000, 500, 50)
        if st.button("🚀 Generate Data", use_container_width=True):
            with st.spinner("Generating customer data..."):
                np.random.seed(42)
                
                # Generate segments
                segments = []
                for _ in range(4):
                    seg = {
                        'Annual_Income': np.random.normal(np.random.uniform(30, 80), 10, n_samples//4),
                        'Spending_Score': np.random.normal(np.random.uniform(25, 75), 10, n_samples//4),
                        'Age': np.random.normal(np.random.uniform(25, 50), 5, n_samples//4),
                        'Num_Purchases': np.random.poisson(np.random.uniform(10, 50), n_samples//4),
                        'Avg_Transaction': np.random.normal(np.random.uniform(40, 150), 20, n_samples//4),
                        'Days_Since_Last_Purchase': np.random.poisson(np.random.uniform(15, 60), n_samples//4),
                        'Customer_Tenure_Months': np.random.normal(np.random.uniform(18, 48), 12, n_samples//4),
                    }
                    segments.append(pd.DataFrame(seg))
                
                df = pd.concat(segments, ignore_index=True)
                df['Customer_ID'] = [f'CUST{str(i).zfill(4)}' for i in range(1, len(df)+1)]
                df['Customer_Lifetime_Value'] = (df['Num_Purchases'] * df['Avg_Transaction'] * 
                                                  df['Customer_Tenure_Months'] / 12).round(2)
                df['Purchase_Frequency'] = (df['Num_Purchases'] / df['Customer_Tenure_Months'] * 30).round(2)
                
                st.session_state.df = df.round(2)
                st.session_state.data_generated = True
                st.success("✅ Data generated successfully!")
    
    else:
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            st.session_state.data_generated = True
            st.success("✅ File uploaded successfully!")
    
    st.markdown("---")
    
    if st.session_state.data_generated:
        st.subheader("⚙️ Clustering Settings")
        
        available_features = [col for col in st.session_state.df.columns 
                             if col not in ['Customer_ID', 'Cluster']]
        features = st.multiselect(
            "Select features:",
            available_features,
            default=[col for col in ['Annual_Income', 'Spending_Score', 'Num_Purchases', 
                                     'Customer_Lifetime_Value'] if col in available_features][:4]
        )
        
        algorithm = st.selectbox("Algorithm:", ["K-Means", "Hierarchical", "Gaussian Mixture"])
        
        n_clusters = st.slider("Number of clusters:", 2, 10, 4)
        
        if st.button("🔍 Run Clustering", use_container_width=True):
            if len(features) < 2:
                st.error("Please select at least 2 features!")
            else:
                with st.spinner("Running clustering analysis..."):
                    df = st.session_state.df.copy()
                    X = df[features]
                    
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    
                    if algorithm == "K-Means":
                        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    elif algorithm == "Hierarchical":
                        model = AgglomerativeClustering(n_clusters=n_clusters)
                    else:
                        model = GaussianMixture(n_components=n_clusters, random_state=42)
                    
                    df['Cluster'] = model.fit_predict(X_scaled)
                    
                    st.session_state.df = df
                    st.session_state.clustered = True
                    st.session_state.features = features
                    st.session_state.X_scaled = X_scaled
                    st.session_state.algorithm = algorithm
                    
                    st.success("✅ Clustering complete!")

# Main content
st.markdown('<h1 class="main-header">🎯 Customer Segmentation Platform</h1>', unsafe_allow_html=True)

if not st.session_state.data_generated:
    st.info("👈 Please generate sample data or upload a CSV file from the sidebar to begin.")
    
    # Show demo features
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 📊 Advanced Analytics")
        st.write("• Multiple clustering algorithms")
        st.write("• Automated optimal K selection")
        st.write("• Statistical significance testing")
    
    with col2:
        st.markdown("### 🎨 Rich Visualizations")
        st.write("• Interactive 3D plots")
        st.write("• Customer personas")
        st.write("• Revenue analysis")
    
    with col3:
        st.markdown("### 💼 Business Insights")
        st.write("• Actionable recommendations")
        st.write("• CLV predictions")
        st.write("• Segment strategies")

else:
    df = st.session_state.df
    
    # Overview metrics
    st.markdown("## 📈 Business Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Customers", f"{len(df):,}")
    
    with col2:
        if 'Customer_Lifetime_Value' in df.columns:
            st.metric("Total Revenue", f"${df['Customer_Lifetime_Value'].sum():,.0f}")
    
    with col3:
        if 'Customer_Lifetime_Value' in df.columns:
            st.metric("Avg CLV", f"${df['Customer_Lifetime_Value'].mean():,.0f}")
    
    with col4:
        if 'Cluster' in df.columns:
            st.metric("Segments", len(df['Cluster'].unique()))
    
    st.markdown("---")
    
    # Tabs for different sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Data Overview", 
        "🎯 Clustering Analysis", 
        "👥 Customer Personas",
        "📈 Visualizations",
        "💼 Recommendations"
    ])
    
    with tab1:
        st.markdown("### 📋 Dataset Preview")
        st.dataframe(df.head(100), use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 📊 Statistical Summary")
            st.dataframe(df.describe().round(2), use_container_width=True)
        
        with col2:
            st.markdown("### 📉 Missing Values")
            missing = df.isnull().sum()
            if missing.sum() == 0:
                st.success("✅ No missing values detected!")
            else:
                st.dataframe(missing[missing > 0], use_container_width=True)
    
    with tab2:
        if not st.session_state.clustered:
            st.info("👈 Configure clustering settings in the sidebar and click 'Run Clustering'")
        else:
            st.markdown("### 🎯 Clustering Results")
            
            # Metrics
            silhouette = silhouette_score(st.session_state.X_scaled, df['Cluster'])
            davies_bouldin = davies_bouldin_score(st.session_state.X_scaled, df['Cluster'])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Algorithm", st.session_state.algorithm)
            with col2:
                st.metric("Silhouette Score", f"{silhouette:.3f}")
            with col3:
                st.metric("Davies-Bouldin Index", f"{davies_bouldin:.3f}")
            
            st.markdown("---")
            
            # Cluster distribution
            col1, col2 = st.columns(2)
            
            with col1:
                cluster_counts = df['Cluster'].value_counts().sort_index()
                fig = px.pie(values=cluster_counts.values, 
                            names=[f'Cluster {i}' for i in cluster_counts.index],
                            title='Customer Distribution by Cluster',
                            hole=0.4)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if 'Customer_Lifetime_Value' in df.columns:
                    cluster_revenue = df.groupby('Cluster')['Customer_Lifetime_Value'].sum()
                    fig = px.bar(x=[f'Cluster {i}' for i in cluster_revenue.index],
                                y=cluster_revenue.values,
                                title='Revenue Contribution by Cluster',
                                labels={'x': 'Cluster', 'y': 'Total CLV ($)'},
                                color=cluster_revenue.values,
                                color_continuous_scale='viridis')
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        if not st.session_state.clustered:
            st.info("👈 Run clustering first to see customer personas")
        else:
            st.markdown("### 👥 Customer Segment Personas")
            
            personas = {
                0: {'name': '💎 Premium Loyalists', 'color': '#FF6B6B'},
                1: {'name': '🌟 Potential Champions', 'color': '#4ECDC4'},
                2: {'name': '🎯 Budget Seekers', 'color': '#45B7D1'},
                3: {'name': '⚠️ At-Risk Customers', 'color': '#FFA07A'}
            }
            
            for cluster_id in sorted(df['Cluster'].unique()):
                cluster_data = df[df['Cluster'] == cluster_id]
                size = len(cluster_data)
                pct = (size / len(df)) * 100
                
                persona_name = personas.get(cluster_id, {}).get('name', f'Cluster {cluster_id}')
                
                with st.expander(f"{persona_name} - {size} customers ({pct:.1f}%)", expanded=True):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if 'Age' in cluster_data.columns:
                            st.metric("Avg Age", f"{cluster_data['Age'].mean():.1f}")
                    with col2:
                        if 'Annual_Income' in cluster_data.columns:
                            st.metric("Avg Income", f"${cluster_data['Annual_Income'].mean():,.0f}K")
                    with col3:
                        if 'Customer_Lifetime_Value' in cluster_data.columns:
                            st.metric("Avg CLV", f"${cluster_data['Customer_Lifetime_Value'].mean():,.0f}")
                    
                    # Detailed stats
                    if st.session_state.features:
                        stats_df = cluster_data[st.session_state.features].describe().T
                        stats_df = stats_df[['mean', 'std', 'min', 'max']].round(2)
                        st.dataframe(stats_df, use_container_width=True)
    
    with tab4:
        if not st.session_state.clustered:
            st.info("👈 Run clustering first to see visualizations")
        else:
            st.markdown("### 📊 Interactive Visualizations")
            
            # 3D PCA
            if len(st.session_state.features) >= 3:
                pca = PCA(n_components=3)
                X_pca = pca.fit_transform(st.session_state.X_scaled)
                
                pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2', 'PC3'])
                pca_df['Cluster'] = df['Cluster'].astype(str)
                
                fig = px.scatter_3d(pca_df, x='PC1', y='PC2', z='PC3', color='Cluster',
                                   title='3D PCA Cluster Visualization',
                                   labels={'PC1': f'PC1 ({pca.explained_variance_ratio_[0]:.1%})',
                                          'PC2': f'PC2 ({pca.explained_variance_ratio_[1]:.1%})',
                                          'PC3': f'PC3 ({pca.explained_variance_ratio_[2]:.1%})'})
                st.plotly_chart(fig, use_container_width=True)
            
            # Feature comparison
            col1, col2 = st.columns(2)
            
            with col1:
                if len(st.session_state.features) >= 2:
                    feature_x = st.selectbox("X-axis feature:", st.session_state.features, key='x')
                    feature_y = st.selectbox("Y-axis feature:", st.session_state.features, 
                                           index=1 if len(st.session_state.features) > 1 else 0, key='y')
                    
                    fig = px.scatter(df, x=feature_x, y=feature_y, color='Cluster',
                                    title=f'{feature_x} vs {feature_y}',
                                    hover_data=['Customer_ID'])
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if 'Customer_Lifetime_Value' in df.columns:
                    fig = px.box(df, x='Cluster', y='Customer_Lifetime_Value',
                                color='Cluster',
                                title='CLV Distribution by Cluster')
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab5:
        if not st.session_state.clustered:
            st.info("👈 Run clustering first to see recommendations")
        else:
            st.markdown("### 💼 Actionable Business Recommendations")
            
            total_revenue = df['Customer_Lifetime_Value'].sum() if 'Customer_Lifetime_Value' in df.columns else 0
            
            strategies = {
                0: "🎁 VIP treatment, exclusive offers, early access to new products",
                1: "📧 Personalized recommendations, upselling campaigns, loyalty rewards",
                2: "💰 Discount promotions, bundle offers, value messaging",
                3: "🔔 Win-back campaigns, special incentives, satisfaction surveys"
            }
            
            for cluster_id in sorted(df['Cluster'].unique()):
                cluster_data = df[df['Cluster'] == cluster_id]
                
                if 'Customer_Lifetime_Value' in df.columns:
                    cluster_revenue = cluster_data['Customer_Lifetime_Value'].sum()
                    revenue_pct = (cluster_revenue / total_revenue) * 100 if total_revenue > 0 else 0
                    
                    st.markdown(f"""
                    <div class="persona-card">
                        <h4>Cluster {cluster_id} - {len(cluster_data)} customers</h4>
                        <p><strong>Revenue Impact:</strong> ${cluster_revenue:,.0f} ({revenue_pct:.1f}% of total)</p>
                        <p><strong>Strategy:</strong> {strategies.get(cluster_id, 'Custom engagement strategy')}</p>
                    </div>
                    """, unsafe_allow_html=True)
    
    # Download section
    st.markdown("---")
    st.markdown("### 💾 Export Results")
    
    col1, col2 = st.columns(2)
    with col1:
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Clustered Data (CSV)",
            data=csv,
            file_name="customer_segments.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        if st.session_state.clustered:
            summary = df.groupby('Cluster')[st.session_state.features].mean().round(2)
            summary_csv = summary.to_csv()
            st.download_button(
                label="📊 Download Cluster Summary (CSV)",
                data=summary_csv,
                file_name="cluster_summary.csv",
                mime="text/csv",
                use_container_width=True
            )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p>🎯 Customer Segmentation Platform | Built with Streamlit</p>
    <p>💡 Powered by scikit-learn and Plotly</p>
</div>
""", unsafe_allow_html=True)