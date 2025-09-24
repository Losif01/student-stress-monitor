import streamlit as st
import pandas as pd
import plotly.express as px
from src.stress_monitor.data.data_loader import DataLoader
from src.stress_monitor.data.data_preprocessor import DataPreprocessor
from src.stress_monitor.models.model_factory import ModelFactory
from src.stress_monitor.visualization.plotter import StressDataVisualizer
from src.stress_monitor.config import ModelType, config
from src.stress_monitor.utils.helpers import print_results, format_percentage
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Student Stress Monitoring",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("🎓 Student Stress Monitoring Dashboard")
    st.markdown("Analyze and predict student stress levels using machine learning")
    
    # Sidebar
    st.sidebar.title("Configuration")
    data_path = st.sidebar.text_input("Data Directory", value="data")
    selected_models = st.sidebar.multiselect(
        "Select Models to Train",
        options=[model.value for model in ModelType],
        default=[model.value for model in ModelType]
    )
    
    try:
        # Load data
        data_loader = DataLoader(data_path)
        stress_data, _ = data_loader.load_all_datasets()
        
        # Data overview
        st.header("📈 Data Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Samples", stress_data.shape[0])
        with col2:
            st.metric("Number of Features", stress_data.shape[1] - 1)
        with col3:
            st.metric("Missing Values", stress_data.isnull().sum().sum())
        with col4:
            avg_stress = stress_data['stress_level'].mean()
            st.metric("Average Stress Level", f"{avg_stress:.2f}")
        
        # Data preview
        st.subheader("Data Preview")
        st.dataframe(stress_data.head())
        
        # Data information
        if st.checkbox("Show Data Information"):
            info_report = data_loader.get_dataset_info(stress_data)
            st.dataframe(info_report)
        
        # Visualizations
        st.header("📊 Visualizations")
        
        viz = StressDataVisualizer()
        
        # Histograms
        if st.checkbox("Show Feature Distributions"):
            hist_fig = viz.create_histograms(stress_data)
            st.plotly_chart(hist_fig, use_container_width=True)
        
        # Correlation heatmap
        if st.checkbox("Show Correlation Heatmap"):
            corr_fig = viz.create_correlation_heatmap(stress_data)
            st.plotly_chart(corr_fig, use_container_width=True)
        
        # Model training section
        st.header("🤖 Model Training")
        
        if st.button("Train Selected Models"):
            with st.spinner("Training models..."):
                # Prepare data
                preprocessor = DataPreprocessor()
                X, y = preprocessor.prepare_features_target(stress_data)
                X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
                
                results = {}
                progress_bar = st.progress(0)
                
                for i, model_type_str in enumerate(selected_models):
                    model_type = ModelType(model_type_str)
                    model = ModelFactory.create_model(model_type)
                    
                    # Update progress
                    progress = (i + 1) / len(selected_models)
                    progress_bar.progress(progress)
                    
                    # Train model
                    try:
                        model.fit(X_train, y_train)
                        result = model.get_result(X_test, y_test)
                        results[model.model_name] = result
                        
                        st.success(f"✅ {model.model_name} trained successfully")
                    except Exception as e:
                        st.error(f"❌ Error training {model.model_name}: {str(e)}")
                
                # Display results
                st.header("📊 Results")
                
                # Results table
                results_df = pd.DataFrame([
                    {
                        'Model': name,
                        'Accuracy': result.accuracy if hasattr(result, 'accuracy') else result,
                        'Accuracy (%)': format_percentage(result.accuracy if hasattr(result, 'accuracy') else result)
                    }
                    for name, result in results.items()
                ])
                
                st.dataframe(results_df.sort_values('Accuracy', ascending=False))
                
                # Results chart
                if results:
                    accuracies = {name: result.accuracy if hasattr(result, 'accuracy') else result 
                                for name, result in results.items()}
                    comparison_fig = viz.create_model_comparison_chart(accuracies)
                    st.plotly_chart(comparison_fig, use_container_width=True)
        
        # Model comparison
        st.sidebar.header("Model Comparison")
        if st.sidebar.checkbox("Show Model Details"):
            st.sidebar.markdown("""
            **Available Models:**
            - Logistic Regression
            - Gradient Boosting
            - Random Forest
            - Tweedie Regressor
            - SGD Classifier
            - CatBoost
            - MLP Classifier
            """)
    
    except FileNotFoundError as e:
        st.error(f"❌ Data file not found: {e}")
        st.info("💡 Please make sure your data files are in the correct directory")
    except Exception as e:
        st.error(f"❌ An error occurred: {e}")

if __name__ == "__main__":
    main()