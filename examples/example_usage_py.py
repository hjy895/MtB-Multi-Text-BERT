"""
Basic Usage Example for Clinical Topic Modeling Framework

This example demonstrates how to use the framework for clinical topic modeling
and classification tasks with your own dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from framework import ClinicalTopicModelingFramework


def create_sample_data():
    """Create sample clinical data for demonstration."""
    np.random.seed(42)
    
    n_patients = 1000
    
    # Generate synthetic clinical features
    data = {
        'patient_id': range(1, n_patients + 1),
        'creatinine_max': np.random.normal(1.0, 0.5, n_patients),
        'gfr_min': np.random.normal(80, 20, n_patients),
        'potassium_max': np.random.normal(4.0, 0.8, n_patients),
        'hemoglobin_min': np.random.normal(12, 2, n_patients),
        'platelet_min': np.random.normal(200, 50, n_patients),
        'sysbp_min': np.random.normal(110, 20, n_patients),
        'spo2_min': np.random.normal(96, 3, n_patients),
        'urineoutput': np.random.normal(1500, 500, n_patients)
    }
    
    # Ensure positive values for certain features
    data['creatinine_max'] = np.abs(data['creatinine_max'])
    data['gfr_min'] = np.abs(data['gfr_min'])
    data['potassium_max'] = np.abs(data['potassium_max'])
    data['hemoglobin_min'] = np.abs(data['hemoglobin_min'])
    data['platelet_min'] = np.abs(data['platelet_min'])
    data['sysbp_min'] = np.abs(data['sysbp_min'])
    data['spo2_min'] = np.clip(data['spo2_min'], 70, 100)
    data['urineoutput'] = np.abs(data['urineoutput'])
    
    df = pd.DataFrame(data)
    
    # Create target labels based on clinical criteria
    # Simple rule: high risk if multiple abnormal values
    conditions = [
        df['creatinine_max'] > 1.2,
        df['gfr_min'] < 60,
        df['potassium_max'] > 5.0,
        df['hemoglobin_min'] < 10,
        df['platelet_min'] < 150,
        df['sysbp_min'] < 90,
        df['spo2_min'] < 94,
        df['urineoutput'] < 500
    ]
    
    # Count abnormal conditions
    abnormal_count = sum(conditions)
    
    # Binary label: 1 if >= 2 abnormal conditions, 0 otherwise
    df['aki_label'] = (abnormal_count >= 2).astype(int)
    
    # 3-point label: 0=low risk, 1=medium risk, 2=high risk
    df['risk_level'] = 0
    df.loc[abnormal_count >= 2, 'risk_level'] = 1
    df.loc[abnormal_count >= 4, 'risk_level'] = 2
    
    return df


def basic_usage_example():
    """Demonstrate basic usage of the framework."""
    print("=== Clinical Topic Modeling Framework - Basic Usage Example ===\n")
    
    # Step 1: Create sample data
    print("1. Creating sample clinical data...")
    data = create_sample_data()
    print(f"   Created dataset with {len(data)} patients")
    print(f"   Features: {list(data.columns)}")
    print(f"   Target distribution: {data['aki_label'].value_counts().to_dict()}")
    print()
    
    # Step 2: Initialize framework
    print("2. Initializing framework...")
    framework = ClinicalTopicModelingFramework()
    
    # Configure for our sample data
    framework.config['clinical_topics']['feature_columns'] = [
        'creatinine_max', 'gfr_min', 'potassium_max', 'hemoglobin_min',
        'platelet_min', 'sysbp_min', 'spo2_min', 'urineoutput'
    ]
    
    framework.config['clinical_topics']['topic_definitions'] = {
        'elevated_creatinine': {'column': 'creatinine_max', 'operator': '>', 'threshold': 1.2},
        'reduced_gfr': {'column': 'gfr_min', 'operator': '<', 'threshold': 60},
        'hyperkalemia': {'column': 'potassium_max', 'operator': '>', 'threshold': 5.0},
        'anemia': {'column': 'hemoglobin_min', 'operator': '<', 'threshold': 10},
        'thrombocytopenia': {'column': 'platelet_min', 'operator': '<', 'threshold': 150},
        'hypotension': {'column': 'sysbp_min', 'operator': '<', 'threshold': 90},
        'hypoxemia': {'column': 'spo2_min', 'operator': '<', 'threshold': 94},
        'oliguria': {'column': 'urineoutput', 'operator': '<', 'threshold': 500}
    }
    
    framework.config['data']['target_column'] = 'aki_label'
    print("   Framework initialized and configured")
    print()
    
    # Step 3: Load and preprocess data
    print("3. Loading and preprocessing data...")
    framework.load_data(data=data)
    print("   Data loaded and preprocessed")
    print()
    
    # Step 4: Extract clinical topics
    print("4. Extracting clinical topics...")
    clinical_texts = framework.extract_clinical_topics()
    
    # Show some examples
    print("   Sample clinical text representations:")
    for i in range(5):
        print(f"   Patient {i+1}: {clinical_texts.iloc[i]}")
    
    # Show topic statistics
    topic_stats = framework.topic_extractor.get_topic_statistics(clinical_texts)
    print(f"\n   Topic Statistics:")
    print(f"   - Patients with topics: {topic_stats['patients_with_topics']}")
    print(f"   - Average topics per patient: {topic_stats['average_topics_per_patient']:.2f}")
    print(f"   - Most common topics: {list(topic_stats['topic_frequencies'].keys())[:5]}")
    print()
    
    # Step 5: Create models (using a subset for faster demo)
    print("5. Creating models...")
    
    # Create a smaller set of models for demonstration
    from src.models.classical_models import create_classical_models
    classical_models = create_classical_models()
    
    # Select a few representative models
    demo_models = {
        'LogisticRegression': classical_models['LogisticRegression'],
        'RandomForestClassifier': classical_models['RandomForestClassifier'],
        'SVC': classical_models['SVC'],
        'MultinomialNB': classical_models['MultinomialNB']
    }
    
    # Add one BERT model
    from src.models.bert_classifier import BERTTopicClassifier
    demo_models['BERT-LR'] = BERTTopicClassifier(
        classifier_type='logistic_regression',
        max_length=64  # Smaller for demo
    )
    
    framework.models = demo_models
    print(f"   Created {len(demo_models)} models for evaluation")
    print()
    
    # Step 6: Run evaluation
    print("6. Running evaluation...")
    print("   This may take a few minutes...")
    
    results = framework.run_binary_classification()
    
    print("   Evaluation completed!")
    print()
    
    # Step 7: Display results
    print("7. Results Summary:")
    
    best_model = framework.get_best_model()
    if best_model:
        print(f"   Best Model: {best_model['model']}")
        print(f"   Extraction Method: {best_model['extraction_type']}")
        print(f"   Extraction Size: {best_model['extraction_size']}")
        print(f"   Score: {best_model['score']:.4f}")
    
    # Show performance summary
    performance_summary = framework.get_model_performance_summary()
    if not performance_summary.empty:
        print("\n   Top 5 Model Performances:")
        top_5 = performance_summary.head()
        for _, row in top_5.iterrows():
            print(f"   - {row['model']}: {row['best_score']:.4f}")
    
    print()
    
    # Step 8: Generate and save report
    print("8. Generating report...")
    
    # Create results directory
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    # Save results
    framework.save_results(str(results_dir))
    
    # Generate plots
    try:
        framework.plot_results(str(results_dir), show_plots=False)
        print("   Plots saved to results directory")
    except Exception as e:
        print(f"   Warning: Could not generate plots: {e}")
    
    # Export to Excel
    try:
        framework.export_results_to_excel(str(results_dir / 'evaluation_results.xlsx'))
        print("   Results exported to Excel")
    except Exception as e:
        print(f"   Warning: Could not export to Excel: {e}")
    
    print(f"   All results saved to: {results_dir.absolute()}")
    print()
    
    print("=== Basic Usage Example Completed ===")
    return framework, results


def custom_model_example():
    """Demonstrate how to add custom models to the framework."""
    print("\n=== Custom Model Example ===\n")
    
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.neural_network import MLPClassifier
    
    # Create framework
    framework = ClinicalTopicModelingFramework()
    
    # Add custom models
    framework.add_custom_model(
        'CustomGradientBoosting',
        GradientBoostingClassifier(n_estimators=50, random_state=42)
    )
    
    framework.add_custom_model(
        'CustomMLP',
        MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=300, random_state=42)
    )
    
    print("Custom models added to framework:")
    for name in framework.models.keys():
        print(f"  - {name}")
    
    print("\nCustom models can now be used in evaluation pipeline.")


def custom_topics_example():
    """Demonstrate how to add custom topic definitions."""
    print("\n=== Custom Topics Example ===\n")
    
    framework = ClinicalTopicModelingFramework()
    
    # Add custom topic definitions
    framework.add_custom_topic_definition(
        'severe_anemia', 'hemoglobin_min', '<', 8.0
    )
    
    framework.add_custom_topic_definition(
        'extreme_hypotension', 'sysbp_min', '<', 70
    )
    
    framework.add_custom_topic_definition(
        'severe_kidney_dysfunction', 'gfr_min', '<', 30
    )
    
    print("Custom topic definitions added:")
    for topic_name, definition in framework.topic_extractor.topic_definitions.items():
        print(f"  - {topic_name}: {definition['column']} {definition['operator']} {definition['threshold']}")


if __name__ == "__main__":
    # Run basic usage example
    framework, results = basic_usage_example()
    
    # Run additional examples
    custom_model_example()
    custom_topics_example()
    
    print("\n" + "="*60)
    print("Example completed! Check the 'results' directory for outputs.")
    print("="*60)