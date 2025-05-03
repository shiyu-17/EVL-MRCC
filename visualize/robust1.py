import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch
import os

def load_json_data(file_path):
    """Load data from a JSON file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        # For testing purposes, return the sample data if file not found
        return sample_data
    except json.JSONDecodeError:
        print(f"Error decoding JSON in: {file_path}")
        return None

# Define a sample data structure based on the provided JSON
sample_data = {
  "original": {
    "recall_at_k": {"1": 0.975},
    "precision_at_k": {"1": 0.975},
    "mrr": 0.9866666666666667,
    "ndcg": 0.968426083752166
  },
  "brightness_low": {
    "recall_at_k": {"1": 0.975},
    "precision_at_k": {"1": 0.975},
    "mrr": 0.9866666666666667,
    "ndcg": 0.968426083752166
  }
}

def create_comparison_visualizations(file_paths, dataset_names):
    """Create comparison visualizations for the given datasets."""
    
    # Define transformations to display
    transformations = ['original', 'brightness_low', 'brightness_high', 'contrast_low', 
                      'contrast_high', 'gaussian_blur', 'random_crop', 'noise', 'rotate_small']
    
    # Readable transformation names for plot
    transformation_labels = ['Original', 'Low\nBrightness', 'High\nBrightness', 'Low\nContrast',
                           'High\nContrast', 'Gaussian\nBlur', 'Random\nCrop', 'Noise', 'Small\nRotation']
    
    # Metrics to compare
    metrics = ['recall@1', 'precision@1', 'mrr', 'ndcg']
    metric_titles = ['Recall@1', 'Precision@1', 'MRR', 'NDCG']
    
    # Set up a custom color palette for the datasets
    colors = list(sns.color_palette('muted', len(dataset_names)))
    
    # Load all datasets
    all_data = []
    for file_path in file_paths:
        data = load_json_data(file_path)
        if data:
            all_data.append(data)
        else:
            print(f"Skipping visualization due to data loading issues with {file_path}")
            return
    
    # Set up the figure for the metrics
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # Set the style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Width of bars and positions
    bar_width = 0.15
    
    for metric_idx, (metric, title) in enumerate(zip(metrics, metric_titles)):
        ax = axes[metric_idx]
        
        for dataset_idx, (data, dataset_name) in enumerate(zip(all_data, dataset_names)):
            values = []
            
            for transform in transformations:
                # Handle recall@1 and precision@1 differently as they are nested
                if metric == 'recall@1':
                    value = data[transform]['recall_at_k']['1']
                elif metric == 'precision@1':
                    value = data[transform]['precision_at_k']['1']
                else:
                    value = data[transform][metric]
                values.append(value)
            
            # Calculate bar positions
            x_positions = np.arange(len(transformations)) + dataset_idx * bar_width
            
            # Plot bars
            ax.bar(x_positions, values, bar_width, label=dataset_name, 
                  color=colors[dataset_idx], alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Set x-tick positions to the center of each group
        ax.set_xticks(np.arange(len(transformations)) + bar_width * (len(dataset_names) - 1) / 2)
        ax.set_xticklabels(transformation_labels, rotation=45, ha='right')
        
        # Title and labels
        ax.set_title(title, fontsize=15, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12)
        
        # Set y-axis to start from a sensible minimum
        all_values = []
        for data in all_data:
            for transform in transformations:
                if metric == 'recall@1':
                    value = data[transform]['recall_at_k']['1']
                elif metric == 'precision@1':
                    value = data[transform]['precision_at_k']['1']
                else:
                    value = data[transform][metric]
                all_values.append(value)
        
        y_min = min(all_values)
        # Set y-axis limit to start slightly below minimum value for better perspective
        # but not less than 0.7 to prevent extreme scaling
        ax.set_ylim(max(0.7, y_min - 0.05), 1.01)  
        
        # Grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add value annotations on top of each bar
        for dataset_idx, data in enumerate(all_data):
            for i, transform in enumerate(transformations):
                if metric == 'recall@1':
                    value = data[transform]['recall_at_k']['1']
                elif metric == 'precision@1':
                    value = data[transform]['precision_at_k']['1']
                else:
                    value = data[transform][metric]
                
                x_pos = i + dataset_idx * bar_width
                ax.text(x_pos, value + 0.01, f'{value:.3f}', 
                       ha='center', va='bottom', fontsize=7, rotation=90, 
                       color='black')
    
    # Create a custom legend
    legend_elements = [Patch(facecolor=colors[i], edgecolor='black', label=name) 
                      for i, name in enumerate(dataset_names)]
    
    # Add the legend outside the plots
    fig.legend(handles=legend_elements, loc='upper center', 
              bbox_to_anchor=(0.5, 0.05), ncol=len(dataset_names), fontsize=12)
    
    plt.suptitle('Performance Metrics Comparison Across Different Image Transformations', 
                fontsize=18, fontweight='bold', y=0.98)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.07, 1, 0.96])
    
    # Create output directory if it doesn't exist
    os.makedirs('output', exist_ok=True)
    
    # Save the figure
    plt.savefig('output/metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('output/metrics_comparison.pdf', bbox_inches='tight')
    
    print("Visualization saved as 'output/metrics_comparison.png' and 'output/metrics_comparison.pdf'")
    
    plt.show()

if __name__ == "__main__":
    # Example usage:
    file_paths = [
        'data/ARKitScenes_cross_time.json',  # Replace with your actual file paths
        'data/7Scenes_cross_time.json',
        'data/TUM_cross_time.json', 
        'data/Cambridge_cross_time.json'
    ]
    
    # Try to test with just the provided sample first
    current_dir = os.path.dirname(os.path.abspath(__file__))
    test_file = os.path.join(current_dir, 'paste.txt')
    
    if os.path.exists(test_file):
        print(f"Found test file: {test_file}")
        file_paths = [test_file] * 4  # Use the test file for all datasets
    
    dataset_names = [
        'ARKitScenes',  # Replace with your preferred dataset names
        '7Scenes', 
        'TUM', 
        'Cambridge'
    ]
    
    create_comparison_visualizations(file_paths, dataset_names)