# PCA vs t-SNE: Dimensionality Reduction Techniques Comparison

This project demonstrates and compares two popular dimensionality reduction techniques: **PCA (Principal Component Analysis)** and **t-SNE (t-Distributed Stochastic Neighbor Embedding)**.

## 📊 Objective

The objective of this project is to visualize and compare how different dimensionality reduction techniques represent a synthetic classification dataset, reducing from 6 dimensions to 2D for easier visualization.

## 🛠️ Implemented Techniques

### 1. PCA (Principal Component Analysis)
- **Linear method** for dimensionality reduction
- Preserves **maximum variance** of the data
- Fast and deterministic
- Ideal for data with linear relationships

### 2. t-SNE (t-Distributed Stochastic Neighbor Embedding)
- **Non-linear method** for dimensionality reduction
- Preserves **local structure** of the data
- Excellent for cluster visualization
- Slower, but reveals complex patterns

## 📁 Project Structure

```
tnse_machine_learning/
├── pca_vs_tnse.py          # Main code with comparison
├── pca.py                  # Basic 3D visualization code
└── README.md               # This file
```

## 📦 Dependencies

```bash
pip install plotly scikit-learn
```

## 🚀 How to Run

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd tnse_machine_learning
   ```

2. **Install dependencies:**
   ```bash
   pip install plotly scikit-learn
   ```

3. **Run the main file:**
   ```bash
   python pca_vs_tnse.py
   ```

## 📈 What the Code Does

### 1. Synthetic Data Generation
- Creates a dataset with **1500 samples**
- **6 features** (3 informative, 3 redundant)
- **3 different classes**
- Uses `make_classification` from scikit-learn

### 2. Generated Visualizations

#### Original 3D Visualization
- Shows the first 3 dimensions of the original data
- Allows understanding the basic distribution of classes

#### PCA (2D)
- Reduces from 6D to 2D preserving maximum variance
- Principal components are linear combinations of original features
- Good global class separation

#### t-SNE (2D)
- Reduces from 6D to 2D preserving local neighborhoods
- Reveals more detailed cluster structures
- Better for identifying sub-groups within classes

## 🔍 Main Observed Differences

|        Aspect        |       PCA         |            t-SNE 
|----------------------|-------------------|------------------------------|
| **Type**             | Linear            | Non-linear                   |
| **Speed**            | Fast              | Slower                       |
| **Preserves**        | Global variance   | Local structure              |
| **Deterministic**    | Yes               | No (depends on random_state) |
| **Interpretability** | High              | Low                          |
| **Visualization**    | Good for overview | Excellent for clusters       |

## 📊 Parameters Used

### Synthetic Dataset
```python
make_classification(
    n_features=6,           # 6 features
    n_classes=3,            # 3 classes
    n_samples=1500,         # 1500 samples
    n_informative=3,        # 3 informative features
    random_state=5,         # Reproducibility
    n_clusters_per_class=1  # 1 cluster per class
)
```

### PCA
```python
PCA(n_components=2)         # Reduction to 2 components
```

### t-SNE
```python
TSNE(
    n_components=2,         # Reduction to 2 components
    random_state=42         # Reproducibility
)
```

## 🎯 Conclusions

- **PCA** is ideal for quick data overview and when interpretability is important
- **t-SNE** is superior for discovering complex structures and visualizing detailed clusters
- Both techniques are complementary and useful in different contexts
- The choice depends on the objective: exploratory analysis (PCA) vs. detailed visualization (t-SNE)

## 🔗 Additional Resources

- [PCA Documentation - Scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
- [t-SNE Documentation - Scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)
- [Plotly Express Documentation](https://plotly.com/python/plotly-express/)