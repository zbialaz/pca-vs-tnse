import plotly.express as px
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Creating the synthetic dataset
X, y = make_classification(
    n_features=6,
    n_classes=3,
    n_samples=1500,
    n_informative=3,
    random_state=5,
    n_clusters_per_class=1
)

# 3D plot (Original 3D visualization)
fig = px.scatter_3d(x=X[:, 0], y=X[:, 1], z=X[:, 2], color=y, opacity=0.8)
fig.show()

# PCA (3D to 2D)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

fig = px.scatter(x=X_pca[:, 0], y=X_pca[:, 1], color=y)
fig.update_layout(
    title='PCA Visualization of Custom Classification dataset',
    xaxis_title='First Principal Component',
    yaxis_title='Second Principal Component'
)

# Applied PCA
fig.show()

# t-SNE (3D to 2D)
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)
tsne.kl_divergence_

fig = px.scatter(x=X_tsne[:, 0], y=X_tsne[:, 1], color=y)
fig.update_layout(
    title='t-SNE Visualization of Custom Classification dataset',
    xaxis_title = 'First t-SNE Component',
    yaxis_title = 'Second t-SNE Component'
)

# Applied t-SNE
fig.show()