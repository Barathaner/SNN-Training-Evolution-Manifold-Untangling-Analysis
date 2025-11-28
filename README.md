# SNN-Training-Evolution-Manifold-Untangling-Analysis
The learning process in Spiking Neural Networks can be geometrically explained as the untangling of the layer manifolds over epochs into linearly separable classes. This library provides a pipeline and an example to analyse your spiking data from the input and your snntorch model. It is based on Manifold Capacity, and dimensionalityreduction (UMAP)




## Setup

Follow these steps to set up the project:

```bash
# Clone the repository
git clone https://github.com/Barathaner/SNN-Training-Evolution-Manifold-Untangling-Analysis.git-url
cd SNN-Training-Evolution-Manifold-Untangling-Analysis

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install the package in editable mode
pip install -e .
```

## Example Usage
In my masters thesis I used this pipeline to run on Spiking Heidelberg Dataset and on a simple feed forward Spiking Neural Network as support that this thesis could be true.
In the folder /examples you find the code of my masters thesis. you or your AI IDE like Cursor (I used cursor) can read it to explain how it works. I also made a Dash Webapp to get a nice idea of the visualisation of the manifolds.

cd examples
runs pipeline complete
python3 shd_snn_ffn_complete/main.py

python3 shd_snn_ffn_complete/webappvisualisation/app.py

To use this in your own code, put your model inside the models folder and integrate the SpikingActivityrecorder at the point where you want. It will then record samples from training from every epoch and save it as a run in the data/activity_logs folder
.... example code on how to integrate it in the neural net....

also use tonic to download your dataset in the data/input folder.

## Results - Explainable AI (XAI for SNN)
- picture of manifold properties over epochs pepr layer and input data as dotted line
- pictures of the umap visualisations
- pictures of the pca explained variance graph
- pictures of the raster plots as examples
- persistence diagramms of each layer
- wasserstein distance over epochs for each layer
- wasserstein distance over layer for each epoch
- output of estimated intrinsic dimension
- estimated betti numbers



### Future Ideas could include:
Feel free to open issues and to program it on your own and then make a pull request. I am happy if somebody wants to hel
- Persistent Homology of each Layer and comparison with Wasserstein Distance / Bottleneck Distance
- Silhoutte Score to examine the cluster quality
- KNN on Labels and then define cluster properties
- Training an autoencoder for latent space
- Persisten Laplacian
- Curvature Analysis for dimension estimation