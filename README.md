# A Theory for Conditional Generative Modeling on Multiple Data Sources

This is the official implementation for [A Theory for Conditional Generative Modeling on Multiple Data Sources]().

## Dependencies
* 64-bit Python 3.9 and PyTorch 2.1 (or later). See https://pytorch.org for PyTorch install instructions. 
    We remommand using Anaconda3 to create your environment `conda create -n [your_env_name] python=3.9`
* Install requirements `pip install -r requirements.txt`

## Simulations on conditional Gaussian estimation

We provide scripts to reproduce the simulation results from Figure 1 of our paper.

### Run experiments
Run the following commands to generate simulation results under `simulations/results/gaussian/`:

```python
python simulations/run_gaussian.py --tag=K  # Run experiment for K
python simulations/run_gaussian.py --tag=n  # Run experiment for n
python simulations/run_gaussian.py --tag=sim  # Run experiment for beta_sim
```

### Visualize results

Use the following commands to visualize the results:

```python
python simulations/plot_gaussian_K.py
python simulations/plot_gaussian_n.py
python simulations/plot_gaussian_sim.py
```


## Real-word experiments on conditional diffusion models

### Preparing datasets

Datasets are stored as uncompressed ZIP archives containing uncompressed PNG or NPY files, along with a metadata file `dataset.json` for labels. When using latent diffusion, it is necessary to create two different versions of a given dataset: the original RGB version, used for evaluation, and a VAE-encoded latent version, used for training.

To set up ImageNet-256:

1. Download the ILSVRC2012 data archive from [Kaggle](https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data) and extract it somewhere, e.g., `downloads/imagenet`.

2. Crop and resize the images to create the original RGB dataset:

```.bash
# Convert raw ImageNet data to a ZIP archive at 256x256 resolution, for example:
python dataset_tool.py convert --source=downloads/imagenet/train \
    --dest=datasets/img256.zip --resolution=256x256 --transform=center-crop-dhariwal
```

3. Run the images through a pre-trained VAE encoder to create the corresponding latent dataset:

```.bash
# Convert the pixel data to VAE latents, for example:
python dataset_tool.py encode --source=datasets/img256.zip \
    --dest=datasets/img256-sd.zip
```

4. Calculate reference statistics for the original RGB dataset, to be used with `calculate_metrics.py`:

```.bash
# Compute dataset reference statistics for calculating metrics, for example:
python calculate_metrics.py ref --data=datasets/img256.zip \
    --dest=dataset-refs/img256.pkl
```

For conveinence, we provide bash files in directory `bash_files/convert/`, 
and you can directly run the bash files like `bash bash_files/convert/convert_1c.sh sim3`

### Training new models

New models can be trained using `train_edm2.py`. For example, to train an XS-sized conditional model for ImageNet-256 using the same hyperparameters as in our paper, run:

```.bash
# Train XS-sized model for ImageNet-256 using 8 GPUs

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=25641 train_edm2.py \
      --outdir=training-runs/{similarity_name}-edm2-img256-xs-{classes_setup}-{sample_num} \
      --data=datasets/latent_datasets/{similarity_name}/inet-256-{classes_setup}-{sample_num}.zip \
      --preset=edm2-img256-xs-{classes_setup}-{sample_num} \
      --batch=$(( 1024 * 8 )) --status=64Ki --snapshot=16Mi --checkpoint=64Mi
```

As we have many setups in our paper, you can edit the parameters which in `{}`. 
We provide example scripts for training in directory `bash_files/train/`

This example performs single-node training using 8 GPUs.

By default, the training script prints status every 128Ki (= 128 [kibi](https://en.wikipedia.org/wiki/Binary_prefix#kibi) = 128&times;2<sup>10</sup>) training images (controlled by `--status`), saves network snapshots every 8Mi (= 8&times;2<sup>20</sup>) training images (controlled by `--snapshot`), and dumps training checkpoints every 128Mi training images (controlled by `--checkpoint`). The status is saved in `log.txt` (one-line summary) and `stats.json` (comprehensive set of statistics). The network snapshots are saved in `network-snapshot-*.pkl`, and they can be used directly with, e.g., `generate_images.py`.

The training checkpoints, saved in `training-state-*.pt`, can be used to resume the training at a later time.
When the training script starts, it will automatically look for the highest-numbered checkpoint and load it if available. 
To resume training, simply run the same `train_edm2.py` command line again &mdash; it is important to use the same set of options to avoid accidentally changing the hyperparameters mid-training.

## Acknowledgments

The code is developed based on the [EDM2](https://github.com/NVlabs/edm2.git) repository. We appreciate their nice implementations.


## License

This code is released under the MIT License. See  [LICENSE](./LICENSE) for details.

