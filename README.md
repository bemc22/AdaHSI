# Deep Adaptive Superpixels for Hadamard Single Pixel Imaging in Near-Infrared Spectrum

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bemc22/AdaHSI/blob/main/notebooks/demo_inferece.ipynb) [![DOI:10.1109/ICASSP49357.2023.10095165](https://zenodo.org/badge/DOI/10.1109/ICASSP49357.2023.10095165.svg)](https://doi.org/10.1109/ICASSP49357.2023.10095165)

## Abstract

Hadamard single-pixel imaging (HSI) is a promising sensing approach for acquiring spectral images in the near-infrared spectrum with high spatial resolution and fast recovery times due to the efficient invertible properties of the Hadamard matrix. The potential of the HSI system is diminished because of the large number of required measurements which implies long acquisition times. Recent advances proposed optimizing the HSI sensing matrix structure based on a superpixels map estimated from a side-information acquisition of the scene, reducing the number of required measurements. However, these matrix designs are detached from the recovery task, which falls on a sub-optimal strategy. In this work, we proposed an adaptive end-to-end sensing methodology for the HSI sensing matrix design based on deep superpixels estimation by coupling the sensing and recovery of the near-infrared spectral images. Experimental results show the superiority of the proposed sensing methodology compared with state-of-art sensing design schemes.


## Demos

| Notebook      | Link          |
| ------------- | ------------- |
| Demo Inference  | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bemc22/AdaHSI/blob/main/notebooks/demo_inferece.ipynb)  |
| Demo Superpixels  | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bemc22/AdaHSI/blob/main/notebooks/demo_deep_superpixels.ipynb)  |


## How to cite
If this code is useful for your and you use it in an academic work, please consider citing this paper as

```bib
@inproceedings{monroy2023deep,
author = {Monroy, Brayan and Bacca, Jorge and Arguello, Henry},
booktitle={ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
pages={1--5},
year={2023},
organization={IEEE}
doi = {10.1109/ICASSP49357.2023.10095165},
}
```
