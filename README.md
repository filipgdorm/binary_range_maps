# WIP! Generating Binary Range Maps

Code for binarizing the output of global-scale species range estimation models. This code enables the recreation of the results from our workshop paper [link]


<table>
  <tr>
    <td valign="top" align="center" style="text-align: center;"><img src="images/two_species_map.png" width="450"></td>
   </tr> 
    <td align="left" valign="top">
      Binary range maps for two different species. These ranges are generated by the <a href="coleICML2023">SINR</a> SDM for <a href="https://www.inaturalist.org/photos/67458827">Wood Thrush</a> (left) and <a href="https://www.inaturalist.org/photos/157601882">Tui Parakeet</a> (right). Converting the continuous SDM outputs to binary range maps requires setting thresholds (e.g. 0.02, 0.1, or 0.5) which result in very different range maps depending on the values chosen. 
More importantly, here the same threshold value is not the best for both species. 
    </td>
</table>


## 🌍 Overview 
Accurate predictions of species’ ranges are crucial for assisting conservation efforts. Traditionally, range maps are manually created by experts. However, species distribution models (SDMs) and, more
recently, deep learning-based variants offer a potential automated alternative. Deep learning-based SDMs generate a continuous probability representing the presence of a species at a given location, which must be binarized by setting per-species thresholds to obtain binary range maps. However, selecting appropriate per-species thresholds to binarize
these predictions is non-trivial, since different species can require different thresholds. In this work, we evaluate different approaches for automatically identifying the best thresholds for binarizing range maps using presence-only data. This includes approaches that require the generation
of additional pseudo-absence data, along with ones that only require presence data. We also propose an extension of an existing presence-only technique that is more robust to outliers. We perform a detailed evaluation of different thresholding techniques on the tasks of binary range estimation and large-scale fine-grained visual classification, and we demonstrate
improved performance over existing approaches using our technique.

<table>
  <tr>
    <td valign="top" align="center" style="text-align: center;"><img src="images/maps.png" width="450"></td>
   </tr> 
    <td align="left">Qualitative examples of estimated binary ranges. Each row depicts a different species, and the columns show the expert-derived range, output from the Target Sampling, and LPT-R approaches, respectively. Inset, we also display the different types of errors. We use an ocean mask for visualization purposes.</td>
</table>

## 🔍 Getting Started 

#### Installing Required Packages

1. We recommend using an isolated Python environment to avoid dependency issues. Install the Anaconda Python 3.9 distribution for your operating system from [here](https://www.anaconda.com/download). 

2. Create a new environment and activate it:
```bash
 conda create -y --name binary_range_maps python==3.9
 conda activate binary_range_maps
```

3. After activating the environment, install the required packages:
```bash
 pip3 install -r requirements.txt
```

#### Downloading the SINR Code
This range map binarization is an extension of https://github.com/elijahcole/sinr which needs to be cloned as a folder into this repo (such that there is a folder `sinr/*`). [note to self: depending on if shipped with clone, include setup instructions or not]


#### SINR Data Download and Preparation
Instructions for downloading the SINR data, which is needed for further experiments, are in `sinr/data/README.md`.

#### Data setup for pseudo-absence generation
In order to create pseudo-absences, the training data needs to be formatted.

#### Downloading models
To use the models in the paper, simply run the following command:
```
curl xxx
```

## 🗺️ Evaluating Methods
The evaluation process consists of two steps: 

1. Generate the thresholds for taxa of interest through any of the methods in `threshold_generation/*`.
2. Pass thresholds and taxons generated as a `.csv` file to `evaluation.py`



##  🙏 Acknowledgements


## 📜 Disclaimer
Extreme care should be taken before making any decisions based on the outputs of models presented here. Our models are trained on biased data and have not been calibrated or validated beyond the experiments illustrated in the paper. 
