# ML Predictions of pH

## Project Overview

This project aims to develop machine learning models for predicting pH from microbial abundances and 16S rRNA gene sequences. The approach incorporates both traditional machine learning methods (e.g., Random Forests and KNN regression) and deep learning techniques, with particular emphasis on language models for predictions from gene sequences.

## Methods

- **Traditional ML**: Random Forests, KNN regression, and other classical approaches
- **Deep Learning**: Language models and other neural network architectures, especially for gene sequence-based predictions

## Data Processing

The `process_rdp_data.py` script is used to aggregate taxonomic counts from RDP classifier output files. The script processes `.tab` files in the `data/RDP_classifier` directory and generates aggregated count tables saved as CSV files in the `data` directory. The output files contain taxonomic counts at various levels (phylum, class, order, family, genus) with samples as rows and taxonomic groups as columns.

**Note**: The source RDP classifier files are not stored in this repository but can be downloaded from the [JMDplots repository](https://github.com/jedick/JMDplots/tree/main/inst/extdata/orp16S/RDP-GTDB).

## Project History

_Notable changes to architecture, data, features, or results will be documented here._
