# Wildfire Burn Area Mapping Using Machine Learning and Sentinel-2 Imagery: Multi-Region Generalisation from Greece, Turkey, and California

## Table of Contents
- [Project Overview](#project-overview)
- [Background and Motivation](#background-and-motivation)
- [Problem Statement](#problem-statement)
- [Methodology](#methodology)
 - [Study Areas and Data Acquisition](#study-areas-and-data-acquisition)
 - [Ground Truth Generation](#ground-truth-generation)
 - [Feature Engineering](#feature-engineering)
 - [Machine Learning Approaches](#machine-learning-approaches)
 - [Spatial Validation Strategy](#spatial-validation-strategy)
- [Results and Performance](#results-and-performance)
- [Environmental Impact Assessment](#environmental-impact-assessment)
- [Discussion and Implications](#discussion-and-implications)
- [Limitations and Future Work](#limitations-and-future-work)
- [Conclusion](#conclusion)
- [Code Tutorial and Rundown](#code-tutorial-and-rundown)
- [References](#references)

## Project Overview

This project develops and evaluates machine learning approaches for automated wildfire burn area detection using pre- and post-fire Sentinel-2 satellite imagery. The study addresses the critical need for rapid, accurate, and scalable wildfire damage assessment by training models on Mediterranean ecosystems (Greece Evia 2021, Turkey Manavgat 2021) and testing cross-ecosystem generalisation on North American wildfires (California August Complex 2020).

Traditional field-based burn mapping approaches are labour-intensive, time-consuming, and often infeasible over large or inaccessible regions, requiring weeks to months and producing significant carbon emissions from vehicle and helicopter operations. This research demonstrates how satellite remote sensing combined with machine learning can provide rapid, cost-effective, and environmentally sustainable alternatives.

<img src="https://github.com/21MN28/GEOL0069_Project/blob/main/methodology_overview.png?raw=true" /> 
*Figure 1: Complete methodology workflow showing image acquisition, RGB burn index calculation, statistical thresholding, and data preparation for machine learning.*

The project implements three distinct machine learning approaches with rigorous spatial validation: Random Forest (supervised ensemble method), Convolutional Neural Networks (supervised deep learning), and K-means clustering (unsupervised pattern recognition). Each approach is evaluated using 50×50 pixel spatial blocking to prevent data leakage and ensure realistic performance assessment for operational deployment.

## Background and Motivation

### The Global Wildfire Challenge

Wildfire activity has increased dramatically across the globe, with climate change extending fire seasons and creating more extreme fire weather conditions. The 2021 Mediterranean fire season saw devastating blazes across Greece and Turkey that burned hundreds of thousands of hectares, while California continues to experience record-breaking wildfire seasons. Rapid assessment of burned areas is essential for emergency response coordination, ecosystem management planning, insurance claim processing, and climate impact studies.

Traditional ground-based damage assessment involves field teams surveying burned areas on foot or by helicopter, a process that can take weeks to months for large fires. This approach faces critical limitations including personnel safety risks in recently burned areas, accessibility challenges in remote mountainous regions, high costs for specialised personnel and helicopter operations, and incomplete spatial coverage leading to sampling bias.

### Remote Sensing Solutions

Satellite remote sensing offers a transformative alternative for wildfire damage assessment. The European Space Agency's Sentinel-2 mission provides multispectral imagery at 10-20m resolution with a 5-day revisit cycle, making it ideal for pre- and post-disaster analysis. The high spatial resolution enables detection of burn patterns at the landscape scale while the frequent revisit cycle ensures timely acquisition of cloud-free imagery crucial for rapid response.

Vegetation indices such as NDVI and NBR are widely used to quantify vegetation health and detect fire-related damage. However, operational constraints in data access sometimes limit the availability of SWIR bands required for NBR calculation, necessitating innovative approaches using RGB and Near-Infrared bands available in most satellite missions.

### Machine Learning in Fire Detection

Machine learning approaches have shown promise for automated burn area detection, offering advantages over traditional threshold-based methods through their ability to handle complex spectral relationships and adapt to varying environmental conditions. However, most studies focus on single regions or ecosystems, limiting their operational applicability. Cross-ecosystem generalisation remains a significant challenge, as spectral signatures of burned vegetation can vary between Mediterranean shrublands, temperate forests, and other biomes.

## Problem Statement

The primary research question addressed by this project is: **Can machine learning models trained on Mediterranean wildfire data accurately detect burn areas in North American ecosystems, and how do different ML approaches compare in cross-ecosystem generalisation performance?**

This addresses a critical gap in current wildfire remote sensing research, where most studies focus on single geographic regions. Understanding cross-ecosystem transferability is essential for developing operational global fire monitoring systems.

Specific research objectives include:

1. **Develop a robust feature extraction pipeline** using Sentinel-2 RGB and NIR bands when SWIR bands are unavailable
2. **Implement and compare three machine learning approaches**: supervised Random Forest, supervised CNN, and unsupervised K-means clustering
3. **Establish proper spatial validation protocols** to prevent data leakage common in remote sensing applications
4. **Evaluate cross-ecosystem generalisation** from Mediterranean to North American fire environments
5. **Assess environmental impact** of ML-based approaches compared to traditional field surveys

## Methodology

<img src="https://github.com/21MN28/GEOL0069_Project/blob/main/data_split.png?raw=true" /> 
*Figure 2: Spatial data splitting strategy showing training/validation allocation across regions and the three machine learning approaches.*

### Study Areas and Data Acquisition

The project focuses on three major wildfire events selected for their contrasting ecosystems:

**Greece - Evia Wildfire (August 2021)**: Burned approximately 50,000 hectares of Mediterranean pine forests and shrublands. The ecosystem is characterised by drought-adapted vegetation including Pinus halepensis and Cistus species typical of Mediterranean maquis.

**Turkey - Manavgat Wildfire (July-August 2021)**: Part of the devastating 2021 Turkish fire season, affecting mixed Mediterranean forests and agricultural areas in Antalya Province. The burn area includes both natural vegetation and human-modified landscapes.

**California - August Complex (August-November 2020)**: The largest wildfire in California recorded history, burning over 400,000 hectares across diverse ecosystems including Douglas fir forests, oak woodlands, chaparral shrublands, and grasslands.

For each fire event, Sentinel-2 Level-2A imagery was acquired with pre-fire images captured 5-10 days before ignition and post-fire images captured 5-10 days after fire containment, both with minimal cloud cover.

### Ground Truth Generation

A critical innovation is the development of a RGB-based burn index for ground truth generation, necessitated by limited access to SWIR bands required for traditional NBR calculation. The RGB Burn Index combines multiple spectral change indicators:

**RGB Burn Index = 0.5 × ΔGRVI + 0.3 × Δgreen + 0.2 × Δred**

Where ΔGRVI captures vegetation vigour loss, Δgreen represents chlorophyll decrease, and Δred indicates exposed soil and char signatures. The threshold of ≥0.06 was determined through visual correlation with post-fire imagery and statistical optimisation using ROC analysis, showing excellent alignment with ROC-optimal values across all regions.

### Feature Engineering

The project develops a comprehensive 24-feature set capturing spectral changes, spatial patterns, and contextual information. Key features include spectral differences (ΔNDVI, Δred, Δgreen, Δblue) that quantify fire-induced changes, the RGB Burn Index as the primary burn indicator, spatial texture measures using gradient magnitude and local standard deviation, and location encoding to capture ecosystem-specific patterns.

### Machine Learning Approaches

#### Random Forest (Supervised)
The Random Forest implementation uses balanced class weights to address natural class imbalance, 100 estimators for stable predictions, and square root feature subsampling for diversity. The model was trained on 500,000 stratified samples from Greece and Turkey data, requiring approximately 4.2 minutes with a computational carbon footprint of 3.9g CO₂.

#### Convolutional Neural Network (Supervised)
The CNN processes 32×32×8 input patches combining pre-RGB, post-RGB, pre-NDVI, and post-NDVI channels. The architecture uses progressive convolution blocks (32→64→128 filters) with batch normalisation, progressive dropout rates, and data augmentation for improved generalisation.

#### K-means Clustering (Unsupervised)
The unsupervised approach demonstrates burn detection without labeled training data. The pipeline uses StandardScaler preprocessing, PCA dimensionality reduction, and K=2 clustering with burn likelihood scoring based on spectral signatures. This achieved 87.7% accuracy without using any labeled data.

### Spatial Validation Strategy

A critical methodological innovation addresses spatial autocorrelation in remote sensing ML studies. The approach uses 50×50 pixel spatial blocks with zero overlap, ensuring complete spatial separation between training, validation, and test sets. For Turkey, blocks are allocated 70% training, 20% validation, and 10% testing. Greece provides additional training/validation data, while California serves as the cross-ecosystem test set.

## Results and Performance

### Model Performance Comparison

| Model | Turkey (Intra-ecosystem) | California (Cross-ecosystem) | Geographic Penalty | F1-Score (California) |
|-------|:------------------------:|:----------------------------:|:-----------------:|:--------------------:|
| **Random Forest** | 92.7% | 86.8% | 5.9 pp | 0.897 |
| **CNN** | 94.5% | 88.1% | 6.4 pp | 0.910 |
| **Unsupervised K-means** | 87.8% | 82.7% | 5.1 pp | 0.874 |

### Key Findings

**Random Forest** achieved the most balanced performance with 92.7% accuracy on Turkey and 86.8% on California, demonstrating robust transferability. Feature importance analysis confirmed that ΔNDVI, RGB burn index, and post-fire NDVI were the most discriminative features.

**CNN** achieved the highest overall accuracy with superior spatial pattern recognition capabilities. The patch-based approach enabled learning of characteristic spatial patterns of burn boundaries and transition zones.

**Unsupervised K-means** achieved remarkably strong performance (87.8% on Turkey) without labeled training data, demonstrating that burn spectral signatures are naturally separable. The slightly lower geographic penalty suggests reduced overfitting to ecosystem-specific characteristics.

All three approaches showed consistent 5-6% accuracy reduction when transferring from Mediterranean to North American ecosystems, indicating reasonable cross-ecosystem transferability while highlighting the importance of diverse training data.

### Statistical Validation

The RGB burn index threshold validation demonstrated excellent alignment across regions:
- **Turkey**: ROC-optimal 0.0600 vs selected 0.06 (perfect alignment)
- **Greece**: ROC-optimal 0.0668 vs selected 0.06 (0.007 difference)
- **California**: ROC-optimal 0.0692 vs selected 0.06 (0.009 difference)

## Environmental Impact Assessment

### The Hidden Cost of AI Research

Machine learning and AI research carry significant but often invisible environmental costs. Every model training session contributes to global energy consumption and carbon emissions. This project explicitly tracks computational carbon footprint to demonstrate environmental responsibility and highlight sustainability benefits of satellite-based approaches over traditional field methods.

### Computational Carbon Footprint

| Phase | Duration | Energy (Wh) | Carbon (g CO₂) |
|-------|:--------:|:-----------:|:--------------:|
| Data Acquisition | 15 min | 2.1 | 1.1 |
| Random Forest Training | 5 min | 7.8 | 3.9 |
| CNN Training | 45 min | 45.3 | 22.7 |
| Unsupervised Analysis | 8 min | 3.2 | 1.6 |
| **Total Project** | **73 min** | **58.4** | **29.3** |

### Environmental Benefits

Traditional field surveys require vehicle transport, helicopter operations, and extended personnel deployment, producing approximately 1,500 kg CO₂ per fire assessment. The ML pipeline produces only 29.3g CO₂, representing a **99.98% reduction**. Beyond direct carbon savings, satellite-based approaches eliminate vehicle emissions in sensitive ecosystems and enable rapid assessment supporting faster protection measures.

## Discussion and Implications

### Scientific Contributions

This research provides the first systematic evaluation of wildfire detection model transferability between Mediterranean and North American biomes. The spatial validation methodology demonstrates the critical importance of proper spatial blocking to prevent overoptimistic performance estimates. The RGB-based burn index offers a practical solution when traditional NBR calculation is unavailable, while the comparison reveals that burn spectral signatures are naturally separable across ecosystems.

### Operational Implications

Results suggest operational wildfire detection systems could achieve 85-90% accuracy across diverse ecosystems using models trained on limited geographic regions. The rapid processing capability enables integration with satellite data pipelines for near real-time burn area mapping, supporting emergency response within hours of image acquisition.

## Limitations and Future Work

### Current Limitations

The approach requires cloud-free imagery and operates at 10m spatial resolution, potentially missing small burn patches. The temporal constraint of 5-10 day windows may not capture rapid vegetation recovery. The ecosystem scope is limited to Mediterranean and temperate forests, with performance on other biomes unknown.

### Future Directions

Multi-sensor fusion incorporating Landsat, MODIS, and SAR could improve temporal coverage and cloud penetration. Real-time processing systems could enable operational deployment. Expansion to burn severity mapping and testing on tropical, boreal, and grassland fires would validate global applicability.

## Conclusion

This study demonstrates that machine learning approaches can effectively detect wildfire burn areas using Sentinel-2 imagery with reasonable generalisation across ecosystems. All three approaches achieved accuracy levels suitable for operational deployment, with cross-ecosystem performance penalties of only 5-6%.

Key findings include the effectiveness of RGB-based burn indices when SWIR data is unavailable, the critical importance of spatial validation in remote sensing applications, and the surprising performance of unsupervised methods without labels. The research provides a foundation for operational wildfire monitoring systems that could transform damage assessment from weeks-long field campaigns to automated processes completed within hours.


## Code Tutorial and Rundown

For a comprehensive walkthrough of the code implementation and methodology, watch the detailed tutorial video:

[![Code Tutorial and Rundown](https://img.youtube.com/vi/pL4hfNbUikI/maxresdefault.jpg)](https://www.youtube.com/watch?v=pL4hfNbUikI)

This video provides step-by-step explanations of the machine learning pipeline, feature engineering process, and model implementation details covered in this project.

## References

1. Chuvieco, E., et al. (2019). Historical background and current developments for mapping burned area from satellite Earth observation. *Remote Sensing of Environment*, 225, 45-64.

2. Fernández-García, V., et al. (2018). Burn severity metrics in fire-prone pine ecosystems along a climatic gradient using Landsat imagery. *Remote Sensing of Environment*, 206, 205-217.

3. Giglio, L., et al. (2018). The Collection 6 MODIS burned area mapping algorithm and product. *Remote Sensing of Environment*, 217, 72-85.

4. Hawbaker, T. J., et al. (2017). The Landsat Burned Area algorithm and products for the conterminous United States. *Remote Sensing of Environment*, 244, 111801.

5. Key, C. H., & Benson, N. C. (2006). Landscape assessment: ground measure of severity, the Composite Burn Index; and remote sensing of severity, the Normalized Burn Ratio. *FIREMON: Fire effects monitoring and inventory system*, 1-51.

6. Roteta, E., et al. (2019). Development of a Sentinel-2 burned area algorithm: Generation of a small fire database for sub-Saharan Africa. *Remote Sensing of Environment*, 222, 1-17.

7. San-Miguel-Ayanz, J., et al. (2022). Forest Fires in Europe, Middle East and North Africa 2021. Publications Office of the European Union.

---

*This research was conducted as part of the GEOL0069 AI for Earth Observation course at University College London, demonstrating the application of machine learning techniques to critical environmental monitoring challenges.*

## Contact

**Author**: Maral Nikkhah  
**Email**: maral.nikkhah.21@ucl.ac.uk  
**Institution**: University College London  
**Course**: GEOL0069 - AI for Earth Observation  

## Acknowledgments

This project was created for GEOL0069 at University College London, taught by Dr. Michel Tsamados and Weibin Chen. Special thanks for their guidance and teaching.
