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
- [References](#references)
- [Repository Structure](#repository-structure)
- [Usage](#usage)

## Project Overview

This project develops and evaluates machine learning approaches for automated wildfire burn area detection using pre- and post-fire Sentinel-2 satellite imagery. The study addresses the critical need for rapid, accurate, and scalable wildfire damage assessment by training models on Mediterranean ecosystems (Greece Evia 2021, Turkey Manavgat 2021) and testing cross-ecosystem generalisation on North American wildfires (California August Complex 2020).

The research demonstrates how satellite remote sensing combined with machine learning can provide rapid, cost-effective, and environmentally sustainable alternatives for post-fire damage assessment. Traditional field-based burn mapping approaches are labour-intensive, time-consuming, and often infeasible over large or inaccessible regions, requiring weeks to months and producing significant carbon emissions from vehicle and helicopter operations.

<img src="https://github.com/21MN28/GEOL0069_Project/blob/main/methodology_overview.png?raw=true" /> 
*Figure 1: Complete methodology workflow showing image acquisition, RGB burn index calculation, statistical thresholding, and data preparation for machine learning. The RGB burn index (BI = 2R - GB) provides an innovative solution when traditional NBR calculation using SWIR bands is unavailable.*

The project implements three distinct machine learning approaches with rigorous spatial validation: Random Forest (supervised ensemble method), Convolutional Neural Networks (supervised deep learning), and K-means clustering (unsupervised pattern recognition). Each approach is evaluated using 50×50 pixel spatial blocking to prevent data leakage and ensure realistic performance assessment for operational deployment.

## Background and Motivation

### The Global Wildfire Challenge

Wildfire activity has increased dramatically across the globe, with climate change extending fire seasons and creating more extreme fire weather conditions. The 2021 Mediterranean fire season saw devastating blazes across Greece and Turkey that burned hundreds of thousands of hectares, while California continues to experience record-breaking wildfire seasons with fires like the August Complex covering over 400,000 hectares. Rapid assessment of burned areas is essential for emergency response coordination, ecosystem management planning, insurance claim processing, and climate impact studies.

Traditional ground-based damage assessment involves field teams surveying burned areas on foot or by helicopter, a process that can take weeks to months for large fires. This approach faces several critical limitations including personnel safety risks in recently burned areas with unstable terrain and toxic smoke, accessibility challenges in remote mountainous regions, high costs for specialised personnel and helicopter operations, and incomplete spatial coverage leading to sampling bias in damage estimates.

### Remote Sensing Solutions

Satellite remote sensing offers a transformative alternative for wildfire damage assessment that addresses these operational challenges. The European Space Agency's Sentinel-2 mission provides multispectral imagery at 10-20m resolution with a 5-day revisit cycle, making it ideal for pre- and post-disaster analysis. The high spatial resolution enables detection of burn patterns at the landscape scale while maintaining sufficient detail for management decisions, and the frequent revisit cycle ensures timely acquisition of cloud-free imagery crucial for rapid response.

Vegetation indices such as the Normalized Difference Vegetation Index (NDVI) and Normalized Burn Ratio (NBR) are widely used to quantify vegetation health and detect fire-related damage. NDVI utilises the contrast between red absorption and near-infrared reflection in healthy vegetation, while NBR additionally incorporates short-wave infrared bands that are sensitive to vegetation moisture content. However, operational constraints in data access sometimes limit the availability of SWIR bands required for NBR calculation, necessitating innovative approaches using RGB and Near-Infrared bands available in most satellite missions.

### Machine Learning in Fire Detection

Machine learning approaches have shown promise for automated burn area detection, offering advantages over traditional threshold-based methods through their ability to handle complex spectral relationships and adapt to varying environmental conditions. Random Forest algorithms excel at handling high-dimensional remote sensing data and providing feature importance rankings. Convolutional Neural Networks can capture spatial patterns and contextual information that single-pixel approaches miss. Unsupervised clustering methods offer the advantage of requiring no labeled training data, crucial for rapid deployment to new regions.

However, most studies focus on single regions or ecosystems, limiting their operational applicability. Cross-ecosystem generalisation remains a significant challenge, as spectral signatures of burned vegetation can vary between Mediterranean shrublands, temperate forests, boreal systems, and other biomes due to differences in vegetation structure, fire behaviour, and post-fire recovery patterns.

## Problem Statement

The primary research question addressed by this project is: **Can machine learning models trained on Mediterranean wildfire data accurately detect burn areas in North American ecosystems, and how do different ML approaches compare in cross-ecosystem generalisation performance?**

This question addresses a critical gap in current wildfire remote sensing research, where most studies focus on single geographic regions or ecosystem types. Understanding cross-ecosystem transferability is essential for developing operational global fire monitoring systems that can provide consistent performance across diverse biomes.

Specific research objectives include:

1. **Develop a robust feature extraction pipeline** using Sentinel-2 RGB and NIR bands when SWIR bands are unavailable, creating spectral and spatial features that capture burn signatures across ecosystem types.

2. **Implement and compare three machine learning approaches**: supervised Random Forest, supervised CNN, and unsupervised K-means clustering to understand the trade-offs between accuracy, computational requirements, and training data needs.

3. **Establish proper spatial validation protocols** to prevent data leakage common in remote sensing applications, ensuring realistic performance estimates for operational deployment.

4. **Evaluate cross-ecosystem generalisation** from Mediterranean to North American fire environments, quantifying the performance penalty and identifying factors affecting transferability.

5. **Assess environmental impact** of ML-based approaches compared to traditional field surveys, including computational carbon footprint and scalability considerations.

## Methodology

<img src="https://github.com/21MN28/GEOL0069_Project/blob/main/data_split.png?raw=true" /> 
*Figure 2: Spatial data splitting strategy showing training/validation allocation across regions and the three machine learning approaches. Turkey provides intra-ecosystem evaluation through spatial blocking, Greece serves as additional training data, and California tests cross-ecosystem generalisation. Each method processes identical spatial blocks to ensure fair comparison.*

### Study Areas and Data Acquisition

The project focuses on three major wildfire events selected for their contrasting ecosystems, fire characteristics, and data availability:

**Greece - Evia Wildfire (August 2021)**: This fire burned approximately 50,000 hectares of Mediterranean pine forests and shrublands on the island of Evia during one of Greece's most severe fire seasons. The ecosystem is characterised by drought-adapted vegetation including Pinus halepensis (Aleppo pine), Quercus coccifera (kermes oak), and Cistus species typical of Mediterranean maquis. The fire exhibited high-intensity crown fire behaviour in forested areas and rapid spread through shrubland communities.

**Turkey - Manavgat Wildfire (July-August 2021)**: Part of the devastating 2021 Turkish fire season that burned over 200,000 hectares nationwide, this fire affected mixed Mediterranean forests and agricultural areas in the Antalya Province. The burn area includes both natural vegetation (pine-dominated forests, oak woodlands) and human-modified landscapes (olive groves, agricultural fields), providing insight into model performance across land use types.

**California - August Complex (August-November 2020)**: The largest wildfire in California recorded history, burning over 400,000 hectares across multiple counties in Northern California. The fire burned through diverse ecosystems including Douglas fir forests, oak woodlands, chaparral shrublands, and grasslands, representing a significant test of model transferability to North American vegetation types and fire regimes.

For each fire event, Sentinel-2 Level-2A (atmospherically corrected Bottom-of-Atmosphere reflectance) imagery was acquired following strict temporal and quality criteria. Pre-fire images were captured 5-10 days before ignition with cloud cover <10% to establish baseline vegetation conditions. Post-fire images were captured 5-10 days after fire containment with cloud cover <5% to capture immediate burn effects before significant vegetation recovery. Both images were selected from the same relative orbit to ensure consistent viewing geometry and minimize radiometric differences unrelated to fire effects.

### Ground Truth Generation

A critical innovation of this project is the development of a RGB-based burn index for ground truth generation, necessitated by limited access to SWIR bands required for traditional NBR calculation through the available API endpoints. The RGB Burn Index was developed based on spectral change theory and validated through visual and statistical analysis.

The index combines multiple spectral change indicators known to characterise burned vegetation:

**RGB Burn Index = 0.5 × ΔGRVI + 0.3 × Δgreen + 0.2 × Δred**

Where ΔGRVI represents the change in Green-Red Vegetation Index [(Green-Red)/(Green+Red)] capturing vegetation vigour loss, Δgreen represents the decrease in green channel values indicating chlorophyll loss, and Δred represents the increase in red channel values indicating exposed soil and char. The weighting scheme prioritises vegetation index changes (0.5) while incorporating direct spectral changes (0.3, 0.2).

The threshold of ≥0.06 was determined through a rigorous validation process involving visual correlation with post-fire RGB imagery to identify obviously burned areas, statistical optimization using ROC analysis to maximize true positive rate while minimizing false positive rate, and cross-regional consistency testing to ensure threshold transferability. This approach was validated statistically across all three regions, with the selected threshold showing excellent alignment with ROC-optimal values, confirming its effectiveness as a surrogate for traditional NBR-based approaches.

### Feature Engineering

The project develops a comprehensive 24-feature set designed to capture spectral, temporal, spatial, and contextual information relevant to burn detection across diverse ecosystems:

**Spectral Features (12)**: Raw RGB channel values from pre- and post-fire imagery provide baseline spectral information. NDVI values calculated as (NIR-Red)/(NIR+Red) from both time periods capture vegetation vigour. Spectral differences (Δred, Δgreen, Δblue, ΔNDVI) quantify fire-induced changes. The Green-Red Vegetation Index change (ΔGRVI) provides an RGB-based vegetation indicator. Brightness changes capture overall reflectance modifications. Red/green ratio changes highlight the characteristic spectral shift toward red wavelengths in burned areas.

**Composite Indices (4)**: The RGB Burn Index serves as the primary target feature for training. An NBR proxy was developed using RGB channels where NIR loss is approximated by green channel decrease and SWIR increase by red channel increase. A vegetation stress index combines NDVI and GRVI changes weighted by their correlation with burn damage. An integrated burn likelihood score combines multiple spectral indicators.

**Spatial Features (2)**: Gradient magnitude calculated using Sobel edge detection on ΔNDVI maps captures burn boundary characteristics important for spatial pattern recognition. Local standard deviation computed using 3×3 kernel operations provides a texture measure that distinguishes burned areas (typically more heterogeneous) from unburned vegetation (more homogeneous).

**Location Encoding (3)**: One-hot encoding for Greece, Turkey, and California captures potential location-specific spectral signatures while allowing models to learn ecosystem-specific patterns without overfitting to geographic coordinates.

**Context Features (3)**: Pre- and post-fire brightness measures provide overall illumination context. Additional spectral ratios capture subtle spectral relationships that may improve discrimination in challenging cases such as partial burns or areas with mixed vegetation recovery.

### Machine Learning Approaches

#### Random Forest (Supervised)

The Random Forest implementation leverages ensemble learning to achieve robust performance across diverse spectral conditions. The model uses balanced class weights to address the natural imbalance between burned and unburned pixels common in wildfire datasets where burned areas typically represent 20-60% of pixels. Key hyperparameters were optimized through cross-validation: 100 estimators provide stable predictions without overfitting, maximum depth of 20 prevents excessive model complexity while allowing sufficient pattern capture, square root feature subsampling promotes diversity among trees, and minimum samples per leaf of 10 ensures smooth decision boundaries.

The model was trained on 500,000 stratified samples drawn from Greece and Turkey data, with careful attention to maintaining class balance and spatial distribution. Training utilized all CPU cores for parallel processing, requiring approximately 4.2 minutes with a computational carbon footprint of 3.9g CO₂. Feature importance analysis revealed that ΔNDVI, RGB burn index, and post-fire NDVI were the most discriminative features across all regions.

#### Convolutional Neural Network (Supervised)

The CNN architecture processes 32×32×8 input patches that combine pre-RGB (3 channels), post-RGB (3 channels), pre-NDVI (1 channel), and post-NDVI (1 channel) information. This multi-temporal input design allows the network to learn both spatial patterns within each time period and temporal changes between periods. The architecture uses progressive convolution blocks (32→64→128 filters) with 3×3 kernels to capture features at multiple scales, from fine textures to broader spatial patterns.

Key design choices optimize performance and prevent overfitting: Batch normalization after each convolutional layer accelerates training and improves stability. Progressive dropout rates (0.1→0.2→0.3→0.5) provide increasing regularization at deeper layers. Data augmentation including random rotation (±10°), horizontal and vertical flipping, zoom (±10%), and contrast adjustment (±10%) improves generalization to new viewing conditions and fire patterns. Global Average Pooling reduces parameters compared to fully connected layers while maintaining spatial translation invariance.

The patch-based training approach extracts 15-20 patches per 50×50 pixel spatial block, with the center pixel of each patch serving as the classification target. This strategy balances computational efficiency with spatial coverage while maintaining the spatial validation framework. Training used early stopping based on validation loss with a patience of 10 epochs to prevent overfitting.

#### K-means Clustering (Unsupervised)

The unsupervised approach demonstrates the potential for burn detection without labeled training data, crucial for rapid deployment to new regions where ground truth may be unavailable. The pipeline begins with StandardScaler preprocessing to ensure all features contribute equally to distance calculations. Principal Component Analysis reduces dimensionality while retaining 95% of variance, typically requiring 8-10 components and improving computational efficiency while removing noise.

K-means clustering with K=2 provides natural binary classification into burned and unburned classes. The algorithm was initialized with K-means++ seeding and run with 20 different initializations to ensure robust cluster identification. Cluster assignment to burned/unburned classes was determined through burn likelihood scoring based on domain knowledge of fire spectral signatures.

The scoring system evaluates each cluster based on multiple burn indicators: higher ΔNDVI values indicate greater vegetation loss, higher RGB burn index values suggest stronger burn signatures, lower post-fire NDVI values indicate reduced vegetation vigour, and higher vegetation stress indicators suggest fire impact. This approach achieved 87.7% accuracy on training regions without using any labeled data, demonstrating the natural separability of burn spectral signatures.

### Spatial Validation Strategy

A critical methodological innovation addresses the widespread problem of spatial autocorrelation in remote sensing machine learning studies. Traditional random train/test splits often include spatially adjacent pixels in both training and test sets, leading to overly optimistic performance estimates due to the spatial correlation inherent in earth observation data.

The spatial validation approach uses 50×50 pixel spatial blocks (approximately 500m × 500m at 10m resolution) with zero overlap between blocks. This block size was selected to exceed the typical spatial autocorrelation range in vegetation spectral signatures while maintaining sufficient samples per block for statistical analysis. Entire blocks are assigned to training, validation, or test sets, ensuring complete spatial separation.

For Turkey (intra-ecosystem evaluation), spatial blocks are randomly allocated with 70% for training, 20% for validation, and 10% for held-out testing. This provides both model development data and independent evaluation within the same ecosystem. For Greece, 80% of blocks serve training and 20% validation, providing additional training diversity. California serves entirely as a cross-ecosystem test set, enabling unbiased evaluation of model transferability.

This spatial blocking approach provides more realistic performance estimates that better reflect operational deployment conditions where models must classify previously unseen geographic areas rather than interpolating between known locations.

## Results and Performance

### Model Performance Comparison

The comprehensive evaluation across three machine learning approaches reveals distinct strengths and trade-offs in wildfire burn detection:

| Model | Turkey (Intra-ecosystem) | California (Cross-ecosystem) | Geographic Penalty | F1-Score (California) | Precision | Recall |
|-------|:------------------------:|:----------------------------:|:-----------------:|:--------------------:|:---------:|:------:|
| **Random Forest** | 92.7% | 86.8% | 5.9 pp | 0.897 | 0.865 | 0.933 |
| **CNN** | 94.5% | 88.1% | 6.4 pp | 0.910 | 0.858 | 0.969 |
| **Unsupervised K-means** | 87.8% | 82.7% | 5.1 pp | 0.874 | 0.797 | 0.966 |

### Detailed Performance Analysis

**Random Forest Performance**: The Random Forest model achieved the most balanced performance across ecosystems with strong generalization characteristics. On Turkey (intra-ecosystem evaluation), the model reached 92.7% accuracy with excellent balance between precision (90.9%) and recall (90.8%). The California cross-ecosystem test demonstrated robust transferability with 86.8% accuracy, though with a shift toward higher recall (93.3%) and lower precision (86.5%). This pattern suggests the model tends to be conservative, preferring to identify potential burns rather than miss them, which is appropriate for disaster response applications.

The confusion matrix for California reveals 290,302 true negatives, 90,460 false positives, 41,636 false negatives, and 577,602 true positives. The false positive rate of 23.8% primarily occurs in areas with natural vegetation stress or agricultural land use changes that mimic burn spectral signatures. Feature importance analysis confirmed that ΔNDVI (18% importance), RGB burn index (15% importance), and post-fire NDVI (12% importance) were the most discriminative features.

**CNN Performance**: The Convolutional Neural Network achieved the highest overall accuracy with superior spatial pattern recognition capabilities. The 94.5% accuracy on Turkey and 88.1% on California demonstrates the value of spatial context in burn detection. The patch-based approach enabled the network to learn characteristic spatial patterns of burn boundaries, heterogeneous burn severity, and the transition zones between burned and unburned areas.

However, pixel-level reconstruction from patches introduced some spatial artifacts. The stride=16 approach used for computational efficiency created slight discontinuities at patch boundaries, resulting in 73-76% accuracy for full-image predictions compared to 88-94% for patch-based evaluation. This trade-off between computational efficiency and spatial accuracy represents a key consideration for operational deployment.

**Unsupervised Performance**: The K-means clustering approach achieved remarkably strong performance considering the complete absence of labeled training data. The 87.8% accuracy on Turkey demonstrates that burn spectral signatures are naturally separable in multispectral feature space. The burn likelihood scoring successfully identified Cluster 1 as representing burned areas based on characteristic spectral signatures: high ΔNDVI (-0.59), high RGB burn index (0.16), and strong vegetation stress indicators.

The slightly lower geographic penalty (5.1 percentage points) compared to supervised methods suggests that unsupervised approaches may be less prone to overfitting to specific ecosystem characteristics present in training data. This finding has important implications for rapid deployment to new geographic regions where labeled data may be unavailable.

### Cross-Ecosystem Generalisation Analysis

All three approaches demonstrated similar geographic penalties of 5-6 percentage points when transferring from Mediterranean to North American ecosystems. This consistency across different algorithmic approaches suggests that the spectral signatures of burned vegetation contain fundamental characteristics that are reasonably transferable across biomes, though ecosystem-specific variations do impact performance.

The relatively modest generalization penalty indicates that models trained on Mediterranean fires can provide operationally useful burn area estimates for North American fires. However, the consistent 5-6% accuracy reduction across all methods highlights the importance of diverse training data for optimal operational deployment. This penalty likely reflects differences in vegetation structure (Mediterranean shrublands vs North American coniferous forests), fire behavior (rapid vs slow-spreading fires), and post-fire spectral signatures.

### Statistical Validation of Ground Truth

The RGB burn index threshold validation demonstrated excellent statistical alignment across all regions, confirming the robustness of the ground truth generation approach:

**Greece**: Burned areas showed mean RGB burn index of 0.154 versus 0.024 for unburned areas (p < 0.001). The threshold correctly identified 94.0% of burned areas with 16.6% false positive rate. ROC-optimal threshold of 0.0668 differed by only 0.007 from the selected value.

**Turkey**: Mean RGB burn index was 0.161 for burned areas versus 0.008 for unburned areas (p < 0.001). The threshold correctly identified 93.5% of burned areas with 10.5% false positive rate. ROC-optimal threshold was exactly 0.0600, confirming perfect alignment.

**California**: Burned areas showed mean RGB burn index of 0.149 versus 0.036 for unburned areas (p < 0.001). The threshold achieved 93.8% true positive rate with 21.0% false positive rate. ROC-optimal threshold of 0.0692 differed by only 0.009 from the selected value.

## Environmental Impact Assessment

### Computational Carbon Footprint

The project includes comprehensive environmental cost tracking that demonstrates the sustainability benefits of satellite-based monitoring approaches:

| Phase | Duration | Energy (Wh) | Carbon (g CO₂) |
|-------|:--------:|:-----------:|:--------------:|
| Data Acquisition | 15 min | 2.1 | 1.1 |
| Random Forest Training | 5 min | 7.8 | 3.9 |
| CNN Training | 45 min | 45.3 | 22.7 |
| Unsupervised Analysis | 8 min | 3.2 | 1.6 |
| **Total Project** | **73 min** | **58.4** | **29.3** |

### Environmental Benefits

Traditional field surveys for large wildfire assessment typically require multiple vehicles, helicopter flights, and weeks of personnel time, resulting in approximately 1,500 kg CO₂ emissions per fire assessment. In contrast, the complete machine learning pipeline produces only 29.3g CO₂, representing a **99.98% reduction** in carbon emissions.

Beyond carbon footprint reduction, the ML approach eliminates vehicle emissions in sensitive burned ecosystems, reduces human disturbance during critical recovery periods, requires no helicopter flights over dangerous terrain, and enables rapid assessment supporting faster ecosystem protection measures. The scalability means environmental cost per fire assessment decreases dramatically as the system is applied to multiple events throughout a fire season.

## Discussion and Implications

### Scientific Contributions

This research makes several important contributions to the field of AI for Earth Observation. The cross-ecosystem validation framework provides the first systematic evaluation of wildfire detection model transferability between Mediterranean and North American biomes, addressing a critical gap in current literature. The spatial validation methodology demonstrates the critical importance of proper spatial blocking in remote sensing machine learning applications, preventing overoptimistic performance estimates common in the field.

The RGB-based burn index offers a practical solution when traditional NBR calculation is unavailable, with statistical validation confirming its effectiveness across diverse ecosystems. The comparison of supervised versus unsupervised approaches reveals that burn spectral signatures are naturally separable, with unsupervised methods achieving competitive performance without labeled training data.

### Operational Implications

The results suggest that operationally deployed wildfire detection systems could achieve 85-90% accuracy across diverse ecosystems using models trained on limited geographic regions. The 5-6% performance penalty for cross-ecosystem transfer is sufficiently small to provide useful damage estimates, though region-specific calibration could further improve accuracy.

The rapid processing capability (seconds per image) enables integration with existing satellite data pipelines for near real-time burn area mapping. This could support emergency response operations, insurance claim processing, and ecosystem management decisions within hours of image acquisition.

### Methodological Insights

The spatial validation results highlight a critical issue in remote sensing machine learning: traditional random train/test splits can severely overestimate model performance due to spatial autocorrelation. The 50×50 pixel spatial blocking approach provides more realistic performance estimates and should be adopted as standard practice in the field.

The success of the unsupervised approach (87.8% accuracy) suggests that spectral signatures of burned vegetation are sufficiently distinct to enable label-free detection. This has important implications for rapid deployment to new regions where labeled training data may not be available.

## Limitations and Future Work

### Current Limitations

The study faces several limitations that should be considered when interpreting results. The approach requires cloud-free imagery, limiting applicability during extended cloudy periods. The 10m spatial resolution may miss small burn patches or narrow burn boundaries. The temporal constraint of 5-10 day pre/post-fire windows may not capture rapid vegetation recovery in some ecosystems.

The ecosystem scope is limited to Mediterranean and temperate forest fires, with performance on tropical, boreal, or grassland fires remaining unknown. The binary burned/unburned classification doesn't capture burn severity gradients that may be important for ecosystem management applications.

### Future Research Directions

Several promising directions emerge from this work. Multi-sensor fusion incorporating Landsat, MODIS, and Synthetic Aperture Radar could improve temporal coverage and cloud penetration capabilities. Real-time processing systems could enable operational deployment for emergency response applications.

Expansion to burn severity mapping would provide more detailed information for ecosystem management. Testing on additional biomes including tropical forests, boreal forests, and grasslands would validate global applicability. Advanced deep learning architectures including Vision Transformers and semantic segmentation networks may improve spatial pattern recognition.

## Conclusion

This comprehensive study demonstrates that machine learning approaches can effectively detect wildfire burn areas using Sentinel-2 imagery with reasonable generalization across ecosystems. The Random Forest, CNN, and unsupervised K-means approaches all achieved accuracy levels suitable for operational deployment, with cross-ecosystem performance penalties of only 5-6%.

Key findings include the effectiveness of RGB-based burn indices when SWIR data is unavailable, the critical importance of spatial validation in remote sensing applications, the surprising performance of unsupervised methods (87.8% accuracy without labels), and the dramatic environmental benefits of ML approaches compared to traditional field surveys (99.98% carbon reduction).

The research provides a foundation for operational wildfire monitoring systems that could transform post-fire damage assessment from a weeks-long field campaign to an automated process completed within hours of satellite image acquisition. The methodological framework developed here provides a template for rigorous evaluation of machine learning applications in Earth observation.

## References

1. Chuvieco, E., Mouillot, F., van der Werf, G. R., San Miguel, J., Tanase, M., Koutsias, N., ... & Giglio, L. (2019). Historical background and current developments for mapping burned area from satellite Earth observation. *Remote Sensing of Environment*, 225, 45-64.

2. Fernández-García, V., Santamarta, M., Fernández-Manso, A., Quintano, C., Marcos, E., & Calvo, L. (2018). Burn severity metrics in fire-prone pine ecosystems along a climatic gradient using Landsat imagery. *Remote Sensing of Environment*, 206, 205-217.

3. Giglio, L., Boschetti, L., Roy, D. P., Humber, M. L., & Justice, C. O. (2018). The Collection 6 MODIS burned area mapping algorithm and product. *Remote Sensing of Environment*, 217, 72-85.

4. Hawbaker, T. J., Vanderhoof, M. K., Schmidt, G. L., Beal, Y. J., Picotte, J. J., Takacs, J. D., ... & Dwyer, J. L. (2017). The Landsat Burned Area algorithm and products for the conterminous United States. *Remote Sensing of Environment*, 244, 111801.

5. Key, C. H., & Benson, N. C. (2006). Landscape assessment: ground measure of severity, the Composite Burn Index; and remote sensing of severity, the Normalized Burn Ratio. *FIREMON: Fire effects monitoring and inventory system*, 1-51.

6. Roteta, E., Bastarrika, A., Padilla, M., Storm, T., & Chuvieco, E. (2019). Development of a Sentinel-2 burned area algorithm: Generation of a small fire database for sub-Saharan Africa. *Remote Sensing of Environment*, 222, 1-17.

7. San-Miguel-Ayanz, J., Durrant, T., Boca, R., Maianti, P., Libertà, G., Artes-Vivancos, T., ... & Leray, T. (2022). Forest Fires in Europe, Middle East and North Africa 2021. Publications Office of the European Union.

## Repository Structure
