# Appendix
## Radiomics
Radiomics features are designed with the goal of providing off-the-shelf quantitative features describing tumors, computed on radiology imaging data. Such features have the pros of offering quantitative and reproducible analyses. A standard pipeline for analyzing radiology imaging data could be:
- Annotating the tumoral zone by a segmentation mask
- Extracting radiomics with a certified software / library
- Plugging those features into usual machine learning / survival regression models

Designing those features requires advanced medical knowledge, and a lot of experience in prognosis / medical models. Hence, a large number of informative features have been identified by groups of medical experts in radio-oncology, related to colour or density heterogeneity, compactness, and many others.

This challenge uses a subset of 53 features which have been identified as worth-to-be-looked-at for the proposed task.


## List of radiomics used in the challenge
The below lists each feature furnished for this challenge.  
Details over each computed feature can be found at https://pyradiomics.readthedocs.io/en/v1.0/radiomics.html#

### Shape
- 'Compactness1'
- 'Compactness2'
- 'Maximum3DDiameter'
- 'SphericalDisproportion'
- 'Sphericity'
- 'SurfaceArea'
- 'SurfaceVolumeRatio'
- 'VoxelVolume'

### Firstorder
- 'Energy'
- 'Entropy'
- 'Kurtosis'
- 'Maximum'
- 'Mean'
- 'MeanAbsoluteDeviation'
- 'Median'
- 'Minimum'
- 'Range'
- 'RootMeanSquared'
- 'Skewness'
- 'StandardDeviation'
- 'Uniformity'
- 'Variance'

### Glcm:
- 'Autocorrelation'
- 'ClusterProminence'
- 'ClusterShade'
- 'ClusterTendency'
- 'Contrast'
- 'Correlation'
- 'DifferenceEntropy'
- 'DifferenceAverage' #Dissimilarity
- 'JointEnergy' 
- 'JointEntropy'
- 'Id' #homogeneity1
- 'Idm' #homogeneity1
- 'Imc1'
- 'Imc2'
- 'Idmn'
- 'Idn'
- 'InverseVariance'
- 'MaximumProbability'
- 'SumAverage'
- 'SumEntropy'

### Glrlm:
- 'ShortRunEmphasis'
- 'LongRunEmphasis'
- 'GrayLevelNonUniformity'
- 'RunLengthNonUniformity'
- 'RunPercentage'
- 'LowGrayLevelRunEmphasis'
- 'HighGrayLevelRunEmphasis'
- 'ShortRunLowGrayLevelEmphasis'
- 'ShortRunHighGrayLevelEmphasis'
- 'LongRunLowGrayLevelEmphasis'
- 'LongRunHighGrayLevelEmphasis'
