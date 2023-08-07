# International Wheat Improvement Network (IWIN)

The International Wheat Improvement Network (IWIN), which involves hundreds of partners and testing sites worldwide, is coordinated by the International Maize and Wheat Improvement Center (CIMMYT). The IWIN has underpinned increases in wheat productivity in the developing world ever since the Green Revolution, and currently develops and disseminates approximately 1,000 new wheat lines each year, with well documented up-to-date genetic gains . In addition, IWIN germplasm is sought after by public and private entities in the developed world, where its impacts are also well documented.

## Best pedigrees in extreme climates

#### Identification of individual wheat lines that outperform under extreme weather events relative to average performance in international nurseries

Authors:
-    **Ernesto Giron Echeverry** (Independent Researcher)
-    **Urs Christoph Schulthess** (CIMMYT-China)
-    **Matthew Paul Reynolds** (CIMMYT)

Last update: Aug 5, 2023


# iPAR Yidel Model Package

This is a small package for estimating wheat yield. it estimates yield under non-stressed conditions as a function of iPAR, temperature and solar radiation.

Copyright (C) 2023 CIMMYT-Henan Collaborative Innovation Center

Authors: 
- Urs Christoph schulthess (CIMMYT-China)
- Ernesto Giron E. - e.giron.e@gmail.com (Independent Researcher, Cali, Colombia)



The following instructions allow you to carry out an analysis to find the best genotypes in low yielding and climate extreme environments.

It will demonstrate step by step the use of the python IWIN library.

## Install libraries


### Install dependencies

- !pip install numpy==1.22.4
- !pip install pandas==1.5.3
- !pip install scikit-learn==1.2.2
- !pip install scipy==1.10.1
- !pip install basemap==1.3.7
- !pip install basemap-data==1.3.2
- !pip install basemap-data-hires==1.3.2
- !pip install pyhull==2015.2.1
- !pip install hdbscan==0.8.33
- !pip install astral==3.2
- !pip install astropy==5.2.2

### Install IWIN package

pip install "git+https://github.com/egiron/IWIN@v2.0.1"


