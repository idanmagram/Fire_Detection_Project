# Fire_Detection_Project
Implementation of a detection tool of fire using transfer leaning and Segment Anything Model

# Background
Wildfires are becoming more frequent and severe due to climate change, threatening ecosystems and communities. Traditional methods for detecting and managing fires are often slow and inaccurate, making it hard to control fires quickly. In this project, we used Computer vision and AI to improve wildfire detection and management by applying image segmentation techniques. Using transfer learning for fire image classification and Meta’s "Segment Anything" model for fire segmentation, we achieved a 95% accuracy rate. This helps identify the most dangerous fire areas, allowing resources to be directed where they are needed most. Our approach can reduce response times and minimize the damage caused by wildfires.

# Prerequisites
| Library    | Version |
|------------|---------|
| Python     | 3.5.5   |
| torch      | 2.3.1   |
| torchvision| 0.18.1  |
| kornia     | 0.7.3   |
| numpy      | 1.26.4  |
| matplotlib | 3.7.1   |
| tqdm       | 4.66.5  |
| seaborn    | 0.13.1  |

## Files in the Repository

| File                        | Description                                                                                     |
|-----------------------------|-------------------------------------------------------------------------------------------------|
| `fire_classification.ipynb`  | Notebook containing fire classifier training using transfer learning with ResNet18 and EfficientNet + results and graphs. |
| `sam.py`                     | Implementation of the Segment Anything Model (SAM) by Meta, adapted for fire segmentation.      |
| `fire_segmentation.py`       | Script to segment fire using the fire classifier and SAM model.                                 |

# Dataset
The [dataset](https://www.kaggle.com/datasets/mohnishsaiprasad/forest-fire-images) contains ~5000 images classified as "Fire" and "no Fire"

# Fire Detection Method Overview

![image](https://github.com/user-attachments/assets/7f811ca3-d333-4440-99cf-791df159f497)

# Saved Models

Saved trained models can be found [here](https://drive.google.com/drive/folders/12Cv1x6MlOM6n0uYE56zHQu-ssfLySUYf)

# Results
![image](https://github.com/user-attachments/assets/5ffb6982-cc92-45bf-9b0f-68b2dfdd2fa9)

# References
[fire dataset at Kaggle website](https://www.kaggle.com/datasets/mohnishsaiprasad/forest-fire-images)

[Segment Anything code by Meta](https://github.com/facebookresearch/segment-anything)



