
# Pneumonia Detection from Chest X-Rays
Deep learning system that detects pneumonia from chest X-ray images using a fine-tuned ResNet-18 model in PyTorch.

## Overview
This project addresses the challenge of accurately identifying pneumonia from chest X-rays, with a focus on minimizing false negatives in a medical screening context. The model is optimized for high recall while maintaining strong overall performance.

## How It Works
**Model Pipeline:**

1. Image Input → Chest X-ray resized to 224×224 and normalized  
2. Feature Extraction → Pretrained ResNet-18 backbone  
3. Fine-Tuning → Unfreeze layer4 and classifier head  
4. Classification → Binary output (Normal vs Pneumonia)  
5. Thresholding → Probability-based decision rule (0.7)
   
## Key Technical Decisions:

- **Pretrained Transfer Learning:** Leverages ImageNet weights to improve generalization on limited medical data  
- **Selective Layer Freezing:** Reduces overfitting and training time  
- **Class-Weighted Loss:** Addresses dataset imbalance between Normal and Pneumonia cases  
- **F1-Optimized Checkpointing:** Model selection based on validation F1 score  
- **Threshold Adjustment:** Prioritizes recall to reduce false negatives

## Evaluation
**Validation Set:**
- Accuracy: 93.75%
- Precision: 88.89%
- Recall: 100.00%
- F1 Score: 94.12%
- ROC-AUC: 100.00%

**Test Set:**
- Accuracy: 91.35%
- Precision: 89.44%
- Recall: 97.69%
- F1 Score: 93.38%
- ROC-AUC: 96.82%

<img width="600" height="500" alt="TestDataConfusionMatrix" src="https://github.com/user-attachments/assets/6c8773db-8ccb-436c-b04c-94d9f17aea78" />

## Tech Stack
**Deep Learning:** PyTorch, Torchvision  
**Data Processing:** NumPy, Pillow  
**Evaluation:** Scikit-learn  
**Visualization:** Matplotlib, Seaborn

## Contact
Vivaan Motwani  
https://github.com/vivaanm867

