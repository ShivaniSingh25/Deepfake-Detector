# Deepfake Detector Submission

## Project Summary
This project studies deepfake face detection using a multimodal detector with:
- RGB semantic features
- Wavelet frequency features
- SRM noise residual features
- Gated fusion
- Prompt-guided bridge
- Classification and weak localization

The strongest final detector was obtained by:
- keeping wavelet branch
- keeping SRM branch
- keeping bridge
- removing reasoning transformer
- keeping contrastive loss

## Final Main Detector
- Training dataset: FF++
- Checkpoint: `checkpoints/stage1_final_detector_no_reasoning.pth`
- Threshold: `0.29`

### Final FF++ Metrics
- Accuracy: 0.9188
- F1: 0.9368
- Precision: 0.9284
- Recall: 0.9453
- AUC: 0.9720
- EER: 0.0822

## Main Cross-Dataset Findings
- FF++ -> Celeb-DF: AUC 0.7852, EER 0.2657
- FF++ -> DFDC: AUC 0.6349, EER 0.4114
- FF++ + Celeb-DF -> DFDC: AUC 0.6941, EER 0.3644

This shows:
- strong in-domain performance
- weak single-source cross-dataset generalization
- multi-source training improves robustness

## Repository Structure
- `configs/` : training configs
- `datasets/` : dataset loaders
- `models/` : main detector code
- `modules/` : architecture building blocks
- `scripts/` : training / evaluation / inference scripts
- `utils/` : metrics, visualization, helpers
- `checkpoints/` : final saved model weights
- `results/` : final summaries and outputs
- `figures/` : images for report
- `report/` : final report files

## Notes
For localization, Grad-CAM is used as the main weak evidence-region visualization.
The learned explanation module was preliminary, so final explanations should be treated as constrained evidence-based rationales.
