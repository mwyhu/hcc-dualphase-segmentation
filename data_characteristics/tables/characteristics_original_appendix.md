**Characteristics of the original source datasets: supplementary characteristics**

| Characteristic | WAW-TACE | MSD | MCT-LTDiag | Liver Lesions |
| --- | --- | --- | --- | --- |
| Without annotated tumours, n (%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 27 (3.2%) |
| Unknown tumour status, n (%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| No valid CT phase, n (%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| ART slices per CT volume | 195 [171, 253] | — | 72 [60, 98] | 203 [185, 231] |
| PV slices per CT volume | 195 [171, 252] | 49 [41, 94] | 72 [60, 98] | 221 [186, 261] |
| ART voxel spacing, x (mm) | 0.78 [0.71, 0.86] | — | 0.74 [0.68, 0.78] | 0.73 [0.69, 0.78] |
| ART voxel spacing, y (mm) | 0.78 [0.71, 0.86] | — | 0.74 [0.68, 0.78] | 0.73 [0.69, 0.78] |
| PV voxel spacing, x (mm) | 0.78 [0.71, 0.86] | 0.80 [0.73, 0.86] | 0.74 [0.68, 0.78] | 0.76 [0.70, 0.82] |
| PV voxel spacing, y (mm) | 0.78 [0.71, 0.86] | 0.80 [0.73, 0.86] | 0.74 [0.68, 0.78] | 0.76 [0.70, 0.82] |
| ART mean tumour attenuation (HU) | 64.6 [54.1, 76.0] | — | 62.2 [52.8, 72.5] | 51.1 [33.4, 78.8] |
| PV mean tumour attenuation (HU) | 84.3 [71.8, 96.2] | 90.9 [75.7, 106.8] | 76.2 [64.2, 88.6] | 60.1 [45.8, 74.6] |

Values are n (%) or median [Q1, Q3]. Each case represents one patient.

Tumour presence refers to the reference annotations. Tumour-positive and tumour-negative percentages use patients with readable segmentations as the denominator; phase percentages use all cases.

Lesions are connected components in the reference masks. Individual lesion-volume summaries give each lesion equal weight. Total tumour volume gives each tumour-positive patient equal weight.

Zero-filled channels are treated as missing phases. Slice spacing is the z-axis voxel spacing and is not necessarily the acquisition slice thickness.

Tumour HU values summarise the per-patient mean over annotated tumour voxels. Missing or undefined measurements are shown as —.

These tables describe the raw datasets supplied to the extraction script, before any subsequent cohort exclusions.

Available n for ART slices per CT volume: WAW-TACE: 194; MSD: 0; MCT-LTDiag: 498; Liver Lesions: 49.

Available n for PV slices per CT volume: WAW-TACE: 197; MSD: 303; MCT-LTDiag: 498; Liver Lesions: 793.

Available n for ART voxel spacing, x (mm): WAW-TACE: 194; MSD: 0; MCT-LTDiag: 498; Liver Lesions: 49.

Available n for ART voxel spacing, y (mm): WAW-TACE: 194; MSD: 0; MCT-LTDiag: 498; Liver Lesions: 49.

Available n for PV voxel spacing, x (mm): WAW-TACE: 197; MSD: 303; MCT-LTDiag: 498; Liver Lesions: 793.

Available n for PV voxel spacing, y (mm): WAW-TACE: 197; MSD: 303; MCT-LTDiag: 498; Liver Lesions: 793.

Available n for ART mean tumour attenuation (HU): WAW-TACE: 194; MSD: 0; MCT-LTDiag: 498; Liver Lesions: 48.

Available n for PV mean tumour attenuation (HU): WAW-TACE: 197; MSD: 301; MCT-LTDiag: 498; Liver Lesions: 767.
