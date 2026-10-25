**Characteristics of the original source datasets**

| Characteristic | WAW-TACE | MSD | MCT-LTDiag | Liver Lesions |
| --- | --- | --- | --- | --- |
| Patients, n | 197 | 303 | 498 | 842 |
| With annotated tumours, n (%) | 197 (100.0%) | 303 (100.0%) | 498 (100.0%) | 815 (96.8%) |
| ART only, n (%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 49 (5.8%) |
| PV only, n (%) | 3 (1.5%) | 303 (100.0%) | 0 (0.0%) | 793 (94.2%) |
| ART and PV, n (%) | 194 (98.5%) | 0 (0.0%) | 498 (100.0%) | 0 (0.0%) |
| Total annotated lesions, n | 311 | 544 | 2,192 | 8,017 |
| Lesions per tumour-positive patient | 1.0 [1.0, 2.0] | 1.0 [1.0, 2.0] | 2.0 [1.2, 4.8] | 5.0 [2.0, 11.0] |
| Individual lesion volume (mL) | 16.05 [4.79, 60.95] | 5.82 [0.94, 38.15] | 0.69 [0.06, 5.27] | 0.29 [0.07, 1.35] |
| Total tumour volume per tumour-positive patient (mL) | 50.79 [18.59, 108.55] | 32.28 [8.44, 104.43] | 46.54 [10.96, 174.72] | 16.40 [2.58, 109.52] |
| ART slice spacing (mm) | 1.25 [1.25, 2.50] | — | 5.00 [5.00, 5.00] | 1.00 [1.00, 1.00] |
| PV slice spacing (mm) | 1.25 [1.25, 2.50] | 5.00 [2.50, 5.00] | 5.00 [5.00, 5.00] | 1.00 [1.00, 1.00] |

Values are n (%) or median [Q1, Q3]. Each case represents one patient.

Tumour presence refers to the reference annotations. Tumour-positive and tumour-negative percentages use patients with readable segmentations as the denominator; phase percentages use all cases.

Lesions are connected components in the reference masks. Individual lesion-volume summaries give each lesion equal weight. Total tumour volume gives each tumour-positive patient equal weight.

Zero-filled channels are treated as missing phases. Slice spacing is the z-axis voxel spacing and is not necessarily the acquisition slice thickness.

Tumour HU values summarise the per-patient mean over annotated tumour voxels. Missing or undefined measurements are shown as —.

These tables describe the raw datasets supplied to the extraction script, before any subsequent cohort exclusions.

Available n for Lesions per tumour-positive patient: WAW-TACE: 197; MSD: 303; MCT-LTDiag: 498; Liver Lesions: 815.

Available n for Individual lesion volume (mL): WAW-TACE: 311; MSD: 544; MCT-LTDiag: 2192; Liver Lesions: 8017.

Available n for Total tumour volume per tumour-positive patient (mL): WAW-TACE: 197; MSD: 303; MCT-LTDiag: 498; Liver Lesions: 815.

Available n for ART slice spacing (mm): WAW-TACE: 194; MSD: 0; MCT-LTDiag: 498; Liver Lesions: 49.

Available n for PV slice spacing (mm): WAW-TACE: 197; MSD: 303; MCT-LTDiag: 498; Liver Lesions: 793.
