**Characteristics of the external source datasets**

| Characteristic | HCC-TACE-Seg | LiTS | TCIA-CRLM | WORC-CRLM | CAIRO5 |
| --- | --- | --- | --- | --- | --- |
| Patients, n | 82 | 131 | 159 | 76 | 23 |
| With annotated tumours, n (%) | 82 (100.0%) | 118 (90.1%) | 159 (100.0%) | 76 (100.0%) | 22 (100.0%) |
| ART only, n (%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| PV only, n (%) | 0 (0.0%) | 131 (100.0%) | 159 (100.0%) | 76 (100.0%) | 23 (100.0%) |
| ART and PV, n (%) | 82 (100.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| Total annotated lesions, n | 127 | 908 | 419 | 92 | 418 |
| Lesions per tumour-positive patient | 1.0 [1.0, 1.0] | 4.5 [1.0, 10.0] | 2.0 [1.0, 3.0] | 1.0 [1.0, 1.0] | 15.5 [8.0, 29.0] |
| Individual lesion volume (mL) | 41.77 [3.48, 220.52] | 0.54 [0.15, 2.73] | 1.62 [0.48, 6.80] | 6.97 [2.65, 21.79] | 0.36 [0.01, 1.96] |
| Total tumour volume per tumour-positive patient (mL) | 138.34 [41.58, 366.17] | 16.56 [3.55, 108.30] | 9.10 [3.76, 28.18] | 8.44 [4.73, 25.38] | 60.89 [33.94, 138.31] |
| ART slice spacing (mm) | 2.50 [2.50, 2.50] | — | — | — | — |
| PV slice spacing (mm) | 2.50 [2.50, 2.50] | 1.00 [0.80, 1.75] | 5.00 [1.50, 5.00] | 5.00 [3.00, 5.00] | 1.00 [0.80, 1.00] |

Values are n (%) or median [Q1, Q3]. Each case represents one patient.
Tumour presence refers to the reference annotations. Tumour-positive and tumour-negative percentages use patients with readable segmentations as the denominator; phase percentages use all cases.
Lesions are connected components in the reference masks. Individual lesion-volume summaries give each lesion equal weight. Total tumour volume gives each tumour-positive patient equal weight.
Zero-filled channels are treated as missing phases. Slice spacing is the z-axis voxel spacing and is not necessarily the acquisition slice thickness.
Tumour HU values summarise the per-patient mean over annotated tumour voxels. Missing or undefined measurements are shown as —.
These tables describe the raw datasets supplied to the extraction script, before any subsequent cohort exclusions.
Available n for Lesions per tumour-positive patient: HCC-TACE-Seg: 82; LiTS: 118; TCIA-CRLM: 159; WORC-CRLM: 76; CAIRO5: 22.
Available n for Individual lesion volume (mL): HCC-TACE-Seg: 127; LiTS: 908; TCIA-CRLM: 419; WORC-CRLM: 92; CAIRO5: 418.
Available n for Total tumour volume per tumour-positive patient (mL): HCC-TACE-Seg: 82; LiTS: 118; TCIA-CRLM: 159; WORC-CRLM: 76; CAIRO5: 22.
Available n for ART slice spacing (mm): HCC-TACE-Seg: 82; LiTS: 0; TCIA-CRLM: 0; WORC-CRLM: 0; CAIRO5: 0.
Available n for PV slice spacing (mm): HCC-TACE-Seg: 82; LiTS: 131; TCIA-CRLM: 159; WORC-CRLM: 76; CAIRO5: 23.
CAIRO5: 1 patients with unknown tumour status.