**Characteristics of the external source datasets: supplementary characteristics**

| Characteristic | HCC-TACE-Seg | LiTS | TCIA-CRLM | WORC-CRLM | CAIRO5 |
| --- | --- | --- | --- | --- | --- |
| Without annotated tumours, n (%) | 0 (0.0%) | 13 (9.9%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| Unknown tumour status, n (%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 1 (4.3%) |
| No valid CT phase, n (%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| ART slices per CT volume | 83 [75, 97] | — | — | — | — |
| PV slices per CT volume | 83 [75, 97] | 432 [190, 684] | 55 [44, 142] | 116 [85, 176] | 685 [608, 768] |
| ART voxel spacing, x (mm) | 0.78 [0.70, 0.82] | — | — | — | — |
| ART voxel spacing, y (mm) | 0.78 [0.70, 0.82] | — | — | — | — |
| PV voxel spacing, x (mm) | 0.78 [0.70, 0.82] | 0.77 [0.70, 0.85] | 0.78 [0.73, 0.84] | 0.73 [0.68, 0.78] | 0.78 [0.73, 0.85] |
| PV voxel spacing, y (mm) | 0.78 [0.70, 0.82] | 0.77 [0.70, 0.85] | 0.78 [0.73, 0.84] | 0.73 [0.68, 0.78] | 0.78 [0.73, 0.85] |
| ART mean tumour attenuation (HU) | 72.9 [63.8, 81.1] | — | — | — | — |
| PV mean tumour attenuation (HU) | 95.5 [85.2, 111.3] | 65.9 [51.4, 79.1] | 85.2 [72.5, 97.7] | 81.0 [68.2, 92.9] | 56.0 [50.6, 61.5] |

Values are n (%) or median [Q1, Q3]. Each case represents one patient.
Tumour presence refers to the reference annotations. Tumour-positive and tumour-negative percentages use patients with readable segmentations as the denominator; phase percentages use all cases.
Lesions are connected components in the reference masks. Individual lesion-volume summaries give each lesion equal weight. Total tumour volume gives each tumour-positive patient equal weight.
Zero-filled channels are treated as missing phases. Slice spacing is the z-axis voxel spacing and is not necessarily the acquisition slice thickness.
Tumour HU values summarise the per-patient mean over annotated tumour voxels. Missing or undefined measurements are shown as —.
These tables describe the raw datasets supplied to the extraction script, before any subsequent cohort exclusions.
Available n for ART slices per CT volume: HCC-TACE-Seg: 82; LiTS: 0; TCIA-CRLM: 0; WORC-CRLM: 0; CAIRO5: 0.
Available n for PV slices per CT volume: HCC-TACE-Seg: 82; LiTS: 131; TCIA-CRLM: 159; WORC-CRLM: 76; CAIRO5: 23.
Available n for ART voxel spacing, x (mm): HCC-TACE-Seg: 82; LiTS: 0; TCIA-CRLM: 0; WORC-CRLM: 0; CAIRO5: 0.
Available n for ART voxel spacing, y (mm): HCC-TACE-Seg: 82; LiTS: 0; TCIA-CRLM: 0; WORC-CRLM: 0; CAIRO5: 0.
Available n for PV voxel spacing, x (mm): HCC-TACE-Seg: 82; LiTS: 131; TCIA-CRLM: 159; WORC-CRLM: 76; CAIRO5: 23.
Available n for PV voxel spacing, y (mm): HCC-TACE-Seg: 82; LiTS: 131; TCIA-CRLM: 159; WORC-CRLM: 76; CAIRO5: 23.
Available n for ART mean tumour attenuation (HU): HCC-TACE-Seg: 75; LiTS: 0; TCIA-CRLM: 0; WORC-CRLM: 0; CAIRO5: 0.
Available n for PV mean tumour attenuation (HU): HCC-TACE-Seg: 75; LiTS: 118; TCIA-CRLM: 159; WORC-CRLM: 76; CAIRO5: 18.
CAIRO5: 1 patients with unknown tumour status.