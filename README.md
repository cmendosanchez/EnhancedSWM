## Enhanced short fiber bundle segmentation

![fs](/img_seg/captura.png)

This repository contains the code used in: "Short fiber bundle filtering and test-retest reproducibility of the superficial white matter". Which enables the segmentation of well-defined Superficial White Matter (SWM) fiber bundles.

Four fiber bundle filters are available to remove spurious fibers from an automatic segmentation method [1] based on a SWM multi-subject atlas [2]. The filtering post-processing step helps to produce smoother bundles with less isolated fibers. Aditionally, the main fiber fascicle (MFF) identification helps to identify well-defined short fiber bundles. Available fiber bundle filters are based on:

- Connectivity Patterns.
- Symmetric Segment-Path Distance.
- Fiber Consistency.
- Convex Hull (Recommended)

The enhanced short fiber bundle segmentation can be done in two ways:
- Fiber bundle segmentation + Fiber bundle filter are applied.
- Fiber bundle segmentation + Fiber bundle filter are applied & Main fiber fascicles identification + Fiber bundle filter are applied.

Also, a folder with segmented fiber bundles can be provided to apply a fiber bundle filter with custom parameters.

## Dependencies

- C++ compiler
- Ubuntu 18.04.6 LTS
- OpenMP >= 4.5
- Python >= 3.6
- Dipy >= 1.4.1
- Numpy >= 1.21.5
- Scipy >= 1.7.3
- Scikit-learn >= 1.0.2
- Nibabel >= 3.2.2
- Joblib >= 1.1.0
  
## Use example
### Fiber bundle segmentation + Fiber bundle filter is applied.
```
python3 main.py --in_data input --extension extension --out_dir out_dir --filter filter_number 
```
* `--in_data` input tractogram in MNI space (tck/trk/bundles)
* `--extension` extension of tractogram (tck/trk/bundles)
* `--out_dir` name of the folder containing the segmented fiber bundles
* `--filter` selection of fiber bundle filter (1: Connectivity Pattern, 2: SSPD, 3: Fiber Consistency, 4: Convex Hull)
  
### Fiber bundle segmentation + Fiber bundle filter are applied & Main fiber fascicle identification + Fiber bundle filter are applied.
```
python3 main.py --in_data input --extension extension --out_dir out_dir --filter filter_number --MFF 1
```
* `--in_data` input tractogram in MNI space (tck/trk/bundles)
* `--extension` extension of tractogram (tck/trk/bundles)
* `--out_dir` name of the folder containing the output files
* `--filter` integer defining a fiber bundle filter  (1: Connectivity Pattern, 2: SSPD, 3: Fiber Consistency, 4: Convex Hull)
* `--MFF` apply main fiber fascicle identification and fiber bundle filtering. 1: True 0: False (default False)

#### Execution with example tractogram
An example tractogram in MNI space can be download from [here](https://drive.google.com/drive/folders/1p-aP8NzO2S3VezMRTGIudy5wwIoPhvuc?usp=sharing) (available in tck/trk/bundles format).

```
python3 main.py --in_data Data/Tractogram_MNI.tck --extension tck --out_dir Ex_seg/Seg --filter_number 4 --MFF 1
```

### Resulting folders
* `--out_dir` folder containing the segmented short fiber bundles
* `Filtered_bundles_${selected filter}_${format tck/trk/}` folder contatining the filtered short fiber bundles. By default a folder with .bundles format is generated for I/O operations.
* `MFF` folder containing the main fiber fascicles (if `--MFF 1`)
* `Filtered_MFF_${selected filter}` folder contatining the filtered main fiber fascicles (if `--MFF 1`)

If the indices of the fibers with respect to the orignal tractogram are needed, these are provided in the respective '_idx' folder. Resulting fiber bundles have 21 equidistand points.

### Fiber bundle filter is applied to a folder of fiber bundles with custom parameters.
```
python3 main.py --in_folder input_folder --extension --filter filter_number --p1 --p2
```
* `--in_folder` input folder with fiber bundles in MNI space (tck/trk/bundles) (bundles must be in MNI to perform data format conversion and I/O operations)
* `--extension` extension of the fiber bundles (tck/trk/bundles)
* `--filter` integer defining a fiber bundle filter  (1: Connectivity Pattern, 2: SSPD, 3: Fiber Consistency, 4: Convex Hull)
* `--p1` first parameter of the fiber bundle filter (percentage of discarded fibers)
* `--p2` second parameter of the fiber bundle filter (1: $\theta_{END}$, 2: $\theta_{SSPD}$ 3: $K_{f}$, 4: $K_{p}$)

#### Execution with example folder of short fiber bundles
An example folder with short fiber bundle can be download from [here](https://drive.google.com/drive/folders/1p-aP8NzO2S3VezMRTGIudy5wwIoPhvuc?usp=sharing) (available in tck/trk/bundles format):
```
python3 main.py --in_folder Bundles/Seg --extension tck --filter 4 --p1 20 --p2 80 
```
will apply the fiber bundle filter based on the Convex Hull, discarding 20% of the fibers and using $K_{p}=80$.

## References
[1] A. Vázquez, N. López-López, N. Labra, M. Figueroa, C. Poupon, J.-F. Mangin, C. Hernández,
and P. Guevara, “Parallel optimization of fiber bundle segmentation for massive tractography
datasets,” in 2019 IEEE 16th International Symposium on Biomedical Imaging (ISBI 2019).
IEEE, apr 2019.

[2] C. Román, C. Hernández, M. Figueroa, J. Houenou, C. Poupon, J.-F. Mangin, and P. Guevara,
“Superficial white matter bundle atlas based on hierarchical fiber clustering over probabilistic
tractography data,” NeuroImage, vol. 262, p. 119550, nov 2022.
