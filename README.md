# EEG-ImageNet-Dataset

This is the official repository for the paper "**EEG-ImageNet: An Electroencephalogram Dataset and Benchmarks with Image Visual Stimuli of Multi-Granularity Labels**".

<img width="776" alt="image" src="https://github.com/user-attachments/assets/55ac9916-e6ff-4f27-afbe-21a5d8206df2">

**Figure 1**: Schematic Diagram of the Data File Storage Structure. 

The dataset is available for download through the provided cloud storage [links](https://cloud.tsinghua.edu.cn/d/d812f7d1fc474b14bbd0/). 
Due to file size limitations on the cloud storage platform, the dataset is split into two parts: EEG-ImageNet_1.pth and EEG-ImageNet_2.pth. Each part contains data from 8 participants. Users can choose to use only one part based on their specific needs or device limitations.

The EEG-ImageNet dataset contains a total of 63,850 EEG-image pairs from 16 participants. 
Each EEG data sample has a size of (n\_channels, $f_s \cdot T$), where n\_channels is the number of EEG electrodes, which is 62 in our dataset; $f_s$ is the sampling frequency of the device, which is 1000 Hz in our dataset; and T is the time window size, which in our dataset is the duration of the image stimulus presentation, i.e., 0.5 seconds.
Due to ImageNet's copyright restrictions, our dataset only provides the file index of each image in ImageNet and the wnid of its category corresponding to each EEG segment.

<img width="921" alt="image" src="https://github.com/user-attachments/assets/a045a0ab-c53c-4536-90d3-aac3cb8cf256">

**Figure 2**: The overall procedure of our dataset construction and benchmark design. The experimental paradigm involves four stages: S1: Category Presentation (displaying the category label), S2: Fixation (500 ms), S3: Image Presentation (each image displayed for 500 ms), and S4: an optional random test to verify participant engagement. Each image presentation sequence includes 50 images from the given category, during which EEG signals are recorded. Data flow is indicated by blue arrows, while collected data is highlighted in gray. The stimuli images are sourced from ImageNet, with EEG signals aligned to image indices, granularity levels, and labels. The benchmarks (image reconstruction and object classification) are designed to evaluate coarse and fine granularities classification tasks.

**Table 1**: The average results of all participants in the object classification task. * indicates the use of time-domain features, otherwise it indicates the use of frequency-domain features. † indicates that the difference compared to the best-performing model is significant with p-value < 0.05.

| **Model**        |            | **Acc (all)** | **Acc (coarse)** | **Acc (fine)** |
|------------------|------------|---------------|------------------|----------------|
| **Classic model**| Ridge      | 0.286±0.074 †       | 0.394±0.081 †          | 0.583±0.074 †        |
|                  | KNN        | 0.304±0.086 †       | 0.401±0.097 †          | 0.696±0.068 †        |
|                  | RandomForest | 0.349±0.087 †     | 0.454±0.105 †          | 0.729±0.072 †        |
|                  | SVM        | **0.392±0.086**    | **0.506±0.099 †**      | **0.778±0.054 †**    |
| **Deep model**   | MLP        | 0.404±0.103        | **0.534±0.115**       | **0.816±0.054**     |
|                  | EEGNet*    | 0.260±0.098 †       | 0.303±0.108 †          | 0.365±0.095 †        |
|                  | RGNN       | **0.405±0.095**    | 0.470±0.092 †          | 0.706±0.073 †        |

<img width="776" alt="image" src="https://github.com/user-attachments/assets/026182bd-5b8d-4b84-aaca-a69ea7e2f0fa">

**Figure 3**: The image reconstruction results of a single participant (S8).


