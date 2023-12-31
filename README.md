# AI_contest
- Monitoring Safety Equipment through a object detection model
- 객체 탐지 모델을 통한 공사 현장 안전 장구류 착용 모니터링 서비스
### Contents
1. Tool / Dataset
2. Training
3. demo
### 1. Tools / Dataset
`Dataset`
- AI Hub (공사현장 안전 장비 인식 이미지)
  - https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=163
![ai hub 그림](https://github.com/GluteusStrength/Capstone-Design/assets/48168432/585207bf-4781-4579-bdc2-5d1c4f58f5af)
- total 6000 images
- classes to detect(6 classes)
  - 안전 벨트 착용 여부(착용/미착용)
  - 안전화 착용 여부(착용/미착용)
  - 안전모 착용 여부(착용/미착용)
<br>

![classes](https://github.com/GluteusStrength/Capstone-Design/assets/48168432/8d191044-2e23-4f78-a7b6-1f2078fc02f4)

<br/>

`Tools`
<br>

<img src="https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=white"> <img src="https://img.shields.io/badge/pytorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"> <img src="https://img.shields.io/badge/openCV-5C3EE8?style=for-the-badge&logo=pytorch&logoColor=white"> <img src="https://img.shields.io/badge/streamlit-FF4B4B?style=for-the-badge&logo=pytorch&logoColor=white">

### 2. Training
#### Model - YOLO for read-time detection
<br>

![그림4](https://github.com/GluteusStrength/Capstone-Design/assets/48168432/b67113a8-f350-4b8c-a71a-0bf2596e9077)

#### Problem
1. Class Imbalance
![classimbalance](https://github.com/GluteusStrength/Capstone-Design/assets/48168432/a4109ec9-0dde-4b7c-8eac-fbbc89bc74d8)
   - `Solution`

     → Data Augmentation (객체 기반 Crop, Rotation(180°))
     
     ※ Reference
     ![image](https://github.com/GluteusStrength/AI_contest/assets/48168432/23b17372-26a7-46c3-befc-7824cd3c84b3)


     ![Augmentation jpg](https://github.com/GluteusStrength/Capstone-Design/assets/48168432/0cf3469f-0657-4629-9c94-0c6d0b990ee3)
     
     
2. Especially hard to detect **"Hard / No hard , Belt / No Belt"** classes
   - hard to detect Hard/No Hard classes(안전화 착용 여부 탐지) since safety shoes are ..
     1. small objects
     2. A small difference between safety shoes and sneakers
   <br>

   `Solution`

   → Not just resize Cropped images into 640 x 640. Apply **Super Resolution techniques** on cropped images. Our team thought that restoring cropped images into high resolution images by super resolution techniques can lead to a high-performance.

    → SRCNN(Super Resolution Convolutional Neural Network)
![image](https://github.com/GluteusStrength/AI_contest/assets/48168432/a463e42c-ccde-438d-bd0a-51cf83060901)

![superresolution](https://github.com/GluteusStrength/Capstone-Design/assets/48168432/2c77c017-dcee-47c4-ab8e-6f2201f5ffe3)

#### Result
![결과1](https://github.com/GluteusStrength/Capstone-Design/assets/48168432/1b6aa6ce-d445-4058-9b0b-f47a2a6d109c)
- 전반적인 mAP 결과
<br>

![결과2](https://github.com/GluteusStrength/Capstone-Design/assets/48168432/f53905b9-12ac-4886-b5ad-9e762b668cc4)
- 클래스 별 결과
