# snoop2head's portfolio

**Capture Questions, Answer with Code**

| Name          | Email                  | Blog                                                  |
| ------------- | ---------------------- | ----------------------------------------------------- |
| Young Jin Ahn | snoop2head@kaist.ac.kr | [snoop2head.github.io](https://snoop2head.github.io/) |

## 📖 Education

### Korea Advanced Institute of Science and Technology (KAIST)

<details>
<summary> Master of Science in Artificial Intelligence </summary>
  <ul>
    <li>Machine Learning for AI (A+)</li>
    <li>Large Language Models (A+)</li>
		<li>Programming for AI (A+)</li>
    <li>Deep Reinforcement Learning (A+)</li>
    <li>Machine Learning for Healthcare (A+)</li>
    <li>Advanced Machine Learning for AI (A0)</li>
    <li>Advanced Deep Learning (A-)</li>
    <li>Scientific Writing (P)</li>
  </ul>
</details>



### Yonsei University

<details>
<summary> Bachelor of Arts in Economics & Minor in Applied Statistics</summary>
  <ul>
    <li>INTRODUCTION TO STATISTICS (A0)</li>
    <li>STATISTICAL METHOD (A+)</li>
    <li>CALCULUS (B+)</li>
    <li>LINEAR ALGEBRA (B+)</li>
    <li>MATHEMATICAL STATISTICS 1 (A+)</li>
    <li>LINEAR REGRESSION (B+)</li>
    <li>R AND PYTHON PROGRAMMING (A+)</li>
    <li>DATA STRUCTURE (B+)</li>
    <li>SPECIAL PROBLEMS IN COMPUTING (A0)</li>
    <li>SOCIAL INFORMATICS (A+)</li>
    <li>TIME SERIES ANALYSIS (A+)</li>
    <li>THEORY AND PRACTICE OF DEEP LEARNING (A+)</li>
  </ul>
</details>


## ✍️ Publications

### [📄 Mixture of Monosemantic Experts for Transformers (Under Review in ICLR 2025) ](https://arxiv.org/abs/2412.04139)

[![arXiv](https://img.shields.io/badge/arXiv-2412.04139-b31b1b?style=flat-square)](https://arxiv.org/abs/2412.04139)
[![Models](https://img.shields.io/badge/%F0%9F%A4%97Hugging_Face-Model_Zoo-ffd200?style=flat-square)](https://huggingface.co/MonetLLM)
[![Demo](https://img.shields.io/badge/%F0%9F%A4%97Hugging_Face-Demo-ffd200?style=flat-square)](https://huggingface.co/spaces/MonetLLM/monet-vd-1.4B-100BT-hf-viewer)
[![code](https://img.shields.io/badge/Github-Code-keygen.svg?logo=github&style=flat-square)](https://github.com/dmis-lab/Monet)

[<img width="750" alt="image" src="./images/monet.png">](https://github.com/dmis-lab/Monet)

**Transparent Large Language Model via scaling the expert count to 262,144.**

- **Parameter-efficient architecture with increased number of experts**: By utilizing a novel expert decomposition method, Monet addresses memory constraints, ensuring that the total number of parameters scales proportionally to the square root of the number of experts.
- **Mechanistic interpretability via monosemantic experts**: Monet facilitates mechanistic interpretability by enabling observations of fine-grained experts’ routing patterns. Our analyses confirm mutual exclusivity of knowledge between groups of experts, while qualitative examples demonstrate individual experts’ parametric knowledge.
- **Robust knowledge manipulation without performance trade-offs**: Monet allows for end-to-end training that extends to robust knowledge manipulation during inference. Without degrading performance, it provides effortless control over knowledge domains, languages, and toxicity mitigation.

### [📄 SyncVSR: Data-Efficient Visual Speech Recognition with End-to-End Crossmodal Audio Token Synchronization (Interspeech 2024) ](https://github.com/KAIST-AILab/SyncVSR)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/syncvsr-data-efficient-visual-speech/lipreading-on-lip-reading-in-the-wild)](https://paperswithcode.com/sota/lipreading-on-lip-reading-in-the-wild?p=syncvsr-data-efficient-visual-speech)  
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/syncvsr-data-efficient-visual-speech/lipreading-on-lrw-1000)](https://paperswithcode.com/sota/lipreading-on-lrw-1000?p=syncvsr-data-efficient-visual-speech)  
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/syncvsr-data-efficient-visual-speech/lipreading-on-lrs2)](https://paperswithcode.com/sota/lipreading-on-lrs2?p=syncvsr-data-efficient-visual-speech)  
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/syncvsr-data-efficient-visual-speech/lipreading-on-lrs3-ted)](https://paperswithcode.com/sota/lipreading-on-lrs3-ted?p=syncvsr-data-efficient-visual-speech)  
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/syncvsr-data-efficient-visual-speech/landmark-based-lipreading-on-lrw)](https://paperswithcode.com/sota/landmark-based-lipreading-on-lrw?p=syncvsr-data-efficient-visual-speech)  
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/syncvsr-data-efficient-visual-speech/landmark-based-lipreading-on-lrs2)](https://paperswithcode.com/sota/landmark-based-lipreading-on-lrs2?p=syncvsr-data-efficient-visual-speech)  

|                     Overview of SyncVSR                      |                Performance of SyncVSR on LRS3                |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| [<img width="300" alt="image" src="./images/SyncVSR.png">](https://github.com/KAIST-AILab/SyncVSR) | [<img width="440" alt="image" src="./images/LRS3.png">](https://github.com/KAIST-AILab/SyncVSR) |

**Frame-level crossmodal supervision with quantized audio tokens for enhanced Visual Speech Recognition.**

- A prominent challenge in VSR is the presence of homophenes—visually similar lip gestures that represent different phonemes.
- Prior approaches have sought to distinguish fine-grained visemes by aligning visual and auditory semantics, but often fell short of full synchronization.
- Our proposed learning framework, SyncVSR, shows versatility across tasks, languages, and input modalities at the cost of a forward pass.
- By integrating a projection layer that synchronizes visual representation with acoustic data, the encoder learns to generate discrete audio tokens from a video sequence in a non-autoregressive manner.

---

## 🛠 Multimodal

### [<img width="18" alt="image" src="./images/GitHub-Mark/PNG/GitHub-Mark-Light-64px.png"> KoDALLE: Text to Fashion (2021)](https://github.com/KR-HappyFace/KoDALLE)

[<img width="700" alt="image" src="https://github.com/KR-HappyFace/KoDALLE/raw/main/assets/README/image-20211227151557604.png">](https://github.com/KR-HappyFace/KoDALLE)

**Generating dress outfit images based on given input text** | [📄 Presentation](https://github.com/KR-HappyFace/KoDALLE/blob/main/README.pdf)

- **Created training pipeline from VQGAN through DALLE**
- **Maintained versions of 1 million pairs image-caption dataset.**
- Trained VQGAN and DALLE model from the scratch.
- Established live demo for the KoDALLE on Huggingface Space via FastAPI.

## 🏆 Competitions

|              Host / Platform               |                                                                    Topic / Task                                                                     |        Result        |                                                                                Repository                                                                                 | Year |
| :----------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------: | :------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :--: |
| National IT Industry<br />Promotion Agency |                           [Machine Reading Compehension](https://aichallenge.or.kr/competition/detail/1/task/5/taskInfo)                            | 🥈 2nd <br />(2/26)  |             [<img width="18" alt="image" src="./images/GitHub-Mark/PNG/GitHub-Mark-Light-64px.png"> MRC_Baseline](https://github.com/QuoQA-NLP/MRC_Baseline)              | 2022 |
|           Ministry of Statistics           | [Korean Standard Industry Classification](https://data.kostat.go.kr/sbchome/bbs/boardList.do?boardId=SBCSBBS_000000025000&curMenuNo=OPT_09_02_00_0) | 🎖 7th <br />(7/311)  |                                                                                     -                                                                                     | 2022 |
|                   Dacon                    |                   [KLUE benchmark Natural Language Inference](https://dacon.io/competitions/official/235875/overview/description)                   | 🥇 1st <br />(1/468) |                                                [🌐 KLUE NLI](https://dacon.io/competitions/official/235875/codeshare/4589)                                                | 2022 |
|              Dacon & AI Frenz              |                          [Python Code Clone Detection](https://dacon.io/competitions/official/235900/overview/description)                          | 🥉 3rd <br />(3/337) |           [<img width="18" alt="image" src="./images/GitHub-Mark/PNG/GitHub-Mark-Light-64px.png"> CloneDetection](https://github.com/sangHa0411/CloneDetection)           | 2022 |
|             Dacon & CCEI Korea             |                    [Stock Price Forecast on KOSPI & KOSDAQ](https://dacon.io/competitions/official/235857/overview/description)                     | 🎖 6th <br />(6/205)  | [<img width="18" alt="image" src="./images/GitHub-Mark/PNG/GitHub-Mark-Light-64px.png"> elastic-stock-prediction](https://github.com/snoop2head/elastic-stock-prediction) | 2021 |

\*\*Dacon is Kaggle alike competition platform in Korea.

## 🔐 Differential Privacy

### [<img width="18" alt="image" src="./images/GitHub-Mark/PNG/GitHub-Mark-Light-64px.png"> Language Model Memorization (2022)](https://github.com/yonsei-cysec/Language_Model_Memorization)

**Implementation of Carlini et al(2020) [Extracting Training Data from Large Language Models](https://arxiv.org/abs/2012.07805)**

- Accelerated inference speed with parallel Multi-GPU usage.
- Ruled out 'low-quality repeated generations' problem of the paper with repetition penalty and with ngram restriction.

### [<img width="18" alt="image" src="./images/GitHub-Mark/PNG/GitHub-Mark-Light-64px.png"> Membership Inference Attack (2022)](https://github.com/yonsei-cysec/Membership_Inference_Attack)

**Implementation of Shokri et al(2016) [Membership Inference Attacks Against Machine Learning Models](https://arxiv.org/pdf/1610.05820.pdf)**

- Prevented overfitting of shadow models' by adding early stop, regularizing with weight decay and allocating train/val/test datasets.
- Referenced [Carlini et al(2021)](https://arxiv.org/abs/2112.03570) to conduct further research on different types of models and metrics.
- Reproduced attack metrics as the following.

| MIA Attack Metrics | Accuracy | Precision | Recall | F1 Score |
| :----------------: | :------: | :-------: | :----: | :------: |
|      CIFAR10       |  0.7761  |  0.7593   | 0.8071 |  0.7825  |
|      CIFAR100      |  0.9746  |  0.9627   | 0.9875 |  0.9749  |

|                                                   MIA ROC Curve CIFAR10                                                    |                                                    MIA ROC Curve CIFAR100                                                    |
| :------------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------: |
| ![roc_curve CIFAR10](https://raw.githubusercontent.com/snoop2head/Membership_Inference_Attack/main/assets/roc_cifar10.png) | ![roc_curve CIFAR100](https://raw.githubusercontent.com/snoop2head/Membership_Inference_Attack/main/assets/roc_cifar100.png) |

## 💬 Natural Language Processing

### [<img width="18" alt="image" src="./images/GitHub-Mark/PNG/GitHub-Mark-Light-64px.png"> KoQuillBot (2022)](https://github.com/QuoQA-NLP/KoQuillBot) & [<img width="18" alt="image" src="./images/GitHub-Mark/PNG/GitHub-Mark-Light-64px.png"> T5 Translation (2022)](https://github.com/QuoQA-NLP/T5_Translation)

**Paraphrasing tool with round trip translation utilizing T5 Machine Translation.** | [🤗 KoQuillBot Demo](https://huggingface.co/spaces/QuoQA-NLP/KoQuillBot) & [🤗 Translator Demo](https://huggingface.co/spaces/QuoQA-NLP/QuoQaGo)

|                   | BLEU Score |                                         Translation Result                                         |
| :---------------: | :--------: | :------------------------------------------------------------------------------------------------: |
| Korean ➡️ English |   45.15    | [🔗 Inference Result](https://huggingface.co/datasets/QuoQA-NLP/KE-T5-Ko2En-Base-Inference-Result) |
| English ➡️ Korean |     -      |                                                 -                                                  |

### [<img width="18" alt="image" src="./images/GitHub-Mark/PNG/GitHub-Mark-Light-64px.png"> Deep Encoder Shallow Decoder (2022)](https://github.com/snoop2head/Deep-Encoder-Shallow-Decoder)

**Implementation of Kasai et al(2020) [Deep Encoder, Shallow Decoder: Reevaluating Non-autoregressive Machine Translation](https://arxiv.org/abs/2006.10369)** | [📄 Translation Output](https://docs.google.com/spreadsheets/d/1IqEuRuEpphPEX3ni1m0EwqYuOU4E4t4-jC6uullpJhE/edit#gid=204599913)

- Composed custom dataset, trainer, inference code in pytorch and huggingface.
- Trained and hosted encoder-decoder transformers model using huggingface.

|                   | BLEU Score |                                                      Translation Result                                                       |
| :---------------: | :--------: | :---------------------------------------------------------------------------------------------------------------------------: |
| Korean ➡️ English |   35.82    | [🔗 Inference Result](https://docs.google.com/spreadsheets/d/1IqEuRuEpphPEX3ni1m0EwqYuOU4E4t4-jC6uullpJhE/edit#gid=204599913) |
| English ➡️ Korean |     -      |                                                               -                                                               |

### [<img width="18" alt="image" src="./images/GitHub-Mark/PNG/GitHub-Mark-Light-64px.png"> KLUE-RBERT (2021)](https://github.com/snoop2head/KLUE-RBERT)

**Extracting relations between subject and object entity in KLUE Benchmark dataset** | [✍️ Blog Post](https://snoop2head.github.io/Engineering/Relation-Extraction-Code/)

- Finetuned RoBERTa model according to RBERT structure in pytorch.
- Applied stratified k-fold cross validation for the custom trainer.

### [<img width="18" alt="image" src="./images/GitHub-Mark/PNG/GitHub-Mark-Light-64px.png"> Conditional Generation with KoGPT (2021)](https://github.com/snoop2head/KoGPT-Joong-2)

**Sentence generation with given emotion conditions** | [🤗 Huggingface Demo](https://huggingface.co/spaces/snoop2head/KoGPT-Conditional-Generation)

- Finetuned KoGPT-Trinity with conditional emotion labels.
- Maintained huggingface hosted model and live demo.

### [<img width="18" alt="image" src="./images/GitHub-Mark/PNG/GitHub-Mark-Light-64px.png"> Machine Reading Comprehension in Naver Boostcamp (2021)](https://github.com/boostcampaitech2/mrc-level2-nlp-15)

**Retrieved and extracted answers from wikipedia texts for given question** | [✍️ Blog Post](https://snoop2head.github.io/Engineering/Custom-MRC-Reader/)

- Attached bidirectional LSTM layers to the backbone transformers model to extract answers.
- Divided benchmark into start token prediction accuracy and end token prediction accuracy.

### [<img width="18" alt="image" src="./images/GitHub-Mark/PNG/GitHub-Mark-Light-64px.png"> Mathpresso Corporation Joint Project (2020)](https://github.com/snoop2head/Mathpresso_Classification)

**Corporate joint project for mathematics problems classification task** | [📄 Presentation](https://github.com/snoop2head/Mathpresso_Classification/blob/main/YBIGTA_%EB%A7%A4%EC%93%B0%ED%94%84%EB%A0%88%EC%86%8C_%EB%AA%BD%EB%8D%B0%EC%9D%B4%ED%81%AC_Final.pdf)

- Preprocessed Korean mathematics problems dataset based on EDA.
- Maintained version of preprocessing module.

### [<img width="18" alt="image" src="./images/GitHub-Mark/PNG/GitHub-Mark-Light-64px.png"> Constructing Emotional Instagram Posts Dataset (2019)](https://github.com/Keracorn/geulstagram)

**Created Emotional Instagram Posts(글스타그램) dataset** | [📄 Presentation](https://github.com/Keracorn/geulstagram/blob/master/README.pdf)

- Managed version control for the project Github Repository.
- Converted Korean texts on image file into text file using Google Cloud Vision API.

## 👀 Computer Vision

### [<img width="18" alt="image" src="./images/GitHub-Mark/PNG/GitHub-Mark-Light-64px.png"> DotNeuralNet (2023)](https://github.com/snoop2head/DotNeuralNet)

**Light-weight Neural Network for Optical Braille Recognition in the wild & on the book.** | [🤗 Huggingface Demo](https://huggingface.co/spaces/snoop2head/braille-detection)

![yolov8 img](https://github.com/snoop2head/DotNeuralNet/raw/main/assets/result_yolov8.png)

- Classified multi label one-hot encoded labels for raised braille patterns.
- Pseudo-labeled Natural Scene Braille dataset.
- Trained single stage object detection YOLO models for braille cell recognition.

### [<img width="18" alt="image" src="./images/GitHub-Mark/PNG/GitHub-Mark-Light-64px.png"> ElimNet (2021)](https://github.com/snoop2head/ELimNet)

**Elimination based Lightweight Neural Net with Pretrained Weights** | [📄 Presentation](https://github.com/snoop2head/ELimNet/blob/main/README.pdf)

- Constructed lightweight CNN model with less than 1M #params by removing top layers from pretrained CNN models.
- Assessed on Trash Annotations in Context(TACO) Dataset sampled for 6 classes with 20,851 images.
- Compared metrics accross VGG11, MobileNetV3 and EfficientNetB0.

### [<img width="18" alt="image" src="./images/GitHub-Mark/PNG/GitHub-Mark-Light-64px.png"> Face Mask, Age, Gender Classification in Naver Boostcamp (2021)](https://github.com/boostcampaitech2/image-classification-level1-23)

**Identifying 18 classes from given images: Age Range(3 classes), Biological Sex(2 classes), Face Mask(3 classes)** | [✍️ Blog Post](https://snoop2head.github.io/Engineering/Mask-Age-Gender-Classification-Competition/)

- Optimized combination of backbone models, losses and optimizers.
- Created additional dataset with labels(age, sex, mask) to resolve class imbalance.
- Cropped facial characteristics with MTCNN and RetinaFace to reduce noise in the image.

### [<img width="18" alt="image" src="./images/GitHub-Mark/PNG/GitHub-Mark-Light-64px.png"> Realtime Desktop Posture Classification (2020)](https://github.com/snoop2head/ml_classification_tutorial)

**Real-time desk posture classification through webcam** | [📷 Demo Video](https://www.youtube.com/watch?v=6z_TJaj71io&t=459s)

- Created real-time detection window using opencv-python.
- Converted image dataset into Yaw/Pitch/Roll numerical dataset using RetinaFace model.
- Trained and optimized random forest classification model with precision rate of 93%.

## 🕸 Web

### [<img width="18" alt="image" src="./images/GitHub-Mark/PNG/GitHub-Mark-Light-64px.png"> Exchange Program Overview Website (2020)](https://github.com/snoop2head/yonsei-exchange-program)

**Overview for student life in foreign universities** | [✈️ Website Demo](https://yonsei-exchange.netlify.app/)

- **3400 Visitors within a year (2021.07 ~ 2022.07)**
- **22000 Pageviews within a year (2021.07 ~ 2022.07)**
- **3 minutes+ of Average Retention Time**

<img height="300" width="200" alt="image" src="./images/yonsei_exchange1.png"><img height="300" width="250" alt="image" src="./images/yonsei_exchange2.png">

- Collected and preprocessed 11200 text review data from the Yonsei website using pandas.
- Visualized department distribution and weather information using matplotlib.
- Sentiment analysis on satisfaction level for foreign universities with pretrained BERT model.
- Clustered universities with provided curriculum with K-means clustering.
- Hosted reports on universities using Gatsby.js, GraphQL, and Netlify.

## 💰 Quantitative Finance

### [<img width="18" alt="image" src="./images/GitHub-Mark/PNG/GitHub-Mark-Light-64px.png"> Forecasting Federal Rate with Lasso Regression Model (2022)](https://github.com/snoop2head/Federal-Rate-Prediction)

**Federal Rate Prediction for the next FOMC Meeting**

- Wrangled quantitative dataset with Finance Data Reader.
- Yielded metrics and compared candidate regression models for the adaquate fit.
- Hyperparameter optimization for the candidate models.

### [<img width="18" alt="image" src="./images/GitHub-Mark/PNG/GitHub-Mark-Light-64px.png"> Korean Spinoff Event Tracker (2020)](https://github.com/snoop2head/spinoff_hunter_kor)

**Get financial data of public companies involved in spinoff events on Google Spreadsheet** | [🧩 Dataset Demo](https://docs.google.com/spreadsheets/d/1chJ2NKHVc0gKjsMaQI1UHEPxdjneV1ZWaTGHseQvxP4/edit?usp=sharing)

- Wrangled finance dataset which are displayed on Google Sheets

## 🏷 Opensource Contributions

### [<img width="18" alt="image" src="./images/GitHub-Mark/PNG/GitHub-Mark-Light-64px.png"> NVlabs/stylegan2-ada-pytorch (2021)](https://github.com/NVlabs/stylegan2-ada-pytorch)

**Fixed torch version comparison fallback error for source repo of NVIDIA Research** | [✍️ Pull Request](https://github.com/NVlabs/stylegan2-ada-pytorch/pull/197)

- Skills: torch, torchvision

### [<img width="18" alt="image" src="./images/GitHub-Mark/PNG/GitHub-Mark-Light-64px.png"> docker/docker.github.io (2020)](https://github.com/docker/docker.github.io)

**Updated PostgreSQL initialization for "Quickstart: dockerizing django" documentation** | [✍️ Pull Request](https://github.com/docker/docker.github.io/pull/10624)

- Skills: Docker, docker-compose, Django
