\documentclass[conference]{IEEEtran}
\IEEEoverridecommandlockouts
\usepackage{cite}
\usepackage{amsmath}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{algorithmic}
\usepackage{graphicx}
\usepackage{textcomp}
\usepackage{xcolor}
\usepackage{booktabs}
\usepackage{url}
\usepackage{fancyhdr}
\usepackage{eso-pic}
\usepackage{array} % put this in the preamble
\def\BibTeX{{\rm B\kern-.05em{\sc i\kern-.025em b}\kern-.08em
    T\kern-.1667em\lower.7ex\hbox{E}\kern-.125emX}}

\makeatletter

% \def\mycopyrightnotice{
%   {\footnotesize 979-8-3315-5525-2/25/$31.00 ©2025 IEEE \copyright~2025 IEEE\hfill} % <--- Change here
%   \gdef\mycopyrightnotice{}
% }

\@ifundefined{showcaptionsetup}{}{
\PassOptionsToPackage{caption=false}{subfig}}
\usepackage{subfig}
\makeatother

\newcommand\AtPageUpperMyright[1]{\AtPageUpperLeft{
 \put(\LenToUnit{0.5\paperwidth},\LenToUnit{-1cm}){
     \parbox{0.5\textwidth}{\raggedleft\fontsize{9}{11}\selectfont #1}}
 }}
\newcommand{\conf}[1]{
\AddToShipoutPictureBG*{
\AtPageUpperMyright{#1}
}
}
\usepackage{fancyhdr}

\lhead{\rmfamily\fontsize{9}{11}\selectfont 
2025 IEEE International Conference on Biomedical Engineering, Computer and Information Technology for Health (BECITHCON)\\
29-30 November 2025, Eastern University, Dhaka, Bangladesh}
\renewcommand{\headrulewidth}{0pt}

\lfoot{\rmfamily\fontsize{9}{0}\selectfont   979-8-3315-6105-5/25/\$31.00 $\copyright$2025 IEEE}

\begin{document}
% Call overlay BEFORE \maketitle (safe; doesn't push content)

% \title{Explainable CNN and 3D-CNN Models for the Classification of AI-Generated Images: An Evaluation on the CIFAKE Dataset Using LIME and Grad-CAM}
\title{CIFAKE: Explainable Deep Learning for Classifying Real and AI-Generated Images Using CNN and 3D-CNN}

\author{
\small
Md. Hamid Hosen\textsuperscript{1}, 
Mikdad Mohammad Asif\textsuperscript{2}, 
Altaf Uddin\textsuperscript{3}, 
Rituparna Chowdhury\textsuperscript{4}, 
Pappuraj Bhottacharjee\textsuperscript{5}, 
Arnob Saha\textsuperscript{6}\\
\textsuperscript{1,2,3,4,6}\textit{Department of Computer Science \& Engineering, East Delta University, Chittagong, Bangladesh}\\
\textsuperscript{5}\textit{Department of Computer Science \& Engineering, Dhaka University of Engineering \& Technology, Gazipur, Bangladesh}\\
Email - mdhamidhosen4@gmail.com, asifmikdad@gmail.com, altafuddinsanim@gmail.com,\\ 
rituparna.chy550@gmail.com, pappuraj.duet@gmail.com, sahaarnob73@gmail.com
}

\maketitle
\thispagestyle{fancy}
\fancyfoot[C]{}

\begin{abstract}
Artificial intelligence has reached a level where distinguishing real images from synthetic ones is increasingly difficult. This creates risks in areas such as misinformation, digital security, and content authenticity. In this study, we address this problem by applying deep learning to the CIFAKE dataset, which contains a balanced collection of real and AI-generated images. Two models were developed: a Convolutional Neural Network (CNN) for extracting spatial features, and a three-dimensional CNN (3D-CNN) for capturing spatiotemporal patterns. The dataset was processed with augmentation and preprocessing to improve model generalization. Both models achieved strong results. The CNN obtained 95.69\% accuracy, 93.67\% precision, 98.00\% recall, and an F1-score of 95.79\%. The 3D-CNN outperformed it, achieving 96.62\% accuracy, 95.97\% precision, 97.33\% recall, and an F1-score of 96.64\%. To improve interpretability, explainable AI methods were applied. LIME provided local feature explanations, while Grad-CAM produced visual heatmaps of the most influential regions in the images. Together, these methods not only improved detection accuracy but also added transparency to the decision-making process. The results highlight the importance of combining robust classification models with explainable techniques for reliable detection of AI-generated images.
\end{abstract}




\begin{IEEEkeywords}
CNN, 3D-CNN, CIFAKE, image classification, explainable AI, LIME, Grad-CAM.
\end{IEEEkeywords}

\section{Introduction}
Artificial Intelligence (AI) has significantly changed the way images are created, making it increasingly difficult to distinguish real photos from machine-generated ones. Traditional camera images usually contain natural visual cues such as consistent lighting, realistic textures, and proper spatial depth. With the rise of GANs and transformer-driven image generators, synthetic visuals have reached a level of realism where they often appear indistinguishable from genuine photographs \cite{b1} \cite{b2}. Although this progress enables new applications in digital design, simulations, and media production, it also raises concerns about the authenticity of visual content.

This development introduces ethical challenges, especially regarding the credibility of digital media. As AI-generated visuals become more convincing, the need for accurate detection systems has grown because of increasing risks of manipulation and disinformation \cite{b3}. Images produced through GAN models and tools such as StyleGAN and Stable Diffusion present major identification difficulties. These systems can replicate real textures, lighting, and structural details so well that traditional pixel-level detection methods often fail \cite{b4}. The easy availability of generative tools allows non-experts to produce believable synthetic images, increasing both the volume and variety of fake content and making detection more difficult. To address these issues, machine learning (ML) and deep learning (DL) techniques are widely used. Convolutional neural networks (CNNs) are commonly applied to analyze visual patterns and identify abnormalities that may indicate synthetic material \cite{b5}.

Researchers have also explored transformer-based models to improve detection accuracy by capturing long-range correlations and subtle discrepancies. Hybrid approaches that combine supervised and unsupervised learning are gaining attention due to their ability to detect a wider range of AI-generated content \cite{b6}. A key challenge is the rapid advancement of generative models, which produce increasingly realistic images, making it hard for detection systems to keep pace. The absence of standard datasets and evaluation protocols further limits the development of generalizable detection methods. Many existing algorithms rely on small or biased datasets, reducing their ability to perform across diverse types of synthetic media \cite{b7}. The spread of manipulated photos and videos also increases concerns related to impersonation, misinformation, and the bypassing of biometric security. These issues highlight the need for scalable and effective detection systems capable of mitigating risks associated with malicious uses of AI-generated content \cite{b8}.

The development of automatic detection systems is therefore essential. In this study, we focus on the CIFAKE dataset, which contains balanced classes of real and synthetic images, offering a reliable benchmark for evaluation. Two deep learning approaches are explored: a convolutional neural network (CNN) for learning spatial features and a three-dimensional CNN (3D-CNN) for capturing spatiotemporal patterns. These models are assessed using standard metrics such as accuracy, precision, recall, and F1-score. Additionally, explainable AI (XAI) techniques, including LIME and Grad-CAM, are used to interpret model decisions by highlighting important visual regions. This combination of performance evaluation and interpretability enhances system reliability and offers a transparent solution for detecting AI-generated images. The main contributions of this research are:

\begin{itemize}
    \item This study presents a systematic evaluation of two deep learning architectures, namely CNN and 3D-CNN, on the CIFAKE dataset. The objective was to achieve reliable classification between real and AI-generated images, which represents a growing challenge in modern computer vision research.  

    \item A set of preprocessing and augmentation strategies was implemented, where each original image was transformed into six additional views. This process enhanced feature diversity and improved the models’ ability to generalise across different image variations.  

    \item The experimental results highlight that the 3D-CNN achieved superior performance compared with the CNN, reaching 96.62\% accuracy, 95.97\% precision, 97.33\% recall, and a 96.64\% F1-score. These findings confirm the effectiveness of incorporating temporal depth in classification tasks.  

    \item To ensure interpretability, explainable AI methods such as LIME and Grad-CAM were applied. These techniques provided meaningful visual explanations of the models’ decision-making processes and revealed the key regions in the images that influenced predictions.  
\end{itemize}

This study is divided into five sections: Section I presents the introduction, Section II reviews the related literature, Section III discusses the model architecture, Section IV presents the experimental results, and Section V concludes the study with suggestions for future work.


\section{LITERATURE REVIEW}
The widespread use of AI-generated images threatens the reliability and trustworthiness of media. It is crucial to identify these images to prevent misleading data, identify deepfakes, and ensure the authenticity of digital content. As AI technologies become more powerful, reliable detection approaches are required to address ethical, legal, and societal concerns about their misuse. The most recent paper on this field has used several deep learning techniques. An overview of several earlier studies in this area, which served as the basis for our analysis, is given in Table~\ref{tab:literature_review}.

\begin{table*}[htbp]
    \centering
    \caption{Overview of existing literature}
    \label{tab:literature_review}
    \begin{tabular}{|p{2cm}|p{3.5cm}|p{3.5cm}|p{2.5cm}|p{4.5cm}|}
        \hline
        \textbf{Author} & \textbf{Dataset} & \textbf{Model} & \textbf{Accuracy} & \textbf{Limitations} \\
        \hline
        J. J. Bird \textit{et al.}\cite{b9} & CIFAKE dataset: 120,000 images (60,000 real, 60,000 synthetic). & CNNs for classification with Explainable AI techniques to interpret predictions. & 92.98\% & No comprehensive evaluation on other datasets; limited comparison with existing models, affecting the understanding of general effectiveness. \\
        \hline

        D. C. Epstein \textit{et al.}\cite{b10} & Dataset of 570,221 images from 14 generative methods; splits: 405,862 train, 48,057 validation, 116,302 test. & Online learning with model additions in simulated release order; CutMix augmentation for inpainting. & 99.0\% detection accuracy for inpainted images; 99.2\% with CutMix. & Classifier performance drops with significant generative model architecture changes; the dataset does not include all models from the period. \\
        \hline

        S. S. Baraheem \textit{et al.}\cite{b11} & Dataset of 24,000 images: 12,000 train, 6,000 validation, 6,000 test. & CNN architectures including VGG19, ResNet variants, InceptionV3, Xception, DenseNet121, InceptionResNetV2, MixConv, MaxViT, EfficientNetB4. & EfficientNetB4 achieved 100\% accuracy on the RSI dataset. & The framework may misclassify GAN images with sharp textures or fine details as fake, causing false positives. \\
        \hline

        R. A. F. SASKORO \textit{et al.}\cite{b12} & The dataset consists of a total of 500,000 images, categorized into two classes: natural images and AI-generated images. & ResNet-50, gated CNN & The gated network achieved the highest performance, with over 96\% average accuracy & The gated CNN model's effectiveness is highly dependent on the quality and variety of training data, leading to variability in results across different applications.\\
        \hline

        F. M. Rodriguezi \textit{et al.}\cite{b13} & Initially, 918 images (459 real, 459 AI-generated), later expanded to 1,252 balanced images using AI from OpenArt, Stable Diffusion, DALL E. & CNNs trained on PRNU and ELA feature-extracted images. & $>$95\% accuracy & PRNU and ELA techniques are only applicable to JPEG images, limiting effectiveness across other formats in real-world scenarios.\\
        \hline

        L. Whittake \textit{et al.}\cite{b14} & Synthetic media includes automatically generated/manipulated photo, audio, and video content. & Discussion focused on deepfake technology and GANs. & Not applicable & The paper lacks a technical analysis of deepfakes and GANs; it focuses on societal implications and threats rather than detection methods.\\
        \hline

    \end{tabular}
\end{table*}


% Recent work on identifying AI-generated images, including studies that use the CIFAKE dataset, has reported encouraging results with CNNs, gated CNNs, and transformer-based models. Even so, several issues remain. Many systems struggle when tested on different datasets or generators, and their performance often depends heavily on the specific architecture or the quality of the training data. In some studies, the datasets are small or unevenly balanced, which limits their ability to represent the wide range of synthetic images seen in real situations. Detection tools may also misread certain GAN outputs or overlook subtle details. These limitations highlight the need for more adaptable detection methods, which our study aims to address.

\section{Methodology}
This section describes the methodology used in this study. The dataset was split into training, validation, and test parts to ensure balanced evaluation. All images were processed through several preprocessing and augmentation steps to improve feature variety and reduce overfitting. The CNN and 3D-CNN models were then trained on the prepared training and validation sets, each learning either spatial or spatiotemporal features. After training, both models were tested on unseen images and evaluated with accuracy, precision, recall, and F1-score to determine the stronger model. To better interpret predictions, LIME and Grad-CAM were used to examine the models’ decision processes.

\subsection{Dataset Description}

In this research, we used the CIFAKE dataset, which is openly available on Kaggle \cite{b15}. The dataset contains two balanced classes labelled \texttt{REAL} and \texttt{FAKE}, totaling 120,000 images, where 100,000 were originally designated for training and 20,000 for testing. Table~\ref{tab:dataset_split} presents the detailed distribution used in this study.



\begin{table}[t]
\centering
\caption{CIFAKE Dataset Split and Class Distribution}
\begin{tabular}{|>{\centering\arraybackslash}p{1.5cm}|
                >{\centering\arraybackslash}p{1.5cm}|
                >{\centering\arraybackslash}p{2cm}|
                >{\centering\arraybackslash}p{2cm}|}
\hline
\textbf{Subset} & \textbf{REAL} & \textbf{FAKE} & \textbf{Total Images} \\
\hline
Training   & 45,000 & 45,000 & 90,000 \\
\hline
Validation & 5,000  & 5,000  & 10,000 \\
\hline
Testing    & 10,000 & 10,000 & 20,000 \\
\hline
\textbf{Total} & 60,000 & 60,000 & 120,000 \\
\hline
\end{tabular}
\label{tab:dataset_split}
\end{table}



% \begin{figure}[h]
%     \centering
%     \includegraphics[width=0.95\linewidth]{dataset.png}
%     \caption{Distribution of images across training, validation, and test sets for both classes (FAKE and REAL).}
%     \label{fig:cifake_distribution}
% \end{figure}

\subsection{Image Preprocessing}

The image preprocessing techniques applied in this study include: (i) green-channel extraction, which emphasizes luminance-related details; (ii) CLAHE (Contrast-Limited Adaptive Histogram Equalization) applied to the green channel to improve local contrast; (iii) Gaussian blurring to reduce image noise; (iv) grayscale conversion to simplify colour information into a single channel; (v) Canny edge detection to highlight important boundaries; and (vi) the Sobel gradient-magnitude operator to capture intensity changes and edges. These steps were carried out on every image in the CIFAKE dataset \cite{b15}, producing six additional processed versions while keeping the original labels.  

Each original image was expanded into six processed versions, increasing the dataset size. The final splits included 600,000 images for training (300k REAL, 300k FAKE), 60,000 for validation (30k REAL, 30k FAKE), and 120,000 for testing (60k REAL, 60k FAKE). All subsets were balanced, and sample outputs from the preprocessing steps are shown in Fig.~\ref{fig:prep_examples}.

% ---- REMOVED FIGURE 1 ----
% (preprocess_dataset.png)

\begin{figure}[h]
    \centering
    \includegraphics[width=0.98\linewidth]{preprocess.png}
    \caption{Preprocessing results for a sample image, including the original and all derived variants.}
    \label{fig:prep_examples}
\end{figure}

\vspace{-0.4cm}

\subsection{Image Augmentation}
In order to reduce overfitting and improve the generalization ability of the model, data augmentation was applied to the training images. The augmentation pipeline consisted of random rotation (up to $\pm 20^{\circ}$), horizontal and vertical flips, random zoom between 80\% and 120\%, random translation of up to 10\% in both height and width, and random contrast adjustment with a factor of 0.2. These transformations introduced controlled variations in orientation, scale, position, and lighting, which helped the network to learn more robust feature representations.  

The augmentation was performed dynamically during training, meaning that each batch contained different transformations of the same images. This strategy ensured that the model did not simply memorize the training data but instead learned to recognize patterns under varying conditions. Such augmentation practices are widely adopted in computer vision tasks to improve classification accuracy and model stability \cite{b16}.  

\vspace{-0.2cm}

\subsection{Proposed Model CNN and 3D-CNN}

In this research, two deep learning models were developed for distinguishing genuine images from AI-generated content: a custom CNN for learning spatial information and a 3D-CNN for capturing spatiotemporal features. CNNs are widely used in vision tasks due to their ability to extract hierarchical representations \cite{b17}, progressing from low-level textures to higher-level semantics. In our framework, the CNN processes individual images, whereas the 3D-CNN operates on stacked inputs, following prior work showing the effectiveness of 3D kernels for sequence-based analysis \cite{b18}. Together, these models provide complementary capabilities for robust detection of synthetic visual content.

The forward propagation of a convolutional layer is expressed as:

\vspace{-0.7cm}

\begin{equation}
F_{i,j,k} = \sigma \left( \sum_{m=1}^{M} \sum_{p=1}^{P} \sum_{q=1}^{Q} W_{p,q,m,k} \cdot X_{i+p, j+q, m} + b_k \right),
\end{equation}

where $X$ is the input tensor, $W$ represents the convolutional kernel of size $P \times Q$, $b_k$ is the bias term, $\sigma$ denotes the non-linear activation function (ReLU in this study), and $k$ indexes the output channel. In 3D-CNNs, this formulation is extended by including the temporal depth dimension, which allows the model to learn sequential structures in addition to spatial features.  

\begin{table*}[t]
\centering
\caption{CNN architecture with batch normalization and dropout}
\label{tab:cnn2d_full_arch}
\setlength{\tabcolsep}{6pt}
\renewcommand{\arraystretch}{1.15}
\begin{tabular}{|l|c|c|c|c|l|l|}
\hline
\textbf{Layers} & \textbf{Filters} & \textbf{Kernel} & \textbf{Padding} & \textbf{Activation} & \textbf{Pooling} & \textbf{Regularization} \\
\hline
2$\times$Conv2D+BN & 32  & $3\times3$ & \texttt{same} & ReLU & MaxPool $2\times2$ & Dropout(0.2) \\
\hline
2$\times$Conv2D+BN & 64  & $3\times3$ & \texttt{same} & ReLU & MaxPool $2\times2$ & Dropout(0.3) \\
\hline
2$\times$Conv2D+BN & 128 & $3\times3$ & \texttt{same} & ReLU & MaxPool $2\times2$ & Dropout(0.3) \\
\hline
2$\times$Conv2D+BN & 256 & $3\times3$ & \texttt{same} & ReLU & MaxPool $2\times2$ & Dropout(0.4) \\
\hline
2$\times$Conv2D+BN & 512 & $3\times3$ & \texttt{same} & ReLU & GlobalAvgPool & Dropout(0.5) \\
\hline
\end{tabular}
\end{table*}

The CNN architecture in Table~\ref{tab:cnn2d_full_arch} contains five convolutional blocks, each with two Conv2D layers, batch normalization, and ReLU activation. Dropout rates increase gradually to reduce overfitting, while MaxPooling layers downsample spatial dimensions. The number of filters grows from 32 to 512, enabling the extraction of richer feature representations. A Global Average Pooling layer provides compact features before the final dense layers and sigmoid classification.
 

\begin{table*}[t]
\centering
\caption{3D CNN architecture with batch normalization and dropout}
\label{tab:cnn3d_full_arch}
\setlength{\tabcolsep}{6pt}
\renewcommand{\arraystretch}{1.15}
\begin{tabular}{|l|c|c|c|c|l|l|}
\hline
\textbf{Layers} & \textbf{Filters} & \textbf{Kernel} & \textbf{Padding} & \textbf{Activation} & \textbf{Pooling} & \textbf{Regularization} \\
\hline
2$\times$Conv3D+BN & 32  & $3\times3\times3$ & \texttt{same} & ReLU & MaxPool3D $(1,2,2)$ & Dropout(0.2) \\
\hline
2$\times$Conv3D+BN & 64  & $3\times3\times3$ & \texttt{same} & ReLU & MaxPool3D $(2,2,1)$ & Dropout(0.3) \\
\hline
2$\times$Conv3D+BN & 128 & $3\times3\times3$ & \texttt{same} & ReLU & MaxPool3D $(2,2,1)$ & Dropout(0.3) \\
\hline
2$\times$Conv3D+BN & 256 & $3\times3\times3$ & \texttt{same} & ReLU & MaxPool3D $(2,2,1)$ & Dropout(0.4) \\
\hline
2$\times$Conv3D+BN & 512 & $3\times3\times3$ & \texttt{same} & ReLU & GlobalAvgPool3D & Dropout(0.5) \\
\hline
\end{tabular}
\end{table*}

The 3D-CNN architecture (Table~\ref{tab:cnn3d_full_arch}) follows the same design principles but employs 3D kernels to learn spatial and temporal patterns. Each block includes two Conv3D layers with batch normalization and ReLU. Asymmetric MaxPool3D layers reduce spatial dimensions while preserving temporal depth. Filter sizes scale from 32 to 512, and dropout rates increase across blocks. A Global Average Pooling 3D layer aggregates spatiotemporal features before the dense classifier.
 

Both CNN and 3D-CNN models were trained using binary cross-entropy loss:  

\vspace{-0.4cm}

\begin{equation}
\mathcal{L} = - \frac{1}{N} \sum_{i=1}^{N} \Big[ y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i) \Big],
\end{equation}

where $y_i$ is the ground truth label, $\hat{y}_i$ is the predicted probability, and $N$ is the number of training samples. The Adam optimizer was employed, with a learning rate of $1 \times 10^{-5}$ for the CNN and $1 \times 10^{-4}$ for the 3D-CNN. Early stopping with best-weight restoration was applied to avoid overfitting, and model performance was evaluated using binary accuracy, precision, and recall.  

The CNN effectively captures fine-grained spatial patterns, whereas the 3D-CNN learns structural dependencies across stacked inputs. Combining these complementary models enhances the system’s ability to distinguish authentic images from synthetic ones.

\subsection{Explainable AI}

To gain a clearer understanding of how the proposed models made their predictions, we applied Explainable Artificial Intelligence (XAI) techniques. Two established methods were used in this study: Local Interpretable Model-agnostic Explanations (LIME) and Gradient-weighted Class Activation Mapping (Grad-CAM).  

LIME explains individual predictions by creating simple, local approximations of a complex model. It perturbs the input image and observes the effect on the output, thereby identifying which regions of the image contribute most to a classification decision \cite{b19}. Grad-CAM, in contrast, generates class-discriminative heatmaps by using the gradients of target class scores that flow into the last convolutional layers. These heatmaps highlight the spatial areas of the image that strongly influenced the model’s decision \cite{b20}.  

\section{Result Analysis}

When the training phase was completed, the effectiveness of CNN and 3D-CNN models in distinguishing between real and AI-generated images was evaluated using the test dataset. The architectural details of these models were discussed in the previous section. In this section, their performance is analyzed. The evaluation considered accuracy-loss curves, confusion matrices, and standard metrics such as accuracy, precision, recall, and F1-score. In addition, Explainable AI (XAI) methods were employed to interpret the internal decision-making process of the models. This combination of quantitative evaluation and interpretability offered a better perspective on the strengths and limitations of the CNN and 3D-CNN frameworks.  

\subsection{Model Training and Evaluation Results}  

Between the two models, the 3D-CNN achieved superior results across most metrics. Fig.~\ref{fig:best_model} illustrates the training and validation curves of the 3D-CNN, which was selected as the best-performing model. The training accuracy rose quickly, surpassing 95\% within 15 epochs, while the validation accuracy stabilized around 96\% with only minor fluctuations. Both training and validation loss decreased steadily, although slight variations were noted in the validation loss. Precision and recall remained consistently above 95\%, reflecting strong generalization capability. The use of early stopping terminated training at epoch 44 for the 3D-CNN, compared with 126 epochs for the CNN, indicating that the 3D-CNN converged faster and more efficiently.  

A detailed comparison of the two models is presented in Table~\ref{tab:model_performance}. The CNN achieved 95.69\% accuracy with a high recall of 98.00\%, demonstrating strong sensitivity. However, the 3D-CNN outperformed it overall, achieving higher accuracy (96.62\%) and precision (95.97\%), as well as the best F1-score (96.64\%). These results confirm the robustness of the 3D-CNN in correctly classifying both real and AI-generated images.  

\begin{figure}[h]
    \centering
    \includegraphics[width=1\linewidth]{3DCNN.png}
    \caption{Training and validation curves for the best-performing model (3D-CNN).}
    \label{fig:best_model}
\end{figure}  

\begin{table}[h]
\centering
\caption{Performance comparison of CNN and 3D-CNN on the CIFAKE dataset.}
\label{tab:model_performance}
\resizebox{\linewidth}{!}{%
\begin{tabular}{|l|c|c|c|c|}
\hline
\textbf{Model} & \textbf{Accuracy} & \textbf{Precision} & \textbf{Recall} & \textbf{F1-Score} \\ \hline
CNN     & 95.69\% & 93.67\% & 98.00\% & 95.79\% \\ \hline
3D-CNN  & 96.62\% & 95.97\% & 97.33\% & 96.64\% \\ \hline
\end{tabular}%
}
\end{table}  

In summary, both models produced strong outcomes, but the 3D-CNN offered the most balanced performance across accuracy, precision, recall, and F1-score. Its capacity to capture both spatial and temporal features explains its improved effectiveness compared with the CNN, establishing it as the more reliable model for this task.  

\subsection{3D-CNN Test Set Confusion Matrix}

The confusion matrix in Table~\ref{tab:3dcnn_confmat} shows how the 3D-CNN model performed on the test set. Out of 10,000 \texttt{FAKE} images, 9,591 were correctly classified, while 409 were predicted as \texttt{REAL}. For the 10,000 \texttt{REAL} images, the model correctly identified 9,733 and misclassified 267 as \texttt{FAKE}. These results indicate strong precision and recall for both classes, with only a small number of errors. Overall, the model demonstrates high reliability in distinguishing real from AI-generated images on the CIFAKE dataset.

\begin{table}[h]
\centering
\caption{3D-CNN Confusion Matrix on the Test Set}
\begin{tabular}{|>{\centering\arraybackslash}p{2.5cm}|
                >{\centering\arraybackslash}p{2cm}|
                >{\centering\arraybackslash}p{2cm}|}
\hline
\textbf{True / Predicted} & \textbf{FAKE (0)} & \textbf{REAL (1)} \\
\hline
\textbf{FAKE (0)} & 9,591 & 409 \\
\hline
\textbf{REAL (1)} & 267 & 9,733 \\
\hline
\end{tabular}
\label{tab:3dcnn_confmat}
\end{table}



% \subsection{3D-CNN ROC Curve}

% The Receiver Operating Characteristic (ROC) curve in Fig.~\ref{fig:3dcnn_roc} shows the performance of the 3D-CNN model on the test data. The curve rises steeply towards the top-left corner, which indicates that the model is able to achieve a high true positive rate while keeping the false positive rate very low. The area under the curve (AUC) is 0.99, reflecting an excellent ability to separate the \texttt{REAL} and \texttt{FAKE} classes. Such a high AUC score confirms that the 3D-CNN provides reliable classification and is highly effective when applied to the CIFAKE dataset.  

% \begin{figure}[h]
%     \centering
%     \includegraphics[width=0.7\linewidth]{3DROC.png}
%     \caption{ROC curve for the 3D-CNN model with an AUC of 0.99.}
%     \label{fig:3dcnn_roc}
% \end{figure}

\vspace{-0.5cm}

\subsection{3D-CNN Grad-CAM Visualisation}

Grad-CAM was applied to the 3D-CNN model to provide a visual explanation of how the network distinguished between \texttt{REAL} and \texttt{FAKE} images. As shown in Fig.~\ref{fig:3d_gradcam}, the first and third rows present correctly classified \texttt{REAL} cases. In these examples, the Grad-CAM heatmaps clearly highlight meaningful regions such as the body shape of the dog and the structural outline of the bird, confirming that the model focused on essential features of the objects.  

The second row illustrates a correctly identified \texttt{FAKE} case. Here, the Grad-CAM heatmap appears less structured, capturing the irregular textures and inconsistent patterns typical of AI-generated images. The resulting overlays emphasise the regions that most influenced the decision, showing how the model effectively differentiated synthetic content from real imagery.  

\begin{figure}[h]
    \centering
    \includegraphics[width=0.9\linewidth]{3dGrad.png}
    \caption{Grad-CAM visualisations highlighting the regions most influential to the 3D-CNN’s predictions on CIFAKE images.}
    \label{fig:3d_gradcam}
\end{figure}

\subsection{3D-CNN LIME Visualisation}

To further interpret the decision-making process of the 3D-CNN, Local Interpretable Model-agnostic Explanations (LIME) were applied. Fig.~\ref{fig:3d_lime} presents visual explanations for both \texttt{REAL} and \texttt{FAKE} samples.  

In the first row, a \texttt{REAL} image is correctly classified with a confidence score of 0.98. The LIME maps highlight the outline and structure of the boat, indicating that the model relied on meaningful object features to support its prediction. In the second row, a \texttt{FAKE} image is also correctly identified. Here, the highlighted evidence is scattered and irregular, reflecting the unstable textures commonly observed in synthetic content. These observations suggest that the 3D-CNN effectively captured reliable cues for distinguishing between real and AI-generated images.  

\begin{figure}[h]
\centering
\includegraphics[width=0.95\linewidth]{3dlime.png}
\caption{LIME explanation images of a few sample images with necessary labels.}
\label{fig:3d_lime}
\end{figure}

\subsection{MODEL COMPARISON WITH PREVIOUS RELATED WORKS}

\begin{table*}[h]
\centering
\caption{MODEL COMPARISON WITH PREVIOUS RELATED WORKS}
\label{tab:comparison_related}
\resizebox{\linewidth}{!}{%
\begin{tabular}{|l|c|c|}
\hline
\textbf{Previous Paper} & \textbf{Dataset} & \textbf{Accuracy/Precision/Recall/F1-Score} \\ \hline

J. J. Bird \textit{et al.} \cite{b9} & CIFAKE & Accuracy 92.98\% \\ \hline

D. C. Epstein \textit{et al.} \cite{b10} & 570,221 images from 14 generative methods & Accuracy 99.0\% (inpainting), 99.2\% (with CutMix) \\ \hline

S. S. Baraheem \textit{et al.} \cite{b11} & RSI dataset (24,000 images) & EfficientNetB4: 100\% Accuracy \\ \hline

R. A. F. SASKORO \textit{et al.} \cite{b12} & 500,000 images (real vs synthetic) & Gated CNN: 96\% Accuracy \\ \hline

F. M. Rodriguezi \textit{et al.} \cite{b13} & 1,252 images (balanced) & Accuracy $>$95\% \\ \hline

\textbf{Our Proposed Models} & CIFAKE & 
CNN: Accuracy 95.69\%, Precision 93.67\%, Recall 98.00\%, F1 95.79\% \\
& & 3D-CNN: Accuracy 96.62\%, Precision 95.97\%, Recall 97.33\%, F1-Score 96.64\% \\ \hline

\end{tabular}%
}
\end{table*}

Table~\ref{tab:comparison_related} presents a comparison of our proposed CNN and 3D-CNN models with previous studies. Earlier works, such as Bird \textit{et al.}, achieved 92.98\% accuracy on CIFAKE, while Epstein \textit{et al.} reported over 99\% detection using CutMix augmentation. Gated CNN approaches and transformer-based models also delivered competitive results but were often dataset-dependent. In contrast, our models combined strong accuracy with interpretability. The CNN achieved 95.69\% accuracy, while the 3D-CNN improved further with 96.62\% and an F1-score of 96.64\%. These findings highlight the robustness and practical value of our framework for AI-generated image detection.


\section{Conclusion and Future Work}

This study addressed the growing challenge of distinguishing real images from AI-generated content by proposing a classification framework based on deep learning. Using the CIFAKE dataset, which provides a balanced set of REAL and FAKE images, we developed two models: a Convolutional Neural Network (CNN) for extracting spatial features and a three-dimensional CNN (3D-CNN) for capturing spatiotemporal patterns. To strengthen generalisation, preprocessing and augmentation techniques were applied, producing multiple image variants for robust training. Both models achieved competitive results; however, the 3D-CNN outperformed the CNN with an accuracy of 96.62\%, precision of 95.97\%, recall of 97.33\%, and an F1-score of 96.64\%. These results confirm that incorporating temporal depth provides significant advantages in classifying synthetic images that often mimic real-world textures and structures. To ensure transparency, explainable AI techniques were employed. LIME offered local feature interpretations by identifying regions most responsible for classification, while Grad-CAM produced visual heatmaps that highlighted the spatial cues guiding predictions. This dual approach strengthened user confidence in the reliability of the models and demonstrated the practical value of combining detection accuracy with interpretability. For future work, we propose expanding this framework to larger and more diverse datasets, including those generated by the latest diffusion and transformer-based models, to test generalisation across unseen domains. We also suggest exploring transformer architectures and additional XAI methods, such as SHAP, for deeper interpretability. Finally, lightweight model variants should be developed to enable real-time classification, making the system suitable for applications in journalism, digital forensics, and online content verification.


\begin{thebibliography}{00}
\bibitem{b1} T. Karras, S. Laine, M. Aittala, J. Hellsten, J. Lehtinen, and T. Aila, “Analyzing and improving the image quality of StyleGAN,” in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2020, pp. 8110–8119.

\bibitem{b2} A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, “Attention is all you need,” in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 30, 2017.

\bibitem{b3} TensorFlow, “Neural machine translation with a Transformer and Keras,” TensorFlow Text Tutorials, Sep. 2023. [Online]. Available: https://www.tensorflow.org/text/tutorials/transformer?hl=en

\bibitem{b4} T. T. Nguyen, Q. V. H. Nguyen, D. T. Nguyen, D. T. Nguyen, T. Huynh-The, S. Nahavandi, T. T. Nguyen, Q.-V. Pham, and C. M. Nguyen, “Deep learning for deepfakes creation and detection: A survey,” *Comput. Vis. Image Underst.*, vol. 223, p. 103525, 2022.

\bibitem{b5} A. Tucker, “Synthetic media and deepfakes: Tactical media in the pluriverse,” *Digit. Stud./Le champ numérique*, vol. 13, no. 1, 2023.

\bibitem{b6} P. Liu, Y. Lin, Y. He, Y. Wei, L. Zhen, J. T. Zhou, R. S. M. Goh, and J. Liu, “Automated deepfake detection,” *arXiv preprint* arXiv:2106.10705, 2021.

\bibitem{b7} D. R. Don, J. Boardman, S. Sayenju, R. Aygun, Y. Zhang, B. Franks, S. Johnston, G. Lee, D. Sullivan, and G. Modgil, “Automation of explainability auditing for image recognition,” *Int. J. Multimedia Data Eng. Manag. (IJMDEM)*, vol. 14, no. 1, pp. 1–17, 2023.

\bibitem{b8} C.-C. Hsu, Y.-X. Zhuang, and C.-Y. Lee, “Deep fake image detection based on pairwise learning,” *Appl. Sci.*, vol. 10, no. 1, p. 370, 2020.

\bibitem{b9} J. J. Bird and A. Lotfi, “CIFAKE: Image classification and explainable identification of AI-generated synthetic images,” *IEEE Access*, vol. 12, pp. 15642–15650, 2024.

\bibitem{b10} D. C. Epstein, I. Jain, O. Wang, and R. Zhang, “Online detection of AI-generated images,” in *Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV)*, 2023, pp. 382–392.

\bibitem{b11} S. S. Baraheem and T. V. Nguyen, “AI vs. AI: Can AI detect AI-generated images?,” *J. Imaging*, vol. 9, no. 10, p. 199, 2023.

\bibitem{b12} R. A. F. Saskoro, N. Yudistira, and T. N. Fatyanosa, “Detection of AI-generated images from various generators using gated expert convolutional neural network,” *IEEE Access*, 2024.

\bibitem{b13} F. Martin-Rodriguez, R. Garcia-Mojon, and M. Fernandez-Barciela, “Detection of AI-created images using pixel-wise feature extraction and convolutional neural networks,” *Sensors*, vol. 23, no. 22, p. 9037, 2023.

\bibitem{b14} L. Whittaker, T. C. Kietzmann, J. Kietzmann, and A. Dabirian, “All around me are synthetic faces: The mad world of AI-generated media,” *IT Prof.*, vol. 22, no. 5, pp. 90–99, 2020.

\bibitem{b15} J. J. Bird, “CIFAKE: Real and AI-Generated Synthetic Images,” Kaggle, 2023. [Online]. Available: \url{https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images/data}

Y. LeCun, Y. Bengio, and G. Hinton, “Deep learning,” *Nature*, vol. 521, no. 7553, pp. 436–444, 2015.

\bibitem{b16} A. Shorten and T. M. Khoshgoftaar, ``A survey on image data augmentation for deep learning,'' \emph{Journal of Big Data}, vol. 6, no. 60, pp. 1–48, 2019.  

\bibitem{b17} Y. LeCun, Y. Bengio, and G. Hinton, “Deep learning,” *Nature*, vol. 521, no. 7553, pp. 436–444, 2015.

\bibitem{b18}  S. Ji, W. Xu, M. Yang, and K. Yu, “3D convolutional neural networks for human action recognition,” \textit{IEEE Transactions on Pattern Analysis and Machine Intelligence}, vol. 35, no. 1, pp. 221–231, 2013.   

\bibitem{b19} M. T. Ribeiro, S. Singh, and C. Guestrin, ``Why should I trust you?: Explaining the predictions of any classifier,'' in \emph{Proc. 22nd ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining}, 2016, pp. 1135–1144.  

\bibitem{b20} R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, ``Grad-CAM: Visual explanations from deep networks via gradient-based localization,'' in \emph{Proc. IEEE Int. Conf. Computer Vision (ICCV)}, 2017, pp. 618–626.  

\end{thebibliography}
\vspace{12pt}

\end{document}
