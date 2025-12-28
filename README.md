# Chapter2_BERT_SBR-
Code for BERT-based Severity Prediction of Bug Report (BERT-SBR) with all Machine Learning (LR, RF, AdaBoost) and Deep Learning (CNN, RNN, LSTM, NN) baselines, including preprocessing, under-sampling, and over-sampling. All experiments are executed using Google Colab.


1. Open Google Colab and enable GPU: Runtime → Change runtime type → Hardware accelerator → GPU.2. Install all required libraries** by running the following commands in Colab: !pip
2. install numpy pandas scipy tqdm
!pip install scikit-learn
!pip install torch torchvision torchaudio
!pip install transformers datasets tokenizers accelerate sentencepiece
!pip install nltk spacy regex
!pip install matplotlib seaborn evaluate imbalanced-learn , 

3. Download NLP resources** import nltk, nltk.download('punkt') , nltk.download('stopwords') ,!python -m spacy download en_core_web_sm
Download the dataset by running the notebook: https://huggingface.co/datasets/sealuzh/app_reviews 
After downloading, perform text cleaning and preprocessing (tokenization, normalization, label encoding). The preprocessed dataset is saved automatically during execution inside the notebooks.

4.  Handle class imbalance using oversampling by running: Oversampling_on_50000_swn.ipynb
5.  
6.  Run Machine Learning baseline models using TF-IDF features by executing: Machine_Learning_on_50000_(Preprocessed+embeddings).ipynb
7.  
8.  Run Deep Learning baseline models by executing: (PA)Project_on_50000_swn_402_.ipynb This notebook trains and evaluates: CNN, RNN, LSTM and Feed-Forward Neural Network (NN)
Run the BERT fine-tuning experiment by executing: (PA)Project_on_50000_swn_128_max_8_batch,_.ipynb

9. This notebook includes: BERT tokenization, Fine-tuning on bug report text, GPU-accelerated training, Final severity prediction results
Ensure reproducibility by: Running all notebooks in the same order, Using Google Colab with GPU enabled, Not skipping any preprocessing cells and Using the default hyperparameters provided
