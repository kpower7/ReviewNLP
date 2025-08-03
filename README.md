# Amazon Local Transfer - NLP Sentiment Analysis

A comprehensive natural language processing project for Amazon product review sentiment analysis, developed as part of MIT's SCM256 course. This project leverages state-of-the-art transformer models including BERT and RoBERTa for fine-grained sentiment classification and supply chain insights.

## 🎯 Project Overview

This repository contains a complete NLP pipeline for analyzing Amazon Grocery & Gourmet Food reviews to extract actionable business intelligence for supply chain management. The project combines:

- **Advanced NLP Models**: BERT, RoBERTa fine-tuning for sentiment classification
- **Large-Scale Data Processing**: Efficient handling of millions of Amazon reviews
- **Multi-Model Comparison**: Comparative analysis across different transformer architectures
- **Supply Chain Applications**: Sentiment-driven insights for inventory and logistics optimization

### Business Problem

Understanding customer sentiment at scale is crucial for supply chain optimization. This project addresses:
- **Demand Forecasting**: Sentiment trends as leading indicators of demand changes
- **Quality Control**: Early detection of product quality issues through review analysis
- **Supplier Performance**: Sentiment analysis for supplier and product evaluation
- **Risk Management**: Identifying potential supply chain disruptions through customer feedback

## 📁 Repository Structure

```
Amazon_Local_Transfer/
├── Amazon_EDA_v2.ipynb              # Comprehensive exploratory data analysis
├── BERT_for_Amazon/                 # Core BERT implementation
│   ├── Amazon_sentiment_analysis.py # Main BERT training pipeline
│   ├── BERT_delayed.py              # Delayed shipment analysis
│   ├── BERT_expire.py               # Product expiration analysis
│   ├── GPT_call.py                  # GPT API integration
│   ├── Deepseek_call.py             # Deepseek API integration
│   └── finetuned_model/             # Trained model artifacts
├── BERT_for_Amazon_Expanded/        # Scaled BERT implementation
│   ├── Amazon_expanded_optimized_v2.py  # Optimized training pipeline
│   ├── Amazon_expanded_distributed.py   # Distributed training setup
│   └── daily_trained/               # Incremental training models
├── BERT_for_Amazon_LowRating/       # Low-rating specific analysis
├── BERT_for_Amazon_combined/        # Multi-model ensemble approach
├── roBERTa_for_Amazon/              # RoBERTa model implementation
└── checkpoint-*/                    # Model checkpoints and training states
```

## 🔧 Key Features

### 1. Multi-Model Architecture
- **BERT Base**: Fine-tuned for 5-class sentiment classification (1-5 stars)
- **RoBERTa**: Robustly optimized BERT approach for improved performance
- **Ensemble Methods**: Combined predictions from multiple transformer models
- **Specialized Models**: Targeted analysis for low ratings and specific use cases

### 2. Advanced Text Processing
- **Preprocessing Pipeline**: HTML cleaning, tokenization, lemmatization
- **Feature Engineering**: Combined review text and summary analysis
- **Multi-threading**: Parallel processing for large-scale data handling
- **Memory Optimization**: Efficient data loading and batch processing

### 3. Supply Chain Applications
- **Delayed Shipment Analysis**: Sentiment correlation with logistics performance
- **Product Expiration Tracking**: Quality control through review sentiment
- **Supplier Performance**: Vendor evaluation through customer feedback
- **Demand Signal Detection**: Early warning systems for inventory management

### 4. Scalable Training Infrastructure
- **GPU Optimization**: CUDA-enabled training for large models
- **Distributed Training**: Multi-node training capabilities
- **Incremental Learning**: Daily model updates with new review data
- **Checkpoint Management**: Robust model versioning and recovery

## 🚀 Getting Started

### Prerequisites

```bash
pip install torch transformers datasets scikit-learn pandas numpy nltk beautifulsoup4
pip install polars matplotlib seaborn tqdm
```

### Quick Start

1. **Data Exploration**:
```bash
jupyter notebook Amazon_EDA_v2.ipynb
```

2. **BERT Training**:
```python
python BERT_for_Amazon/Amazon_sentiment_analysis.py
```

3. **Optimized Training**:
```python
python BERT_for_Amazon_Expanded/Amazon_expanded_optimized_v2.py
```

### Data Requirements

- **Amazon Reviews Dataset**: Grocery & Gourmet Food reviews in JSONL format
- **Required Fields**: reviewText, summary, overall (rating), asin, reviewTime
- **Scale**: Optimized for millions of reviews with efficient memory usage

## 📊 Model Performance

### Evaluation Metrics
- **Accuracy**: Multi-class classification accuracy (1-5 stars)
- **Precision/Recall/F1**: Per-class performance analysis
- **Confusion Matrix**: Detailed classification performance
- **Training Loss**: Convergence monitoring and optimization

### Model Comparison
- **BERT vs RoBERTa**: Comparative analysis of transformer architectures
- **Fine-tuning Strategies**: Different approaches to domain adaptation
- **Ensemble Performance**: Combined model predictions for improved accuracy
- **Computational Efficiency**: Training time and resource utilization analysis

## 🔍 Key Insights

### Sentiment Patterns
- **Rating Distribution**: Analysis of 1-5 star rating patterns
- **Temporal Trends**: Seasonal and temporal sentiment variations
- **Product Categories**: Category-specific sentiment characteristics
- **Review Length Correlation**: Relationship between review length and sentiment

### Supply Chain Intelligence
- **Quality Indicators**: Sentiment as early warning for quality issues
- **Logistics Performance**: Correlation between delivery experience and sentiment
- **Supplier Insights**: Vendor performance through customer feedback analysis
- **Demand Forecasting**: Sentiment trends as demand predictors

## 🛠️ Technical Implementation

### Model Architecture
- **Input Processing**: Tokenization with BERT/RoBERTa tokenizers
- **Feature Combination**: Review text + summary concatenation
- **Classification Head**: 5-class sentiment classification layer
- **Training Strategy**: Fine-tuning with domain-specific data

### Optimization Techniques
- **Batch Processing**: Efficient data loading with optimal batch sizes
- **Memory Management**: Gradient checkpointing and mixed precision training
- **Parallel Processing**: Multi-threading for data preprocessing
- **Model Checkpointing**: Regular saving for training recovery

### API Integration
- **GPT Integration**: Comparison with OpenAI models
- **Deepseek Integration**: Alternative LLM comparison
- **Model Serving**: Inference pipeline for real-time predictions
- **Batch Prediction**: Efficient processing of large review datasets

## 📈 Business Impact

### Supply Chain Optimization
- **Inventory Planning**: Sentiment-driven demand forecasting
- **Quality Assurance**: Early detection of product quality issues
- **Supplier Management**: Data-driven supplier performance evaluation
- **Customer Experience**: Proactive identification of service issues

### Operational Benefits
- **Risk Mitigation**: Early warning system for potential issues
- **Cost Reduction**: Optimized inventory based on sentiment trends
- **Revenue Enhancement**: Improved customer satisfaction through insights
- **Strategic Planning**: Long-term trend analysis for business decisions

## 🔬 Research Applications

This project demonstrates advanced concepts in:
- **Transfer Learning**: Fine-tuning pre-trained transformers for domain-specific tasks
- **Large-Scale NLP**: Processing millions of text documents efficiently
- **Multi-Model Ensembles**: Combining different architectures for improved performance
- **Supply Chain Analytics**: Practical NLP applications in operations management

## 📚 Technical References

- Devlin, J., et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers
- Liu, Y., et al. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach
- Rogers, A., et al. (2020). A Primer on Neural Network Models for Natural Language Processing
- Qiu, X., et al. (2020). Pre-trained Models for Natural Language Processing: A Survey

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

---

*Developed as part of MIT SCM256 - advancing the application of natural language processing in supply chain management and operations research.*
