# Student Stress Monitoring System

## Project Overview

A comprehensive machine learning system for monitoring and predicting student stress levels. This project analyzes student stress datasets and implements multiple machine learning models to classify stress levels based on various physiological and environmental factors.

## Features

- **Data Analysis**: Comprehensive exploratory data analysis with statistical reports and visualizations
- **Multiple ML Models**: Implementation of seven different machine learning algorithms
- **Modular Architecture**: Well-structured codebase following software engineering best practices
- **Web Interface**: Streamlit-based GUI for interactive data exploration and model evaluation
- **Type Safety**: Full type hinting and Pydantic validation throughout the codebase

## Project Structure

```
student-stress-monitoring/
├── src/
│   └── stress_monitor/
│       ├── config.py              # Configuration management
│       ├── data/
│       │   ├── data_loader.py     # Data loading utilities
│       │   └── data_preprocessor.py # Data preprocessing pipelines
│       ├── models/
│       │   ├── base_model.py      # Abstract base model class
│       │   └── model_factory.py   # Model creation factory
│       ├── visualization/
│       │   └── plotter.py         # Data visualization utilities
│       └── utils/
│           └── helpers.py         # Utility functions
├── tests/                         # Unit tests
├── app.py                         # Streamlit application
├── requirements.txt               # Python dependencies
└── setup.py                      # Package configuration
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Setup

1. Clone or download the project files
   ```bash
   git clone https://gitlab.com/machine-learning-lc0/student-stress-monitoring
   ```
2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Prepare the data:
   - Create a `data` directory in the project root
   - Place your CSV files in the data directory:
     - `StressLevelDataset.csv`
     - `Stress_Dataset.csv`

## Usage

### Running the Web Application

Start the Streamlit dashboard:
```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

### Using the Python Package

You can also use the modules programmatically:

```python
from src.stress_monitor.data.data_loader import DataLoader
from src.stress_monitor.models.model_factory import ModelFactory
from src.stress_monitor.config import ModelType

# Load data
loader = DataLoader()
data = loader.load_dataset("stress_level")

# Create and train a model
model = ModelFactory.create_model(ModelType.RANDOM_FOREST)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
```

## Machine Learning Models

The system implements the following machine learning algorithms:

1. **Logistic Regression** - Baseline classification model
2. **Gradient Boosting Classifier** - Ensemble method with sequential tree building
3. **Random Forest Classifier** - Ensemble method with parallel tree building
4. **Tweedie Regressor** - Generalized linear model for various distributions
5. **SGD Classifier** - Stochastic gradient descent optimization
6. **CatBoost Classifier** - Gradient boosting with categorical feature support
7. **MLP Classifier** - Multi-layer perceptron neural network

## Configuration

The system uses Pydantic for configuration management. Key configuration options in `src/stress_monitor/config.py`:

- Data file paths and directories
- Model hyperparameters
- Test/train split ratios
- Random state for reproducibility

## Data Sources

The project is designed to work with student stress monitoring datasets containing features such as:

- Physiological indicators (headache, blood pressure, sleep quality)
- Environmental factors (noise level, living conditions, safety)
- Academic pressures (study load, academic performance)
- Social factors (peer pressure, social support, bullying)

## Development

### Running Tests

Execute the test suite:
```bash
python -m pytest tests/
```

### Code Style

The project follows PEP 8 guidelines and includes comprehensive type hints. Key practices:

- Type annotations for all function parameters and return values
- Pydantic models for data validation
- Abstract base classes for extensibility
- Factory pattern for model creation

### Extending the System

To add new models:

1. Create a new model class inheriting from `BaseStressModel`
2. Implement the required abstract methods
3. Register the model in `ModelFactory._models`
4. Add configuration parameters in `config.py`

## Dependencies

### Core Dependencies

- **pandas** - Data manipulation and analysis
- **scikit-learn** - Machine learning algorithms and utilities
- **matplotlib** - Basic plotting and visualization
- **seaborn** - Statistical data visualization

### Web Interface

- **streamlit** - Web application framework
- **plotly** - Interactive visualizations

### Development Tools

- **pydantic** - Data validation and settings management
- **rich** - Enhanced console output and formatting

## Performance

The system includes performance monitoring and model comparison capabilities. The web interface provides:

- Accuracy scores for all trained models
- Comparative visualization of model performance
- Feature importance analysis
- Correlation heatmaps

##  Contact Me
Have questions about the project or interested in collaboration? Feel free to reach out!
[![LinkedIn](https://img.shields.io/badge/LinkedIn-%230077B5.svg?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/your-profile/)
[![GitLab](https://img.shields.io/badge/GitLab-%23181717.svg?style=for-the-badge&logo=gitlab&logoColor=white)](https://gitlab.com/skillIssueCM)
[![Gmail](https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:losif.ai.2050@gmail.com)
### Connect With Me

<p align="left">
  <a href="https://www.linkedin.com/in/yousef-fawzi/" target="_blank">
    <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/linkedin/linkedin-original.svg" alt="linkedin" width="40" height="40"/>
  </a>
    <a href="https://gitlab.com/skillIssueCM" target="_blank">
    <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/gitlab/gitlab-original.svg" alt="gitlab" width="40" height="40"/>
  </a>
  <a href="mailto:losif.ai.2050@gmail.com" target="_blank">
    <img src="https://imgs.search.brave.com/YjZyc-VnhgEy7ANjFgVM-SlrvLHkQ7FeRZU7_OtLHo8/rs:fit:500:0:1:0/g:ce/aHR0cHM6Ly93d3cu/c3ZncmVwby5jb20v/c2hvdy80NTIyMTMv/Z21haWwuc3Zn" alt="gmail" width="40" height="40"/>
  </a>
</p>

**Email:** [losif.ai.2050@gmail.com](mailto:losif.ai.2050@gmail.com)  
**LinkedIn:** [Yousef F.](https://www.linkedin.com/in/yousef-fawzi/)

*Open to opportunities, collaborations, and discussions about data science, HR analytics, and machine learning applications in business.*