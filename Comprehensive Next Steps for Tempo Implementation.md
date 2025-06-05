# Comprehensive Next Steps for Tempo Implementation

This document provides detailed guidance on the key next steps for customizing and extending the Tempo productivity and time management agent using the provided implementation tool. It covers customizing data models, implementing machine learning features, extending the API, adapting the frontend UI, and integrating external services.

---


# Customizing Data Models for Tempo Implementation

## Introduction

The Tempo implementation tool provides a foundational data structure designed for productivity and time management. However, to fully realize Tempo's potential for your specific use case, you'll need to customize and extend these data models. This guide provides a comprehensive approach to adapting the data models to your requirements while maintaining the integrity of the system architecture.

## Understanding the Existing Data Model Structure

The Tempo implementation includes several interconnected data models that form the core of the productivity management system:

### Database Schema Level

The PostgreSQL schema (`init-postgres.sql`) defines the fundamental data structures with tables for:

- Users and authentication
- Tasks and their attributes
- Calendar events
- Time blocks for focused work
- Time tracking entries
- Productivity metrics
- External integrations
- User preferences

The TimescaleDB extension (`init-timescaledb.sql`) adds time-series optimization for temporal data, particularly important for:

- Time entries tracking
- Productivity metrics over time
- Schedule analysis and pattern recognition

### Application Level Models

At the application level, Pydantic models provide data validation, serialization, and documentation for the API. These models are defined in the `backend/app/models/` directory and include:

- Data transfer objects (DTOs) for API requests and responses
- Internal models for business logic
- Database models that map to the SQL schema

## Customization Approach

When customizing data models for your specific Tempo implementation, follow this structured approach:

### 1. Identify Your Specific Requirements

Begin by documenting your specific requirements that extend beyond the base implementation. Consider:

- Additional task attributes specific to your workflow (e.g., energy levels, context tags)
- Custom calendar integration needs (e.g., meeting categories, preparation time)
- Specialized time tracking metrics (e.g., deep work scoring, interruption tracking)
- User preference extensions (e.g., preferred working hours, focus conditions)
- Organization-specific productivity metrics (e.g., alignment with OKRs, project-specific metrics)

### 2. Database Schema Extensions

To extend the database schema:

#### Adding New Fields to Existing Tables

For adding fields to existing tables, create migration scripts in the `db/migrations/` directory (create this if it doesn't exist):

```sql
-- Example migration: Add complexity_score to tasks table
ALTER TABLE tempo.tasks 
ADD COLUMN complexity_score INTEGER;

-- Add domain-specific fields
ALTER TABLE tempo.tasks
ADD COLUMN domain VARCHAR(100),
ADD COLUMN stakeholders TEXT[];
```

#### Creating New Related Tables

For entirely new entities that relate to existing ones:

```sql
-- Example: Create a projects table that tasks can belong to
CREATE TABLE IF NOT EXISTS tempo.projects (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES tempo.users(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    status VARCHAR(50) NOT NULL DEFAULT 'active',
    start_date DATE,
    target_end_date DATE,
    actual_end_date DATE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Add project relationship to tasks
ALTER TABLE tempo.tasks
ADD COLUMN project_id UUID REFERENCES tempo.projects(id) ON DELETE SET NULL;

-- Create appropriate indexes
CREATE INDEX IF NOT EXISTS idx_tasks_project_id ON tempo.tasks(project_id);
CREATE INDEX IF NOT EXISTS idx_projects_user_id ON tempo.projects(user_id);
```

#### TimescaleDB Considerations

When adding time-series data, consider whether it should be optimized with TimescaleDB:

```sql
-- Example: Create a new time-series table for focus metrics
CREATE TABLE IF NOT EXISTS tempo.focus_measurements (
    id UUID DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES tempo.users(id) ON DELETE CASCADE,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    focus_level INTEGER NOT NULL, -- 1-10 scale
    activity_context TEXT,
    environment_factors JSONB,
    notes TEXT
);

-- Convert to hypertable
SELECT create_hypertable('tempo.focus_measurements', 'timestamp', 
                         chunk_time_interval => interval '1 day',
                         if_not_exists => TRUE);
```

### 3. Pydantic Model Extensions

After extending the database schema, update the Pydantic models to reflect these changes:

#### Create or Modify Model Files

Create a new file in `backend/app/models/` for each major entity or modify existing ones:

```python
# Example: backend/app/models/project.py
from pydantic import BaseModel, Field, UUID4
from typing import Optional, List
from datetime import date
from enum import Enum

class ProjectStatus(str, Enum):
    active = "active"
    on_hold = "on_hold"
    completed = "completed"
    cancelled = "cancelled"

class ProjectBase(BaseModel):
    name: str
    description: Optional[str] = None
    status: ProjectStatus = ProjectStatus.active
    start_date: Optional[date] = None
    target_end_date: Optional[date] = None

class ProjectCreate(ProjectBase):
    pass

class ProjectUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    status: Optional[ProjectStatus] = None
    start_date: Optional[date] = None
    target_end_date: Optional[date] = None
    actual_end_date: Optional[date] = None

class Project(ProjectBase):
    id: UUID4
    user_id: UUID4
    actual_end_date: Optional[date] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True
```

#### Update Related Models

Ensure you update related models to reflect new relationships:

```python
# Example: Modify backend/app/models/task.py to include project relationship
from pydantic import UUID4
from typing import Optional

# Add to existing TaskBase class
class TaskBase(BaseModel):
    # ... existing fields
    project_id: Optional[UUID4] = None
    complexity_score: Optional[int] = None
    domain: Optional[str] = None
    stakeholders: Optional[List[str]] = None
```

### 4. SQLAlchemy ORM Model Updates

If you're using SQLAlchemy for database interactions, update the ORM models:

```python
# Example: backend/app/models/orm/project.py
from sqlalchemy import Column, String, Date, ForeignKey, Enum as SQLAEnum
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.orm import relationship
from datetime import datetime

from app.db.base import Base
from app.models.project import ProjectStatus

class Project(Base):
    __tablename__ = "projects"
    __table_args__ = {"schema": "tempo"}

    id = Column(UUID(as_uuid=True), primary_key=True, server_default=text("uuid_generate_v4()"))
    user_id = Column(UUID(as_uuid=True), ForeignKey("tempo.users.id", ondelete="CASCADE"), nullable=False)
    name = Column(String, nullable=False)
    description = Column(String)
    status = Column(SQLAEnum(ProjectStatus), default=ProjectStatus.active)
    start_date = Column(Date)
    target_end_date = Column(Date)
    actual_end_date = Column(Date)
    created_at = Column(TIMESTAMP(timezone=True), default=datetime.now)
    updated_at = Column(TIMESTAMP(timezone=True), default=datetime.now, onupdate=datetime.now)

    # Relationships
    tasks = relationship("Task", back_populates="project")
    user = relationship("User", back_populates="projects")
```

### 5. Database Migration Strategy

For managing schema changes over time, implement a migration strategy:

1. Create a `db/migrations/` directory if it doesn't exist
2. Use Alembic for Python-based migrations:

```python
# Install Alembic
# pip install alembic

# Initialize Alembic
# alembic init migrations

# Create a migration
# alembic revision -m "add_project_table"

# In the generated migration file:
def upgrade():
    op.create_table(
        'projects',
        sa.Column('id', postgresql.UUID(as_uuid=True), server_default=sa.text('uuid_generate_v4()'), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('name', sa.String(), nullable=False),
        # ... other columns
        sa.ForeignKeyConstraint(['user_id'], ['tempo.users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        schema='tempo'
    )
    op.add_column('tempo.tasks', sa.Column('project_id', postgresql.UUID(as_uuid=True), nullable=True))
    op.create_foreign_key(None, 'tasks', 'projects', ['project_id'], ['id'], source_schema='tempo', referent_schema='tempo', ondelete='SET NULL')
```

## Advanced Customization Techniques

### 1. Implementing Domain-Specific Validation

Extend Pydantic models with custom validators for domain-specific rules:

```python
from pydantic import validator

class TaskCreate(TaskBase):
    # ... existing fields
    
    @validator('estimated_duration')
    def validate_duration(cls, v, values):
        if v is not None and v <= 0:
            raise ValueError('Duration must be positive')
        if v is not None and v > 480:  # 8 hours
            raise ValueError('Tasks longer than 8 hours should be broken down')
        return v
        
    @validator('due_date')
    def validate_due_date(cls, v, values):
        if v is not None and v < datetime.now().date():
            raise ValueError('Due date cannot be in the past')
        return v
```

### 2. Creating Composite Models for Advanced Views

Develop composite models that combine data from multiple sources for rich views:

```python
class TaskWithContext(BaseModel):
    task: Task
    project: Optional[Project] = None
    related_calendar_events: List[CalendarEvent] = []
    time_entries: List[TimeEntry] = []
    total_time_spent: int = 0  # in minutes
    
    class Config:
        arbitrary_types_allowed = True
```

### 3. Implementing Temporal Data Patterns

For time-series data specific to productivity management:

```python
class ProductivityTimeWindow(BaseModel):
    start_time: datetime
    end_time: datetime
    focus_score: float
    tasks_completed: int
    interruption_count: int
    context_switches: int
    primary_activity: Optional[str] = None
    
    @property
    def duration_minutes(self) -> float:
        return (self.end_time - self.start_time).total_seconds() / 60
        
    @property
    def efficiency_score(self) -> float:
        if self.duration_minutes == 0:
            return 0
        return (self.tasks_completed * 10) / self.duration_minutes
```

## Integration with External Systems

When customizing data models to integrate with external systems:

### 1. Creating Adapter Models

Develop adapter models that translate between your system and external APIs:

```python
class GoogleCalendarEvent(BaseModel):
    google_id: str
    summary: str
    description: Optional[str] = None
    start: dict  # Google Calendar specific format
    end: dict
    attendees: Optional[List[dict]] = None
    
    def to_tempo_calendar_event(self, user_id: UUID4) -> CalendarEvent:
        """Convert Google Calendar event to Tempo calendar event"""
        start_time = parse_google_datetime(self.start)
        end_time = parse_google_datetime(self.end)
        
        return CalendarEvent(
            user_id=user_id,
            title=self.summary,
            description=self.description,
            start_time=start_time,
            end_time=end_time,
            calendar_source="google",
            external_id=self.google_id,
            attendees=[{"email": a.get("email"), "name": a.get("displayName")} 
                      for a in (self.attendees or [])]
        )
```

### 2. Handling Synchronization State

Add fields to track synchronization state with external systems:

```python
class ExternalSyncState(BaseModel):
    last_sync_time: datetime
    sync_status: str  # "success", "partial", "failed"
    error_message: Optional[str] = None
    items_synced: int
    items_failed: List[dict] = []
```

## Best Practices for Data Model Customization

1. **Maintain Backward Compatibility**: When extending models, avoid breaking changes to existing fields.

2. **Document All Changes**: Add comprehensive comments to both SQL and Python files explaining the purpose of new fields and entities.

3. **Use Enums for Constrained Values**: Define enums for fields with a fixed set of possible values to ensure data integrity.

4. **Consider Performance Implications**: When adding fields that will be frequently queried, create appropriate indexes.

5. **Implement Data Validation**: Use Pydantic validators to enforce business rules at the application level.

6. **Plan for Data Migration**: When changing existing models, develop a strategy for migrating existing data.

7. **Test Thoroughly**: Create unit tests for new models and integration tests for database interactions.

## Conclusion

Customizing data models for your Tempo implementation allows you to tailor the system to your specific productivity management needs. By following a structured approach to extending both the database schema and application models, you can maintain system integrity while adding powerful new capabilities.

Remember that data model changes often cascade through multiple system layers, from database to API to frontend. Always consider the full impact of your changes and update all affected components accordingly.
# Implementing ML Models in the Tempo Structure

## Introduction

Machine learning is a core component of the Tempo agent, enabling intelligent time management, task prioritization, and productivity optimization. This guide provides a comprehensive approach to implementing ML models within the Tempo structure, covering model selection, data preparation, training pipelines, and deployment strategies.

## Understanding the ML Requirements for Tempo

Before implementing ML models, it's important to understand the specific machine learning needs of a productivity and time management agent:

### Key ML Capabilities for Tempo

The Tempo agent requires several distinct ML capabilities:

1. **Time Estimation**: Predicting how long tasks will take based on historical data and task attributes
2. **Priority Optimization**: Suggesting task priorities based on deadlines, importance, and user patterns
3. **Schedule Generation**: Creating optimal daily/weekly schedules based on tasks, calendar events, and user preferences
4. **Productivity Pattern Recognition**: Identifying when, where, and how the user is most productive
5. **Interruption Prediction**: Forecasting potential interruptions and suggesting mitigation strategies
6. **Focus Session Optimization**: Recommending ideal duration and timing for deep work sessions

## ML Implementation Architecture

The Tempo implementation tool provides a structured approach to integrating ML capabilities:

### Directory Structure

The ML components are organized in the `backend/app/ml/` directory with the following structure:

```
backend/app/ml/
├── __init__.py
├── models/              # Model definitions
│   ├── __init__.py
│   ├── time_estimator.py
│   ├── priority_optimizer.py
│   ├── schedule_generator.py
│   └── pattern_recognizer.py
├── data/                # Data processing utilities
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── features.py
│   └── validation.py
├── training/            # Training pipelines
│   ├── __init__.py
│   ├── pipelines.py
│   └── evaluation.py
├── inference/           # Inference services
│   ├── __init__.py
│   ├── predictors.py
│   └── explainers.py
└── utils/               # Shared utilities
    ├── __init__.py
    ├── metrics.py
    └── visualization.py
```

## Step-by-Step Implementation Guide

### 1. Define Model Requirements and Specifications

Begin by clearly defining the requirements for each ML model:

```python
# Example: backend/app/ml/models/time_estimator.py
from typing import Dict, List, Optional, Union
import numpy as np
import torch
import torch.nn as nn
from pydantic import BaseModel

class TimeEstimatorConfig(BaseModel):
    """Configuration for the time estimation model."""
    input_features: int = 15  # Number of input features
    hidden_layers: List[int] = [64, 32]  # Hidden layer dimensions
    dropout_rate: float = 0.2
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    early_stopping_patience: int = 10
    
class TimeEstimatorInputFeatures(BaseModel):
    """Input features for time estimation."""
    task_title: str
    task_description: Optional[str] = None
    task_tags: List[str] = []
    estimated_duration: Optional[int] = None  # User's estimate in minutes
    priority: int
    due_date_distance: Optional[int] = None  # Days until due
    user_experience_level: Optional[int] = None  # 1-5 scale
    similar_tasks_avg_duration: Optional[float] = None
    time_of_day: Optional[int] = None  # Hour of day (0-23)
    day_of_week: Optional[int] = None  # 0-6
    context_switches_count: Optional[int] = None  # Prior to this task
    consecutive_work_time: Optional[int] = None  # Minutes already worked
    energy_level: Optional[int] = None  # User's reported energy (1-5)
    complexity_score: Optional[int] = None  # 1-5 scale
    
class TimeEstimatorOutput(BaseModel):
    """Output from the time estimation model."""
    predicted_duration: float  # In minutes
    confidence_score: float  # 0-1 scale
    prediction_range: Dict[str, float]  # {"min": min_duration, "max": max_duration}
    factors: Dict[str, float]  # Feature importance for this prediction
```

### 2. Implement Data Processing Pipeline

Create data processing utilities to transform raw data into model-ready features:

```python
# Example: backend/app/ml/data/preprocessing.py
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
from torch.utils.data import Dataset, DataLoader

class TaskDataProcessor:
    """Processes raw task data for ML models."""
    
    def __init__(self):
        self.text_vectorizer = TfidfVectorizer(max_features=100)
        self.numerical_scaler = StandardScaler()
        self.categorical_encoder = OneHotEncoder(sparse=False)
        self.fitted = False
        
    def fit(self, tasks_df: pd.DataFrame):
        """Fit the processors on training data."""
        # Process text features
        text_data = tasks_df['task_title'] + ' ' + tasks_df['task_description'].fillna('')
        self.text_vectorizer.fit(text_data)
        
        # Process numerical features
        numerical_features = [
            'estimated_duration', 'priority', 'due_date_distance',
            'user_experience_level', 'similar_tasks_avg_duration',
            'time_of_day', 'day_of_week', 'context_switches_count',
            'consecutive_work_time', 'energy_level', 'complexity_score'
        ]
        numerical_data = tasks_df[numerical_features].fillna(0)
        self.numerical_scaler.fit(numerical_data)
        
        # Process categorical features
        categorical_features = ['task_tags']
        # Flatten tag lists into a set of all possible tags
        all_tags = set()
        for tags in tasks_df['task_tags']:
            if isinstance(tags, list):
                all_tags.update(tags)
        
        # Create dummy tag presence features
        tag_dummies = pd.DataFrame()
        for tag in all_tags:
            tag_dummies[f'tag_{tag}'] = tasks_df['task_tags'].apply(
                lambda x: 1 if tag in x else 0 if isinstance(x, list) else 0
            )
        
        self.categorical_encoder.fit(tag_dummies)
        self.fitted = True
        
    def transform(self, tasks_df: pd.DataFrame) -> np.ndarray:
        """Transform raw task data into model features."""
        if not self.fitted:
            raise ValueError("DataProcessor must be fitted before transform")
            
        # Process text features
        text_data = tasks_df['task_title'] + ' ' + tasks_df['task_description'].fillna('')
        text_features = self.text_vectorizer.transform(text_data).toarray()
        
        # Process numerical features
        numerical_features = [
            'estimated_duration', 'priority', 'due_date_distance',
            'user_experience_level', 'similar_tasks_avg_duration',
            'time_of_day', 'day_of_week', 'context_switches_count',
            'consecutive_work_time', 'energy_level', 'complexity_score'
        ]
        numerical_data = tasks_df[numerical_features].fillna(0)
        numerical_features = self.numerical_scaler.transform(numerical_data)
        
        # Process categorical features (tags)
        all_tags = set()
        for tags in tasks_df['task_tags']:
            if isinstance(tags, list):
                all_tags.update(tags)
        
        tag_dummies = pd.DataFrame()
        for tag in all_tags:
            tag_dummies[f'tag_{tag}'] = tasks_df['task_tags'].apply(
                lambda x: 1 if tag in x else 0 if isinstance(x, list) else 0
            )
        
        categorical_features = self.categorical_encoder.transform(tag_dummies)
        
        # Combine all features
        combined_features = np.hstack([
            text_features, 
            numerical_features, 
            categorical_features
        ])
        
        return combined_features
        
class TaskDataset(Dataset):
    """PyTorch dataset for task data."""
    
    def __init__(self, features: np.ndarray, targets: np.ndarray):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        
    def __len__(self):
        return len(self.features)
        
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]
        
def create_dataloaders(
    features: np.ndarray, 
    targets: np.ndarray, 
    batch_size: int = 32, 
    train_ratio: float = 0.8
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders."""
    # Split data
    n_samples = len(features)
    n_train = int(n_samples * train_ratio)
    
    indices = np.random.permutation(n_samples)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    # Create datasets
    train_dataset = TaskDataset(
        features[train_indices], 
        targets[train_indices]
    )
    val_dataset = TaskDataset(
        features[val_indices], 
        targets[val_indices]
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    return train_loader, val_loader
```

### 3. Implement Model Architecture

Define the model architecture based on your requirements:

```python
# Example: backend/app/ml/models/time_estimator.py (continued)
class TimeEstimatorModel(nn.Module):
    """Neural network for estimating task completion time."""
    
    def __init__(self, config: TimeEstimatorConfig):
        super().__init__()
        self.config = config
        
        # Build layers
        layers = []
        input_dim = config.input_features
        
        for hidden_dim in config.hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config.dropout_rate))
            input_dim = hidden_dim
            
        # Output layer
        layers.append(nn.Linear(input_dim, 1))
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x).squeeze()
        
    def predict_with_uncertainty(self, x, n_samples=10):
        """Monte Carlo dropout prediction with uncertainty."""
        self.train()  # Enable dropout
        
        predictions = []
        for _ in range(n_samples):
            with torch.no_grad():
                pred = self.forward(x).cpu().numpy()
                predictions.append(pred)
                
        predictions = np.array(predictions)
        mean_prediction = np.mean(predictions, axis=0)
        std_prediction = np.std(predictions, axis=0)
        
        return mean_prediction, std_prediction
```

### 4. Implement Training Pipeline

Create a training pipeline that handles model training, validation, and checkpointing:

```python
# Example: backend/app/ml/training/pipelines.py
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
import mlflow
from datetime import datetime

from app.ml.models.time_estimator import TimeEstimatorModel, TimeEstimatorConfig
from app.ml.data.preprocessing import TaskDataProcessor, create_dataloaders

class TimeEstimatorTrainer:
    """Training pipeline for the time estimation model."""
    
    def __init__(
        self, 
        config: TimeEstimatorConfig,
        model_dir: str = "models/time_estimator",
        track_mlflow: bool = True
    ):
        self.config = config
        self.model_dir = model_dir
        self.track_mlflow = track_mlflow
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create model
        self.model = TimeEstimatorModel(config).to(self.device)
        
        # Create optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=config.learning_rate
        )
        
        # Create loss function
        self.criterion = nn.MSELoss()
        
        # Create data processor
        self.data_processor = TaskDataProcessor()
        
        # Ensure model directory exists
        os.makedirs(model_dir, exist_ok=True)
        
    def train(
        self, 
        tasks_df: pd.DataFrame, 
        actual_durations: np.ndarray
    ) -> Dict[str, Any]:
        """Train the model on task data."""
        # Prepare data
        self.data_processor.fit(tasks_df)
        features = self.data_processor.transform(tasks_df)
        
        # Create dataloaders
        train_loader, val_loader = create_dataloaders(
            features, 
            actual_durations, 
            batch_size=self.config.batch_size
        )
        
        # Initialize tracking
        if self.track_mlflow:
            mlflow.start_run()
            mlflow.log_params(self.config.dict())
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []
        
        for epoch in range(self.config.epochs):
            # Training phase
            self.model.train()
            epoch_loss = 0
            
            for batch_features, batch_targets in train_loader:
                batch_features = batch_features.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(batch_features)
                loss = self.criterion(outputs, batch_targets)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                
            train_loss = epoch_loss / len(train_loader)
            train_losses.append(train_loss)
            
            # Validation phase
            self.model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for batch_features, batch_targets in val_loader:
                    batch_features = batch_features.to(self.device)
                    batch_targets = batch_targets.to(self.device)
                    
                    outputs = self.model(batch_features)
                    loss = self.criterion(outputs, batch_targets)
                    val_loss += loss.item()
                    
            val_loss = val_loss / len(val_loader)
            val_losses.append(val_loss)
            
            # Log metrics
            if self.track_mlflow:
                mlflow.log_metrics({
                    "train_loss": train_loss,
                    "val_loss": val_loss
                }, step=epoch)
                
            print(f"Epoch {epoch+1}/{self.config.epochs} - "
                  f"Train Loss: {train_loss:.4f} - "
                  f"Val Loss: {val_loss:.4f}")
                
            # Check for early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Save best model
                self._save_model()
            else:
                patience_counter += 1
                if patience_counter >= self.config.early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
                    
        # End tracking
        if self.track_mlflow:
            mlflow.end_run()
            
        # Return training history
        return {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "best_val_loss": best_val_loss,
            "epochs_trained": len(train_losses)
        }
        
    def _save_model(self):
        """Save the model and its configuration."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(self.model_dir, f"model_{timestamp}.pt")
        config_path = os.path.join(self.model_dir, f"config_{timestamp}.json")
        processor_path = os.path.join(self.model_dir, f"processor_{timestamp}.pkl")
        
        # Save model weights
        torch.save(self.model.state_dict(), model_path)
        
        # Save config
        with open(config_path, "w") as f:
            json.dump(self.config.dict(), f)
            
        # Save data processor
        import pickle
        with open(processor_path, "wb") as f:
            pickle.dump(self.data_processor, f)
            
        # Save latest model pointer
        with open(os.path.join(self.model_dir, "latest_model.txt"), "w") as f:
            f.write(timestamp)
            
        print(f"Model saved to {model_path}")
```

### 5. Implement Inference Service

Create an inference service that loads trained models and makes predictions:

```python
# Example: backend/app/ml/inference/predictors.py
import os
import json
import pickle
import torch
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional

from app.ml.models.time_estimator import TimeEstimatorModel, TimeEstimatorConfig, TimeEstimatorOutput

class TimeEstimatorPredictor:
    """Inference service for time estimation."""
    
    def __init__(self, model_dir: str = "models/time_estimator"):
        self.model_dir = model_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load latest model
        self.model, self.config, self.processor = self._load_latest_model()
        
    def _load_latest_model(self):
        """Load the latest trained model."""
        # Get latest model timestamp
        try:
            with open(os.path.join(self.model_dir, "latest_model.txt"), "r") as f:
                timestamp = f.read().strip()
        except FileNotFoundError:
            raise ValueError(f"No trained model found in {self.model_dir}")
            
        # Load config
        config_path = os.path.join(self.model_dir, f"config_{timestamp}.json")
        with open(config_path, "r") as f:
            config_dict = json.load(f)
            config = TimeEstimatorConfig(**config_dict)
            
        # Load model
        model_path = os.path.join(self.model_dir, f"model_{timestamp}.pt")
        model = TimeEstimatorModel(config).to(self.device)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        
        # Load processor
        processor_path = os.path.join(self.model_dir, f"processor_{timestamp}.pkl")
        with open(processor_path, "rb") as f:
            processor = pickle.load(f)
            
        return model, config, processor
        
    def predict(self, task_data: Dict[str, Any]) -> TimeEstimatorOutput:
        """Predict time for a single task."""
        # Convert to DataFrame for processor
        task_df = pd.DataFrame([task_data])
        
        # Process features
        features = self.processor.transform(task_df)
        features_tensor = torch.tensor(features, dtype=torch.float32).to(self.device)
        
        # Get prediction with uncertainty
        mean, std = self.model.predict_with_uncertainty(features_tensor)
        
        # Calculate prediction range (95% confidence interval)
        min_duration = max(0, mean - 1.96 * std)
        max_duration = mean + 1.96 * std
        
        # Calculate confidence score (inverse of normalized std)
        confidence = 1.0 / (1.0 + std / mean) if mean > 0 else 0.5
        
        # Get feature importance (simple approach)
        # For a more sophisticated approach, use SHAP or similar
        importance = self._calculate_feature_importance(features_tensor)
        
        return TimeEstimatorOutput(
            predicted_duration=float(mean),
            confidence_score=float(confidence),
            prediction_range={"min": float(min_duration), "max": float(max_duration)},
            factors=importance
        )
        
    def _calculate_feature_importance(self, features_tensor):
        """Calculate simple feature importance."""
        # This is a simplified approach
        # For production, consider using SHAP or other explainability tools
        with torch.no_grad():
            baseline = self.model(features_tensor).item()
            importance = {}
            
            # Perturb each feature and measure impact
            for i in range(features_tensor.shape[1]):
                perturbed = features_tensor.clone()
                perturbed[0, i] = 0  # Zero out this feature
                
                new_pred = self.model(perturbed).item()
                impact = abs(baseline - new_pred)
                
                feature_name = f"feature_{i}"
                importance[feature_name] = float(impact)
                
            # Normalize
            total = sum(importance.values())
            if total > 0:
                for k in importance:
                    importance[k] /= total
                    
        return importance
```

### 6. Integrate ML Services with API

Connect the ML services to the API endpoints:

```python
# Example: backend/app/routers/tasks.py
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from typing import List, Optional
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.task import TaskCreate, Task, TaskUpdate
from app.services.task_service import TaskService
from app.ml.inference.predictors import TimeEstimatorPredictor

router = APIRouter()

# Initialize predictor
time_predictor = TimeEstimatorPredictor()

@router.post("/", response_model=Task)
def create_task(
    task: TaskCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Create a new task with ML-estimated duration."""
    task_service = TaskService(db)
    
    # If no duration estimate provided, use ML model
    if task.estimated_duration is None:
        # Convert task to format expected by predictor
        task_data = {
            "task_title": task.title,
            "task_description": task.description,
            "task_tags": task.tags,
            "priority": task.priority,
            # Add other fields as needed
        }
        
        # Get prediction
        try:
            prediction = time_predictor.predict(task_data)
            task.estimated_duration = int(prediction.predicted_duration)
            task.ml_confidence = prediction.confidence_score
        except Exception as e:
            # Log error but continue with creation
            print(f"Error predicting duration: {e}")
            # Use a default value
            task.estimated_duration = 30  # 30 minutes default
            
    # Create task
    db_task = task_service.create_task(task)
    
    # Schedule background training if needed
    background_tasks.add_task(
        task_service.update_ml_training_queue,
        db_task.id
    )
    
    return db_task
```

### 7. Implement Continuous Learning

Set up a background process for continuous model improvement:

```python
# Example: backend/app/services/ml_training_service.py
import pandas as pd
from sqlalchemy.orm import Session
from fastapi import BackgroundTasks
import logging
from datetime import datetime, timedelta

from app.db.session import SessionLocal
from app.ml.training.pipelines import TimeEstimatorTrainer
from app.ml.models.time_estimator import TimeEstimatorConfig

logger = logging.getLogger(__name__)

class MLTrainingService:
    """Service for managing ML model training."""
    
    def __init__(self):
        self.training_queue = set()
        self.last_training_time = datetime.now() - timedelta(days=1)
        
    def add_to_training_queue(self, task_id: int):
        """Add a task to the training queue."""
        self.training_queue.add(task_id)
        
        # Check if we should trigger training
        self._check_training_trigger()
        
    def _check_training_trigger(self):
        """Check if we should trigger model retraining."""
        # Train if enough new data or enough time has passed
        should_train = (
            len(self.training_queue) >= 50 or  # At least 50 new tasks
            (datetime.now() - self.last_training_time) > timedelta(days=1)  # Daily training
        )
        
        if should_train and len(self.training_queue) > 0:
            # Trigger background training
            background_tasks = BackgroundTasks()
            background_tasks.add_task(self.retrain_models)
            
    def retrain_models(self):
        """Retrain ML models with latest data."""
        logger.info("Starting model retraining")
        
        try:
            # Get database session
            db = SessionLocal()
            
            # Get completed tasks with actual duration
            query = """
                SELECT 
                    t.id, t.title, t.description, t.priority, t.estimated_duration,
                    t.due_date, t.tags, t.context, t.energy_required, t.complexity_score,
                    te.duration as actual_duration
                FROM 
                    tempo.tasks t
                JOIN 
                    tempo.time_entries te ON t.id = te.task_id
                WHERE 
                    te.duration IS NOT NULL
                    AND t.status = 'completed'
                ORDER BY 
                    t.completed_at DESC
                LIMIT 1000
            """
            
            tasks_df = pd.read_sql(query, db.bind)
            
            # Skip if not enough data
            if len(tasks_df) < 10:
                logger.info("Not enough data for retraining")
                return
                
            # Prepare training data
            actual_durations = tasks_df['actual_duration'].values / 60  # Convert to minutes
            
            # Configure and train model
            config = TimeEstimatorConfig(
                input_features=len(tasks_df.columns) - 1,  # Exclude actual_duration
                hidden_layers=[64, 32],
                epochs=50
            )
            
            trainer = TimeEstimatorTrainer(config)
            training_results = trainer.train(tasks_df, actual_durations)
            
            logger.info(f"Model retraining complete: {training_results}")
            
            # Update state
            self.training_queue.clear()
            self.last_training_time = datetime.now()
            
        except Exception as e:
            logger.error(f"Error during model retraining: {e}")
            
        finally:
            db.close()

# Create singleton instance
ml_training_service = MLTrainingService()
```

## Advanced ML Implementation Techniques

### 1. Hybrid Models for Cold Start

Implement hybrid models that combine rules and ML for new users with limited data:

```python
class HybridTimeEstimator:
    """Combines rule-based and ML-based time estimation."""
    
    def __init__(self, ml_predictor, min_data_points=50):
        self.ml_predictor = ml_predictor
        self.min_data_points = min_data_points
        
    def predict(self, task_data, user_task_count):
        """Predict using appropriate method based on data availability."""
        if user_task_count >= self.min_data_points:
            # Enough data for ML prediction
            return self.ml_predictor.predict(task_data)
        else:
            # Use rule-based fallback with ML confidence adjustment
            ml_prediction = self.ml_predictor.predict(task_data)
            rule_prediction = self._rule_based_prediction(task_data)
            
            # Blend predictions based on available data
            blend_ratio = min(user_task_count / self.min_data_points, 0.8)
            blended_duration = (
                blend_ratio * ml_prediction.predicted_duration +
                (1 - blend_ratio) * rule_prediction
            )
            
            # Adjust confidence
            adjusted_confidence = ml_prediction.confidence_score * blend_ratio
            
            return TimeEstimatorOutput(
                predicted_duration=blended_duration,
                confidence_score=adjusted_confidence,
                prediction_range={
                    "min": blended_duration * 0.5,
                    "max": blended_duration * 1.5
                },
                factors=ml_prediction.factors
            )
            
    def _rule_based_prediction(self, task_data):
        """Rule-based prediction for cold start."""
        # Base estimate
        base_minutes = 30
        
        # Adjust for priority
        priority_factor = {
            1: 1.5,  # High priority often underestimated
            2: 1.2,
            3: 1.0,  # Neutral
            4: 0.9,
            5: 0.8   # Low priority often overestimated
        }.get(task_data.get("priority", 3), 1.0)
        
        # Adjust for complexity
        complexity_factor = {
            1: 0.7,
            2: 0.85,
            3: 1.0,
            4: 1.3,
            5: 1.7
        }.get(task_data.get("complexity_score", 3), 1.0)
        
        # Adjust for description length (proxy for complexity)
        description = task_data.get("task_description", "")
        if description:
            words = len(description.split())
            desc_factor = min(1.5, max(0.8, words / 100 + 0.8))
        else:
            desc_factor = 1.0
            
        return base_minutes * priority_factor * complexity_factor * desc_factor
```

### 2. Implementing Transfer Learning

Leverage pre-trained models to improve performance with limited data:

```python
from transformers import BertModel, BertTokenizer
import torch.nn as nn

class TaskEmbeddingModel(nn.Module):
    """Uses BERT to create task embeddings."""
    
    def __init__(self, bert_model_name="bert-base-uncased"):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.bert = BertModel.from_pretrained(bert_model_name)
        
        # Freeze BERT parameters for transfer learning
        for param in self.bert.parameters():
            param.requires_grad = False
            
        # Add task-specific layers
        self.task_projection = nn.Linear(768, 128)
        self.activation = nn.ReLU()
        
    def forward(self, text):
        """Create embeddings from task text."""
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=128
        )
        
        outputs = self.bert(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        
        task_embedding = self.activation(self.task_projection(cls_embedding))
        return task_embedding
```

### 3. Implementing Multi-Task Learning

Train models that simultaneously learn multiple related productivity tasks:

```python
class ProductivityMultiTaskModel(nn.Module):
    """Multi-task model for productivity predictions."""
    
    def __init__(self, input_dim, shared_dims=[128, 64], task_specific_dims=[32]):
        super().__init__()
        
        # Shared layers
        shared_layers = []
        current_dim = input_dim
        
        for dim in shared_dims:
            shared_layers.append(nn.Linear(current_dim, dim))
            shared_layers.append(nn.ReLU())
            shared_layers.append(nn.BatchNorm1d(dim))
            current_dim = dim
            
        self.shared_network = nn.Sequential(*shared_layers)
        
        # Task-specific networks
        self.time_estimation_network = self._create_task_network(
            current_dim, task_specific_dims, 1
        )
        self.priority_network = self._create_task_network(
            current_dim, task_specific_dims, 1
        )
        self.focus_score_network = self._create_task_network(
            current_dim, task_specific_dims, 1
        )
        
    def _create_task_network(self, input_dim, hidden_dims, output_dim):
        layers = []
        current_dim = input_dim
        
        for dim in hidden_dims:
            layers.append(nn.Linear(current_dim, dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(dim))
            current_dim = dim
            
        layers.append(nn.Linear(current_dim, output_dim))
        return nn.Sequential(*layers)
        
    def forward(self, x):
        """Forward pass for all tasks."""
        shared_features = self.shared_network(x)
        
        time_estimate = self.time_estimation_network(shared_features)
        priority_score = self.priority_network(shared_features)
        focus_score = self.focus_score_network(shared_features)
        
        return {
            "time_estimate": time_estimate.squeeze(),
            "priority_score": priority_score.squeeze(),
            "focus_score": focus_score.squeeze()
        }
```

### 4. Implementing Explainable AI

Add explainability to help users understand model predictions:

```python
import shap
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64

class ModelExplainer:
    """Explains model predictions using SHAP."""
    
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names
        
        # Create explainer
        self.explainer = shap.Explainer(
            lambda x: self.model(torch.tensor(x, dtype=torch.float32)).detach().numpy(),
            feature_names=feature_names
        )
        
    def explain_prediction(self, features):
        """Generate explanation for a prediction."""
        # Get SHAP values
        shap_values = self.explainer(features)
        
        # Create explanation dict
        explanation = {}
        for i, name in enumerate(self.feature_names):
            explanation[name] = float(shap_values.values[0, i])
            
        return explanation
        
    def generate_explanation_plot(self, features):
        """Generate visual explanation plot."""
        # Get SHAP values
        shap_values = self.explainer(features)
        
        # Create plot
        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(shap_values[0], show=False)
        
        # Save to bytes
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close()
        buf.seek(0)
        
        # Convert to base64
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        return f"data:image/png;base64,{img_str}"
```

## Best Practices for ML Implementation

### 1. Data Quality and Preparation

- **Handle Missing Data**: Implement robust strategies for missing values in productivity data
- **Feature Engineering**: Create domain-specific features relevant to productivity (e.g., context switches, energy levels)
- **Data Augmentation**: For limited data, use techniques like synthetic task generation
- **Validation Strategy**: Use time-based validation to reflect real-world usage patterns

### 2. Model Selection and Training

- **Start Simple**: Begin with simpler models (linear regression, random forests) before complex neural networks
- **Regularization**: Apply appropriate regularization to prevent overfitting on limited productivity data
- **Hyperparameter Tuning**: Use Bayesian optimization for efficient hyperparameter search
- **Evaluation Metrics**: Focus on metrics relevant to productivity (e.g., mean absolute error in minutes)

### 3. Deployment and Monitoring

- **Model Versioning**: Maintain clear versioning for all deployed models
- **A/B Testing**: Test new models against existing ones before full deployment
- **Monitoring**: Track model drift and performance degradation over time
- **Feedback Loop**: Incorporate user feedback to improve model predictions

### 4. Ethical Considerations

- **Privacy**: Ensure user productivity data is handled securely and anonymized for training
- **Transparency**: Make clear when predictions are ML-based vs. rule-based
- **User Control**: Allow users to override ML predictions and provide feedback
- **Bias Mitigation**: Regularly audit for biases in productivity predictions

## Conclusion

Implementing ML models within the Tempo structure requires a thoughtful approach to data processing, model architecture, training pipelines, and deployment. By following the structured approach outlined in this guide, you can create intelligent productivity features that learn from user behavior and provide increasingly accurate predictions over time.

Remember that productivity prediction is highly personal, and models should adapt to individual users' patterns. The hybrid approach combining rules and ML is particularly valuable during the initial deployment phase when user data is limited.

As you implement these ML capabilities, focus on creating a seamless user experience where the intelligence augments the user's productivity without becoming intrusive or rigid.
# Extending API Endpoints for Additional Functionality in Tempo

## Introduction

The Tempo implementation tool provides a foundational API structure based on FastAPI, but to fully realize the potential of your productivity and time management agent, you'll need to extend and customize the API endpoints. This guide provides a comprehensive approach to adding new endpoints, enhancing existing ones, and ensuring your API remains well-structured, documented, and maintainable as it grows.

## Understanding the Existing API Structure

Before extending the API, it's important to understand how the existing endpoints are organized:

### Core API Organization

The Tempo API follows a modular structure with endpoints grouped by domain:

- `/api/tasks` - Task management endpoints
- `/api/calendar` - Calendar event endpoints
- `/api/analytics` - Productivity analytics endpoints
- `/api/integrations` - External service integration endpoints

Each domain is implemented as a separate router in the `backend/app/routers/` directory, with the main FastAPI application in `backend/app/main.py` combining these routers.

### API Layer Architecture

The API implementation follows a layered architecture:

1. **Router Layer** (`/routers`): Handles HTTP requests/responses, input validation, and route definitions
2. **Service Layer** (`/services`): Contains business logic and orchestrates operations
3. **Repository Layer** (`/repositories`): Manages data access and persistence
4. **Model Layer** (`/models`): Defines data structures and validation rules

This separation of concerns makes it easier to extend functionality while maintaining code quality.

## Step-by-Step Guide to Extending API Endpoints

### 1. Identify New Functionality Requirements

Begin by clearly defining what new functionality you need to add. For example, let's say you want to add support for:

- Focus sessions (pomodoro-style time blocks)
- Task dependencies and relationships
- Team collaboration features
- Advanced productivity analytics

For each feature, identify:
- What data needs to be stored
- What operations users need to perform
- How it integrates with existing features
- What security considerations apply

### 2. Define New Data Models

Before implementing API endpoints, define the Pydantic models that will represent the request and response data:

```python
# Example: backend/app/models/focus_session.py
from pydantic import BaseModel, Field, UUID4
from typing import Optional, List
from datetime import datetime
from enum import Enum

class FocusSessionStatus(str, Enum):
    planned = "planned"
    in_progress = "in_progress"
    completed = "completed"
    interrupted = "interrupted"

class FocusSessionBase(BaseModel):
    title: str
    planned_duration_minutes: int = Field(ge=1, le=180)
    task_id: Optional[UUID4] = None
    description: Optional[str] = None

class FocusSessionCreate(FocusSessionBase):
    scheduled_start: Optional[datetime] = None

class FocusSessionUpdate(BaseModel):
    title: Optional[str] = None
    status: Optional[FocusSessionStatus] = None
    actual_duration_minutes: Optional[int] = Field(None, ge=0)
    interruption_count: Optional[int] = Field(None, ge=0)
    notes: Optional[str] = None

class FocusSession(FocusSessionBase):
    id: UUID4
    user_id: UUID4
    status: FocusSessionStatus
    scheduled_start: Optional[datetime] = None
    actual_start: Optional[datetime] = None
    actual_end: Optional[datetime] = None
    actual_duration_minutes: Optional[int] = None
    interruption_count: int = 0
    productivity_score: Optional[float] = None
    notes: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True
```

### 3. Create or Update Database Models

If your new functionality requires database changes, update your database models accordingly:

```python
# Example: backend/app/models/orm/focus_session.py
from sqlalchemy import Column, String, Integer, Float, ForeignKey, Enum as SQLAEnum
from sqlalchemy.dialects.postgresql import UUID, TIMESTAMP
from sqlalchemy.orm import relationship
from sqlalchemy.sql import text
from datetime import datetime

from app.db.base import Base
from app.models.focus_session import FocusSessionStatus

class FocusSession(Base):
    __tablename__ = "focus_sessions"
    __table_args__ = {"schema": "tempo"}

    id = Column(UUID(as_uuid=True), primary_key=True, server_default=text("uuid_generate_v4()"))
    user_id = Column(UUID(as_uuid=True), ForeignKey("tempo.users.id", ondelete="CASCADE"), nullable=False)
    task_id = Column(UUID(as_uuid=True), ForeignKey("tempo.tasks.id", ondelete="SET NULL"), nullable=True)
    title = Column(String, nullable=False)
    description = Column(String, nullable=True)
    status = Column(SQLAEnum(FocusSessionStatus), nullable=False, default=FocusSessionStatus.planned)
    planned_duration_minutes = Column(Integer, nullable=False)
    scheduled_start = Column(TIMESTAMP(timezone=True), nullable=True)
    actual_start = Column(TIMESTAMP(timezone=True), nullable=True)
    actual_end = Column(TIMESTAMP(timezone=True), nullable=True)
    actual_duration_minutes = Column(Integer, nullable=True)
    interruption_count = Column(Integer, nullable=False, default=0)
    productivity_score = Column(Float, nullable=True)
    notes = Column(String, nullable=True)
    created_at = Column(TIMESTAMP(timezone=True), nullable=False, default=datetime.now)
    updated_at = Column(TIMESTAMP(timezone=True), nullable=False, default=datetime.now, onupdate=datetime.now)

    # Relationships
    user = relationship("User", back_populates="focus_sessions")
    task = relationship("Task", back_populates="focus_sessions")
```

Don't forget to create a database migration for your changes:

```python
# Using Alembic for migrations
# alembic revision -m "add_focus_sessions_table"

def upgrade():
    op.create_table(
        'focus_sessions',
        sa.Column('id', postgresql.UUID(as_uuid=True), server_default=sa.text('uuid_generate_v4()'), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('task_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('title', sa.String(), nullable=False),
        # ... other columns
        sa.ForeignKeyConstraint(['user_id'], ['tempo.users.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['task_id'], ['tempo.tasks.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id'),
        schema='tempo'
    )
    op.create_index(op.f('ix_tempo_focus_sessions_user_id'), 'focus_sessions', ['user_id'], unique=False, schema='tempo')
```

### 4. Implement Service Layer Logic

Create a service class to handle the business logic for your new feature:

```python
# Example: backend/app/services/focus_session_service.py
from typing import List, Optional, Dict, Any
from uuid import UUID
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from fastapi import HTTPException, status

from app.models.focus_session import FocusSessionCreate, FocusSessionUpdate, FocusSessionStatus
from app.models.orm.focus_session import FocusSession as DBFocusSession
from app.services.notification_service import NotificationService

class FocusSessionService:
    def __init__(self, db: Session):
        self.db = db
        self.notification_service = NotificationService(db)
        
    def create_focus_session(self, user_id: UUID, session_data: FocusSessionCreate) -> DBFocusSession:
        """Create a new focus session."""
        db_session = DBFocusSession(
            user_id=user_id,
            task_id=session_data.task_id,
            title=session_data.title,
            description=session_data.description,
            planned_duration_minutes=session_data.planned_duration_minutes,
            scheduled_start=session_data.scheduled_start,
            status=FocusSessionStatus.planned
        )
        
        self.db.add(db_session)
        self.db.commit()
        self.db.refresh(db_session)
        
        # Schedule notifications if needed
        if session_data.scheduled_start:
            self.notification_service.schedule_focus_session_reminder(
                user_id=user_id,
                session_id=db_session.id,
                scheduled_time=session_data.scheduled_start - timedelta(minutes=5)
            )
            
        return db_session
        
    def get_focus_sessions(self, user_id: UUID, skip: int = 0, limit: int = 100) -> List[DBFocusSession]:
        """Get all focus sessions for a user."""
        return self.db.query(DBFocusSession)\
            .filter(DBFocusSession.user_id == user_id)\
            .order_by(DBFocusSession.created_at.desc())\
            .offset(skip)\
            .limit(limit)\
            .all()
            
    def get_focus_session(self, user_id: UUID, session_id: UUID) -> DBFocusSession:
        """Get a specific focus session."""
        session = self.db.query(DBFocusSession)\
            .filter(DBFocusSession.id == session_id, DBFocusSession.user_id == user_id)\
            .first()
            
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Focus session not found"
            )
            
        return session
        
    def update_focus_session(self, user_id: UUID, session_id: UUID, session_data: FocusSessionUpdate) -> DBFocusSession:
        """Update a focus session."""
        session = self.get_focus_session(user_id, session_id)
        
        # Update fields
        for key, value in session_data.dict(exclude_unset=True).items():
            setattr(session, key, value)
            
        # Handle status transitions
        if session_data.status == FocusSessionStatus.in_progress and session.actual_start is None:
            session.actual_start = datetime.now()
        elif session_data.status == FocusSessionStatus.completed and session.actual_end is None:
            session.actual_end = datetime.now()
            if session.actual_start:
                # Calculate actual duration
                duration = (session.actual_end - session.actual_start).total_seconds() / 60
                session.actual_duration_minutes = int(duration)
                
                # Calculate productivity score (simple example)
                interruption_penalty = session.interruption_count * 0.1
                duration_ratio = min(1.0, session.actual_duration_minutes / session.planned_duration_minutes)
                session.productivity_score = max(0, min(10, 10 * duration_ratio * (1 - interruption_penalty)))
        
        self.db.commit()
        self.db.refresh(session)
        return session
        
    def delete_focus_session(self, user_id: UUID, session_id: UUID) -> bool:
        """Delete a focus session."""
        session = self.get_focus_session(user_id, session_id)
        self.db.delete(session)
        self.db.commit()
        return True
        
    def start_focus_session(self, user_id: UUID, session_id: UUID) -> DBFocusSession:
        """Start a focus session."""
        return self.update_focus_session(
            user_id, 
            session_id, 
            FocusSessionUpdate(status=FocusSessionStatus.in_progress)
        )
        
    def complete_focus_session(self, user_id: UUID, session_id: UUID, notes: Optional[str] = None) -> DBFocusSession:
        """Complete a focus session."""
        return self.update_focus_session(
            user_id,
            session_id,
            FocusSessionUpdate(status=FocusSessionStatus.completed, notes=notes)
        )
        
    def record_interruption(self, user_id: UUID, session_id: UUID) -> DBFocusSession:
        """Record an interruption during a focus session."""
        session = self.get_focus_session(user_id, session_id)
        session.interruption_count += 1
        self.db.commit()
        self.db.refresh(session)
        return session
        
    def get_focus_statistics(self, user_id: UUID, days: int = 30) -> Dict[str, Any]:
        """Get focus session statistics for a user."""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Get all completed sessions in the period
        sessions = self.db.query(DBFocusSession)\
            .filter(
                DBFocusSession.user_id == user_id,
                DBFocusSession.status == FocusSessionStatus.completed,
                DBFocusSession.actual_end >= cutoff_date
            )\
            .all()
            
        if not sessions:
            return {
                "total_sessions": 0,
                "total_focus_time": 0,
                "avg_session_duration": 0,
                "avg_productivity_score": 0,
                "total_interruptions": 0,
                "completion_rate": 0
            }
            
        # Calculate statistics
        total_sessions = len(sessions)
        total_focus_time = sum(s.actual_duration_minutes or 0 for s in sessions)
        avg_session_duration = total_focus_time / total_sessions if total_sessions > 0 else 0
        avg_productivity_score = sum(s.productivity_score or 0 for s in sessions) / total_sessions if total_sessions > 0 else 0
        total_interruptions = sum(s.interruption_count for s in sessions)
        
        # Get all sessions (including non-completed)
        all_sessions = self.db.query(DBFocusSession)\
            .filter(
                DBFocusSession.user_id == user_id,
                DBFocusSession.created_at >= cutoff_date
            )\
            .all()
            
        completion_rate = total_sessions / len(all_sessions) if all_sessions else 0
        
        return {
            "total_sessions": total_sessions,
            "total_focus_time": total_focus_time,
            "avg_session_duration": avg_session_duration,
            "avg_productivity_score": avg_productivity_score,
            "total_interruptions": total_interruptions,
            "completion_rate": completion_rate
        }
```

### 5. Create API Router

Now implement the API router with endpoints for your new feature:

```python
# Example: backend/app/routers/focus_sessions.py
from fastapi import APIRouter, Depends, HTTPException, status, Query
from typing import List, Optional
from uuid import UUID
from datetime import datetime, timedelta

from app.models.focus_session import FocusSession, FocusSessionCreate, FocusSessionUpdate
from app.services.focus_session_service import FocusSessionService
from app.services.auth import get_current_user
from app.db.session import get_db
from sqlalchemy.orm import Session

router = APIRouter()

@router.post("/", response_model=FocusSession)
def create_focus_session(
    session_data: FocusSessionCreate,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Create a new focus session.
    
    A focus session represents a dedicated time block for focused work,
    optionally associated with a specific task.
    """
    service = FocusSessionService(db)
    return service.create_focus_session(current_user.id, session_data)

@router.get("/", response_model=List[FocusSession])
def get_focus_sessions(
    skip: int = 0,
    limit: int = 100,
    status: Optional[str] = None,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get all focus sessions for the current user.
    
    Optionally filter by status.
    """
    service = FocusSessionService(db)
    return service.get_focus_sessions(current_user.id, skip, limit)

@router.get("/{session_id}", response_model=FocusSession)
def get_focus_session(
    session_id: UUID,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get a specific focus session by ID."""
    service = FocusSessionService(db)
    return service.get_focus_session(current_user.id, session_id)

@router.put("/{session_id}", response_model=FocusSession)
def update_focus_session(
    session_id: UUID,
    session_data: FocusSessionUpdate,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update a focus session."""
    service = FocusSessionService(db)
    return service.update_focus_session(current_user.id, session_id, session_data)

@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_focus_session(
    session_id: UUID,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete a focus session."""
    service = FocusSessionService(db)
    service.delete_focus_session(current_user.id, session_id)
    return None

@router.post("/{session_id}/start", response_model=FocusSession)
def start_focus_session(
    session_id: UUID,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Start a focus session."""
    service = FocusSessionService(db)
    return service.start_focus_session(current_user.id, session_id)

@router.post("/{session_id}/complete", response_model=FocusSession)
def complete_focus_session(
    session_id: UUID,
    notes: Optional[str] = None,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Complete a focus session."""
    service = FocusSessionService(db)
    return service.complete_focus_session(current_user.id, session_id, notes)

@router.post("/{session_id}/interruption", response_model=FocusSession)
def record_interruption(
    session_id: UUID,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Record an interruption during a focus session."""
    service = FocusSessionService(db)
    return service.record_interruption(current_user.id, session_id)

@router.get("/statistics", response_model=dict)
def get_focus_statistics(
    days: int = Query(30, ge=1, le=365),
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get focus session statistics for the current user."""
    service = FocusSessionService(db)
    return service.get_focus_statistics(current_user.id, days)
```

### 6. Register the Router in the Main Application

Add your new router to the main FastAPI application:

```python
# Example: backend/app/main.py
from fastapi import FastAPI
from app.routers import tasks, calendar, analytics, integrations, focus_sessions

app = FastAPI(
    title="Tempo API",
    description="Productivity and Time Management Agent API",
    version="0.1.0"
)

# Include routers
app.include_router(tasks.router, prefix="/api/tasks", tags=["tasks"])
app.include_router(calendar.router, prefix="/api/calendar", tags=["calendar"])
app.include_router(analytics.router, prefix="/api/analytics", tags=["analytics"])
app.include_router(integrations.router, prefix="/api/integrations", tags=["integrations"])
app.include_router(focus_sessions.router, prefix="/api/focus-sessions", tags=["focus-sessions"])
```

### 7. Implement API Tests

Create tests for your new endpoints to ensure they work as expected:

```python
# Example: backend/tests/api/test_focus_sessions.py
import pytest
from fastapi.testclient import TestClient
from uuid import uuid4
from datetime import datetime, timedelta

from app.main import app
from app.models.focus_session import FocusSessionStatus

client = TestClient(app)

def test_create_focus_session(authenticated_client, test_user):
    """Test creating a focus session."""
    response = authenticated_client.post(
        "/api/focus-sessions/",
        json={
            "title": "Deep work on project X",
            "planned_duration_minutes": 45,
            "description": "Focus on implementing feature Y"
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["title"] == "Deep work on project X"
    assert data["planned_duration_minutes"] == 45
    assert data["status"] == "planned"
    assert data["user_id"] == str(test_user.id)
    
def test_get_focus_sessions(authenticated_client, create_test_focus_session):
    """Test getting all focus sessions."""
    # Create a few test sessions
    session1 = create_test_focus_session("Session 1")
    session2 = create_test_focus_session("Session 2")
    
    response = authenticated_client.get("/api/focus-sessions/")
    assert response.status_code == 200
    data = response.json()
    assert len(data) >= 2
    assert any(s["id"] == str(session1.id) for s in data)
    assert any(s["id"] == str(session2.id) for s in data)
    
def test_start_focus_session(authenticated_client, create_test_focus_session):
    """Test starting a focus session."""
    session = create_test_focus_session("Test session")
    
    response = authenticated_client.post(f"/api/focus-sessions/{session.id}/start")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "in_progress"
    assert data["actual_start"] is not None
    
def test_complete_focus_session(authenticated_client, create_test_focus_session):
    """Test completing a focus session."""
    session = create_test_focus_session("Test session")
    
    # First start the session
    authenticated_client.post(f"/api/focus-sessions/{session.id}/start")
    
    # Then complete it
    response = authenticated_client.post(
        f"/api/focus-sessions/{session.id}/complete",
        params={"notes": "Completed successfully"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "completed"
    assert data["actual_end"] is not None
    assert data["actual_duration_minutes"] is not None
    assert data["productivity_score"] is not None
    assert data["notes"] == "Completed successfully"
```

### 8. Enhance Existing Endpoints

Often, you'll need to enhance existing endpoints to support new functionality:

```python
# Example: Enhancing task endpoints to support focus sessions
# backend/app/routers/tasks.py

@router.get("/{task_id}/focus-sessions", response_model=List[FocusSession])
def get_task_focus_sessions(
    task_id: UUID,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get all focus sessions associated with a specific task."""
    task_service = TaskService(db)
    task = task_service.get_task(current_user.id, task_id)
    
    focus_service = FocusSessionService(db)
    sessions = db.query(DBFocusSession)\
        .filter(DBFocusSession.task_id == task_id, DBFocusSession.user_id == current_user.id)\
        .order_by(DBFocusSession.created_at.desc())\
        .all()
        
    return sessions

@router.post("/{task_id}/focus-sessions", response_model=FocusSession)
def create_task_focus_session(
    task_id: UUID,
    session_data: FocusSessionCreate,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a focus session for a specific task."""
    task_service = TaskService(db)
    task = task_service.get_task(current_user.id, task_id)
    
    # Override task_id in the session data
    session_data_dict = session_data.dict()
    session_data_dict["task_id"] = task_id
    
    # If no title provided, use task title
    if not session_data_dict.get("title"):
        session_data_dict["title"] = f"Focus on: {task.title}"
        
    focus_service = FocusSessionService(db)
    return focus_service.create_focus_session(
        current_user.id, 
        FocusSessionCreate(**session_data_dict)
    )
```

### 9. Implement Advanced Query Parameters

For more complex endpoints, implement advanced query parameters:

```python
# Example: Advanced filtering for focus sessions
@router.get("/", response_model=List[FocusSession])
def get_focus_sessions(
    skip: int = 0,
    limit: int = 100,
    status: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    task_id: Optional[UUID] = None,
    min_duration: Optional[int] = None,
    max_duration: Optional[int] = None,
    sort_by: str = "created_at",
    sort_order: str = "desc",
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get focus sessions with advanced filtering options.
    
    - Filter by status, date range, task, or duration
    - Sort by various fields
    - Paginate results
    """
    service = FocusSessionService(db)
    
    # Build query
    query = db.query(DBFocusSession).filter(DBFocusSession.user_id == current_user.id)
    
    # Apply filters
    if status:
        query = query.filter(DBFocusSession.status == status)
    if start_date:
        query = query.filter(DBFocusSession.created_at >= start_date)
    if end_date:
        query = query.filter(DBFocusSession.created_at <= end_date)
    if task_id:
        query = query.filter(DBFocusSession.task_id == task_id)
    if min_duration:
        query = query.filter(DBFocusSession.planned_duration_minutes >= min_duration)
    if max_duration:
        query = query.filter(DBFocusSession.planned_duration_minutes <= max_duration)
        
    # Apply sorting
    sort_column = getattr(DBFocusSession, sort_by, DBFocusSession.created_at)
    if sort_order.lower() == "asc":
        query = query.order_by(sort_column.asc())
    else:
        query = query.order_by(sort_column.desc())
        
    # Apply pagination
    query = query.offset(skip).limit(limit)
    
    return query.all()
```

### 10. Implement Batch Operations

For efficiency, implement batch operations for common scenarios:

```python
# Example: Batch operations for focus sessions
@router.post("/batch", response_model=List[FocusSession])
def create_batch_focus_sessions(
    sessions_data: List[FocusSessionCreate],
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create multiple focus sessions in a single request."""
    service = FocusSessionService(db)
    results = []
    
    for session_data in sessions_data:
        session = service.create_focus_session(current_user.id, session_data)
        results.append(session)
        
    return results

@router.delete("/batch", status_code=status.HTTP_204_NO_CONTENT)
def delete_batch_focus_sessions(
    session_ids: List[UUID],
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete multiple focus sessions in a single request."""
    service = FocusSessionService(db)
    
    for session_id in session_ids:
        try:
            service.delete_focus_session(current_user.id, session_id)
        except HTTPException as e:
            if e.status_code != status.HTTP_404_NOT_FOUND:
                raise
                
    return None
```

### 11. Implement WebSocket Endpoints for Real-Time Features

For real-time features like focus session timers, implement WebSocket endpoints:

```python
# Example: WebSocket for focus session timer
from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict
import json
import asyncio

# Connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[UUID, Dict[str, WebSocket]] = {}
        
    async def connect(self, websocket: WebSocket, user_id: UUID, session_id: UUID):
        await websocket.accept()
        if user_id not in self.active_connections:
            self.active_connections[user_id] = {}
        self.active_connections[user_id][str(session_id)] = websocket
        
    def disconnect(self, user_id: UUID, session_id: UUID):
        if user_id in self.active_connections:
            self.active_connections[user_id].pop(str(session_id), None)
            if not self.active_connections[user_id]:
                self.active_connections.pop(user_id, None)
                
    async def send_update(self, user_id: UUID, session_id: UUID, data: dict):
        if user_id in self.active_connections and str(session_id) in self.active_connections[user_id]:
            websocket = self.active_connections[user_id][str(session_id)]
            await websocket.send_json(data)
            
    async def broadcast_to_user(self, user_id: UUID, data: dict):
        if user_id in self.active_connections:
            for websocket in self.active_connections[user_id].values():
                await websocket.send_json(data)

manager = ConnectionManager()

# Add to focus_sessions.py router
@router.websocket("/{session_id}/ws")
async def focus_session_websocket(
    websocket: WebSocket,
    session_id: UUID,
    token: str,
    db: Session = Depends(get_db)
):
    """WebSocket endpoint for real-time focus session updates."""
    # Authenticate user from token
    try:
        user = await get_user_from_token(token, db)
    except:
        await websocket.close(code=1008)  # Policy violation
        return
        
    # Verify session exists and belongs to user
    service = FocusSessionService(db)
    try:
        session = service.get_focus_session(user.id, session_id)
    except HTTPException:
        await websocket.close(code=1008)
        return
        
    # Accept connection
    await manager.connect(websocket, user.id, session_id)
    
    try:
        # Send initial session state
        await websocket.send_json({
            "type": "session_state",
            "session": {
                "id": str(session.id),
                "status": session.status,
                "elapsed_seconds": (datetime.now() - session.actual_start).total_seconds() if session.actual_start else 0,
                "interruption_count": session.interruption_count
            }
        })
        
        # Listen for messages
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message["type"] == "interruption":
                # Record interruption
                session = service.record_interruption(user.id, session_id)
                
                # Broadcast update
                await manager.send_update(user.id, session_id, {
                    "type": "interruption",
                    "interruption_count": session.interruption_count
                })
                
            elif message["type"] == "complete":
                # Complete session
                notes = message.get("notes")
                session = service.complete_focus_session(user.id, session_id, notes)
                
                # Send completion confirmation
                await manager.send_update(user.id, session_id, {
                    "type": "completed",
                    "session": {
                        "id": str(session.id),
                        "status": session.status,
                        "actual_duration_minutes": session.actual_duration_minutes,
                        "productivity_score": session.productivity_score
                    }
                })
                
    except WebSocketDisconnect:
        manager.disconnect(user.id, session_id)
```

### 12. Implement API Versioning

As your API evolves, implement versioning to maintain backward compatibility:

```python
# Example: Version prefix in main.py
app = FastAPI(
    title="Tempo API",
    description="Productivity and Time Management Agent API",
    version="0.2.0"
)

# Version 1 (legacy)
v1_app = FastAPI(
    title="Tempo API v1",
    description="Legacy API endpoints",
    version="0.1.0"
)
v1_app.include_router(legacy_tasks.router, prefix="/tasks", tags=["tasks"])
# ... other v1 routers

# Version 2 (current)
v2_app = FastAPI(
    title="Tempo API v2",
    description="Current API endpoints",
    version="0.2.0"
)
v2_app.include_router(tasks.router, prefix="/tasks", tags=["tasks"])
v2_app.include_router(focus_sessions.router, prefix="/focus-sessions", tags=["focus-sessions"])
# ... other v2 routers

# Mount versioned APIs
app.mount("/api/v1", v1_app)
app.mount("/api/v2", v2_app)

# Default to latest version
app.include_router(tasks.router, prefix="/api/tasks", tags=["tasks"])
app.include_router(focus_sessions.router, prefix="/api/focus-sessions", tags=["focus-sessions"])
# ... other current routers
```

## Best Practices for API Extension

### 1. Maintain Consistent Response Formats

Ensure all endpoints follow a consistent response format:

```python
# Example: Standardized response format
from fastapi import status
from fastapi.responses import JSONResponse

def create_response(data=None, message=None, success=True, status_code=status.HTTP_200_OK):
    """Create a standardized API response."""
    response = {
        "success": success,
        "message": message,
        "data": data
    }
    return JSONResponse(content=response, status_code=status_code)

# Usage in endpoint
@router.get("/statistics", response_model=None)
def get_focus_statistics(
    days: int = Query(30, ge=1, le=365),
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get focus session statistics for the current user."""
    service = FocusSessionService(db)
    stats = service.get_focus_statistics(current_user.id, days)
    return create_response(
        data=stats,
        message="Focus statistics retrieved successfully"
    )
```

### 2. Implement Comprehensive Error Handling

Create a robust error handling system:

```python
# Example: Custom exception handler
from fastapi import Request, status
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError
from pydantic import ValidationError

class TempoException(Exception):
    """Base exception for Tempo API."""
    def __init__(self, message: str, status_code: int = status.HTTP_400_BAD_REQUEST):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)

# Register exception handlers
@app.exception_handler(TempoException)
async def tempo_exception_handler(request: Request, exc: TempoException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"success": False, "message": exc.message, "data": None}
    )

@app.exception_handler(SQLAlchemyError)
async def sqlalchemy_exception_handler(request: Request, exc: SQLAlchemyError):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"success": False, "message": "Database error occurred", "data": None}
    )

@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "success": False, 
            "message": "Validation error", 
            "data": {"errors": exc.errors()}
        }
    )
```

### 3. Implement Rate Limiting

Protect your API with rate limiting:

```python
# Example: Rate limiting middleware
from fastapi import Request, Response
import time
from typing import Dict, Tuple

class RateLimiter:
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.reset_interval = 60  # seconds
        self.user_requests: Dict[str, Tuple[int, float]] = {}
        
    async def __call__(self, request: Request, response: Response):
        # Get user identifier (IP or user ID from token)
        user_id = request.client.host
        if "Authorization" in request.headers:
            try:
                token = request.headers["Authorization"].split(" ")[1]
                user_id = get_user_id_from_token(token)
            except:
                pass
                
        # Check rate limit
        current_time = time.time()
        if user_id in self.user_requests:
            count, start_time = self.user_requests[user_id]
            
            # Reset if interval passed
            if current_time - start_time > self.reset_interval:
                self.user_requests[user_id] = (1, current_time)
            else:
                # Increment count
                count += 1
                if count > self.requests_per_minute:
                    response.headers["Retry-After"] = str(int(self.reset_interval - (current_time - start_time)))
                    response.status_code = status.HTTP_429_TOO_MANY_REQUESTS
                    return {"success": False, "message": "Rate limit exceeded", "data": None}
                    
                self.user_requests[user_id] = (count, start_time)
        else:
            self.user_requests[user_id] = (1, current_time)
            
        # Add rate limit headers
        count, start_time = self.user_requests[user_id]
        response.headers["X-Rate-Limit-Limit"] = str(self.requests_per_minute)
        response.headers["X-Rate-Limit-Remaining"] = str(max(0, self.requests_per_minute - count))
        response.headers["X-Rate-Limit-Reset"] = str(int(start_time + self.reset_interval))

# Add to main.py
rate_limiter = RateLimiter(requests_per_minute=100)
app.middleware("http")(rate_limiter)
```

### 4. Implement Comprehensive API Documentation

Enhance the automatic documentation with detailed descriptions:

```python
# Example: Enhanced API documentation
from fastapi.openapi.utils import get_openapi

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
        
    openapi_schema = get_openapi(
        title="Tempo API",
        version="0.2.0",
        description="Productivity and Time Management Agent API",
        routes=app.routes,
    )
    
    # Add security scheme
    openapi_schema["components"]["securitySchemes"] = {
        "bearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
        }
    }
    
    # Apply security globally
    openapi_schema["security"] = [{"bearerAuth": []}]
    
    # Add custom documentation
    openapi_schema["tags"] = [
        {
            "name": "focus-sessions",
            "description": "Operations related to focus sessions (pomodoro-style time blocks)",
            "externalDocs": {
                "description": "Focus Technique Documentation",
                "url": "https://en.wikipedia.org/wiki/Pomodoro_Technique",
            },
        },
        # ... other tags
    ]
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi
```

### 5. Implement API Monitoring and Logging

Add comprehensive logging to track API usage and performance:

```python
# Example: API logging middleware
import logging
import time
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("api")

class APILoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Extract request details
        method = request.method
        url = str(request.url)
        client_ip = request.client.host
        user_agent = request.headers.get("User-Agent", "")
        
        # Process request
        try:
            response = await call_next(request)
            status_code = response.status_code
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Log successful request
            logger.info(
                f"Request: {method} {url} - Status: {status_code} - "
                f"Duration: {duration:.3f}s - IP: {client_ip} - UA: {user_agent}"
            )
            
            return response
            
        except Exception as e:
            # Log failed request
            duration = time.time() - start_time
            logger.error(
                f"Request: {method} {url} - Error: {str(e)} - "
                f"Duration: {duration:.3f}s - IP: {client_ip} - UA: {user_agent}"
            )
            raise

# Add to main.py
app.add_middleware(APILoggingMiddleware)
```

## Conclusion

Extending the API endpoints for your Tempo implementation allows you to add powerful new productivity features while maintaining a clean, well-structured codebase. By following the layered architecture pattern and best practices outlined in this guide, you can ensure that your API remains maintainable, performant, and secure as it grows.

Remember that a well-designed API is the foundation for both backend functionality and frontend user experience. Take the time to plan your endpoints carefully, implement comprehensive validation and error handling, and document your API thoroughly to make integration with the frontend as smooth as possible.

As you extend the API, regularly review your overall architecture to ensure it remains coherent and aligned with your productivity management goals. Consider gathering usage metrics to identify which endpoints are most valuable to users and prioritize further enhancements accordingly.
# Customizing Frontend Components for Tempo UI Needs

## Introduction

The frontend is the primary interface through which users interact with the Tempo productivity and time management agent. While the implementation tool provides a basic frontend structure using React, customizing these components is crucial to create a user experience that aligns with your specific productivity workflows and visual identity. This guide provides a comprehensive approach to adapting existing components, creating new ones, integrating with the API, and applying UI/UX best practices for a productivity-focused application.

## Understanding the Frontend Structure

Before customizing, familiarize yourself with the existing frontend architecture located in `frontend/src/`:

- **`components/`**: Contains reusable UI elements (e.g., `Button`, `TaskItem`, `CalendarDay`). These are the building blocks of your interface.
- **`pages/`**: Represents distinct views or screens of the application (e.g., `DashboardPage`, `TasksPage`, `SettingsPage`). Pages typically compose multiple components.
- **`services/`**: Includes API client functions for interacting with the backend API. These functions handle fetching and sending data.
- **`contexts/` or `store/`**: Manages global application state (e.g., user authentication, shared data). The specific implementation might use React Context, Redux, Zustand, or another state management library.
- **`hooks/`**: Contains custom React hooks for encapsulating reusable logic (e.g., `useFetch`, `useTimer`).
- **`utils/`**: Holds utility functions (e.g., date formatting, data transformation).
- **`App.js` or `main.jsx`**: The main application component, responsible for routing and global layout.
- **`index.js` or `main.jsx`**: The entry point of the React application.

## Step-by-Step Guide to Frontend Customization

### 1. Define UI Requirements for New Features

Translate your new backend features into concrete UI requirements. For the "Focus Sessions" example from the API guide, you might need:

- A way to create new focus sessions (a form or button).
- A list displaying planned, ongoing, and completed focus sessions.
- A timer component for active focus sessions.
- Visualizations for focus session statistics.
- Integration into the main dashboard or a dedicated focus page.

Sketch wireframes or mockups to visualize how these elements fit into the existing UI.

### 2. Create New React Components

Develop new components for the required UI elements in the `components/` directory:

```jsx
// Example: frontend/src/components/FocusSession/FocusSessionTimer.jsx
import React, { useState, useEffect, useCallback } from 'react';
import PropTypes from 'prop-types';
import { useFocusSession } from '../../hooks/useFocusSession'; // Custom hook
import { Button, Progress, Typography, Space, notification } from 'antd'; // Example UI library
import { PlayCircleOutlined, PauseCircleOutlined, CheckCircleOutlined, CloseCircleOutlined } from '@ant-design/icons';

const FocusSessionTimer = ({ sessionId }) => {
    const { 
        session, 
        startSession, 
        completeSession, 
        recordInterruption, 
        isLoading, 
        error 
    } = useFocusSession(sessionId);
    
    const [timeLeft, setTimeLeft] = useState(0);
    const [isRunning, setIsRunning] = useState(false);

    // Initialize timer based on session status and duration
    useEffect(() => {
        if (session) {
            if (session.status === 'in_progress' && session.actual_start) {
                const elapsedSeconds = (new Date() - new Date(session.actual_start)) / 1000;
                const remaining = Math.max(0, (session.planned_duration_minutes * 60) - elapsedSeconds);
                setTimeLeft(remaining);
                setIsRunning(true);
            } else if (session.status === 'planned') {
                setTimeLeft(session.planned_duration_minutes * 60);
                setIsRunning(false);
            } else {
                setTimeLeft(0);
                setIsRunning(false);
            }
        }
    }, [session]);

    // Timer countdown logic
    useEffect(() => {
        let interval = null;
        if (isRunning && timeLeft > 0) {
            interval = setInterval(() => {
                setTimeLeft(prevTime => Math.max(0, prevTime - 1));
            }, 1000);
        } else if (isRunning && timeLeft === 0) {
            // Auto-complete when timer finishes
            handleComplete();
            notification.success({ message: 'Focus session completed!' });
        }
        return () => clearInterval(interval);
    }, [isRunning, timeLeft]);

    const handleStart = async () => {
        const success = await startSession();
        if (success) {
            setIsRunning(true);
            setTimeLeft(session.planned_duration_minutes * 60);
        }
    };

    const handleComplete = async () => {
        const success = await completeSession('Completed via timer');
        if (success) {
            setIsRunning(false);
            setTimeLeft(0);
        }
    };

    const handleInterrupt = async () => {
        await recordInterruption();
        // Optionally pause the timer or just log the interruption
        notification.info({ message: 'Interruption recorded' });
    };

    if (isLoading) return <Typography.Text>Loading timer...</Typography.Text>;
    if (error) return <Typography.Text type="danger">Error: {error}</Typography.Text>;
    if (!session) return <Typography.Text>Session not found.</Typography.Text>;

    const formatTime = (seconds) => {
        const minutes = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${minutes}:${secs < 10 ? '0' : ''}${secs}`;
    };

    const percentComplete = session.planned_duration_minutes > 0 
        ? ((session.planned_duration_minutes * 60 - timeLeft) / (session.planned_duration_minutes * 60)) * 100
        : 0;

    return (
        <div style={{ padding: '20px', border: '1px solid #eee', borderRadius: '8px' }}>
            <Typography.Title level={4}>{session.title}</Typography.Title>
            <Typography.Text type="secondary">Status: {session.status}</Typography.Text>
            <Progress percent={percentComplete} showInfo={false} status={isRunning ? 'active' : 'normal'} />
            <Typography.Title level={2} style={{ textAlign: 'center', margin: '20px 0' }}>
                {formatTime(timeLeft)}
            </Typography.Title>
            <Space direction="vertical" style={{ width: '100%' }}>
                <Space style={{ justifyContent: 'center', width: '100%' }}>
                    {session.status === 'planned' && (
                        <Button type="primary" icon={<PlayCircleOutlined />} onClick={handleStart} size="large">
                            Start Session
                        </Button>
                    )}
                    {session.status === 'in_progress' && (
                        <>
                            <Button danger icon={<CloseCircleOutlined />} onClick={handleInterrupt} title="Record Interruption">
                                Interrupt
                            </Button>
                            <Button type="primary" icon={<CheckCircleOutlined />} onClick={handleComplete} size="large">
                                Complete Now
                            </Button>
                        </>
                    )}
                    {session.status === 'completed' && (
                        <Typography.Text strong>Session Completed!</Typography.Text>
                    )}
                </Space>
                {session.status === 'in_progress' && (
                    <Typography.Text type="secondary" style={{ textAlign: 'center', display: 'block' }}>
                        Interruptions: {session.interruption_count}
                    </Typography.Text>
                )}
            </Space>
        </div>
    );
};

FocusSessionTimer.propTypes = {
    sessionId: PropTypes.string.isRequired,
};

export default FocusSessionTimer;
```

### 3. Modify Existing Components

Adapt existing components to integrate the new functionality. For instance, add a button to the `TaskItem` component to start a focus session for that task:

```jsx
// Example: Modifying frontend/src/components/Task/TaskItem.jsx
import React from 'react';
import { Card, Checkbox, Button, Tag, Space } from 'antd';
import { ClockCircleOutlined } from '@ant-design/icons';
import { useHistory } from 'react-router-dom'; // Assuming React Router
import { useFocusSessionActions } from '../../hooks/useFocusSessionActions'; // Hook for actions

const TaskItem = ({ task, onToggleComplete }) => {
    const history = useHistory();
    const { createFocusSessionForTask } = useFocusSessionActions();

    const handleStartFocus = async () => {
        const newSession = await createFocusSessionForTask(task.id, task.title);
        if (newSession) {
            // Navigate to a page where the timer is displayed, or show a modal
            history.push(`/focus/${newSession.id}`);
        }
    };

    return (
        <Card 
            size="small" 
            style={{ marginBottom: '8px', opacity: task.status === 'completed' ? 0.6 : 1 }}
            hoverable
        >
            <Space align="start" style={{ width: '100%' }}>
                <Checkbox 
                    checked={task.status === 'completed'} 
                    onChange={() => onToggleComplete(task.id, task.status !== 'completed')}
                />
                <div style={{ flexGrow: 1 }}>
                    <Typography.Text delete={task.status === 'completed'}>{task.title}</Typography.Text>
                    {task.description && <Typography.Paragraph type="secondary" ellipsis={{ rows: 1 }}>{task.description}</Typography.Paragraph>}
                    <Space size="small" wrap>
                        {task.tags?.map(tag => <Tag key={tag}>{tag}</Tag>)}
                        {task.due_date && <Tag color="blue">Due: {new Date(task.due_date).toLocaleDateString()}</Tag>}
                        <Tag color="purple">Priority: {task.priority}</Tag>
                    </Space>
                </div>
                <Button 
                    icon={<ClockCircleOutlined />} 
                    onClick={handleStartFocus} 
                    size="small"
                    title="Start Focus Session"
                    disabled={task.status === 'completed'}
                >
                    Focus
                </Button>
            </Space>
        </Card>
    );
};

export default TaskItem;
```

### 4. Update API Service Functions

Create or modify functions in `frontend/src/services/` to call the new API endpoints:

```javascript
// Example: frontend/src/services/focusSessionApi.js
import apiClient from './apiClient'; // Your configured Axios or Fetch instance

export const createFocusSession = async (sessionData) => {
    try {
        const response = await apiClient.post('/focus-sessions/', sessionData);
        return response.data; // Assuming API returns the created session
    } catch (error) {
        console.error("Error creating focus session:", error.response?.data || error.message);
        throw error;
    }
};

export const getFocusSessionById = async (sessionId) => {
    try {
        const response = await apiClient.get(`/focus-sessions/${sessionId}`);
        return response.data;
    } catch (error) {
        console.error(`Error fetching focus session ${sessionId}:`, error.response?.data || error.message);
        throw error;
    }
};

export const startFocusSessionApi = async (sessionId) => {
    try {
        const response = await apiClient.post(`/focus-sessions/${sessionId}/start`);
        return response.data;
    } catch (error) {
        console.error(`Error starting focus session ${sessionId}:`, error.response?.data || error.message);
        throw error;
    }
};

// ... other API functions for complete, interrupt, get list, etc.
```

### 5. Manage Application State

Integrate the new data and UI state into your state management solution. This might involve:

- Adding new state slices or context providers (e.g., `FocusSessionContext`).
- Creating actions/reducers or Zustand store slices for focus session data.
- Updating existing state when related actions occur (e.g., updating a task's associated focus sessions).

```javascript
// Example: Using Zustand for state management
import create from 'zustand';
import { 
    getFocusSessionById, 
    startFocusSessionApi, 
    // ... other API calls
} from '../services/focusSessionApi';

const useFocusSessionStore = create((set, get) => ({
    sessions: {},
    loading: {},
    error: {},

    fetchSession: async (sessionId) => {
        if (get().sessions[sessionId] && !get().error[sessionId]) return; // Already fetched
        set(state => ({ loading: { ...state.loading, [sessionId]: true }, error: { ...state.error, [sessionId]: null } }));
        try {
            const session = await getFocusSessionById(sessionId);
            set(state => ({
                sessions: { ...state.sessions, [sessionId]: session },
                loading: { ...state.loading, [sessionId]: false },
            }));
            return session;
        } catch (err) {
            set(state => ({ 
                loading: { ...state.loading, [sessionId]: false }, 
                error: { ...state.error, [sessionId]: err.message }
            }));
            return null;
        }
    },

    startSession: async (sessionId) => {
        set(state => ({ loading: { ...state.loading, [sessionId]: true } }));
        try {
            const updatedSession = await startFocusSessionApi(sessionId);
            set(state => ({
                sessions: { ...state.sessions, [sessionId]: updatedSession },
                loading: { ...state.loading, [sessionId]: false },
            }));
            return true;
        } catch (err) {
            set(state => ({ 
                loading: { ...state.loading, [sessionId]: false }, 
                error: { ...state.error, [sessionId]: err.message }
            }));
            return false;
        }
    },
    // ... other actions (complete, interrupt, etc.)
}));

export default useFocusSessionStore;
```

### 6. Implement Routing

If your new feature requires dedicated pages, add routes using your routing library (e.g., React Router):

```jsx
// Example: frontend/src/App.js
import React from 'react';
import { BrowserRouter as Router, Route, Switch, Redirect } from 'react-router-dom';
import DashboardPage from './pages/DashboardPage';
import TasksPage from './pages/TasksPage';
import FocusSessionPage from './pages/FocusSessionPage'; // New page
import LoginPage from './pages/LoginPage';
import PrivateRoute from './components/Auth/PrivateRoute';

function App() {
    return (
        <Router>
            <Switch>
                <Route path="/login" component={LoginPage} />
                <PrivateRoute exact path="/" component={DashboardPage} />
                <PrivateRoute path="/tasks" component={TasksPage} />
                <PrivateRoute path="/focus/:sessionId" component={FocusSessionPage} /> {/* New Route */}
                {/* Add other routes here */}
                <Redirect to="/" />
            </Switch>
        </Router>
    );
}

export default App;
```

### 7. Apply Consistent Styling

Ensure new components match the existing visual style. Use the established styling approach (e.g., CSS Modules, Tailwind CSS, a UI library like Ant Design or Material UI) consistently.

```css
/* Example: frontend/src/components/FocusSession/FocusSessionTimer.module.css */
.timerContainer {
    padding: 20px;
    border: 1px solid var(--border-color-light);
    border-radius: 8px;
    background-color: var(--background-color-secondary);
    box-shadow: var(--shadow-sm);
}

.timerDisplay {
    font-size: 3rem;
    font-weight: bold;
    text-align: center;
    margin: 20px 0;
    color: var(--text-color-primary);
}

.controlsContainer {
    display: flex;
    justify-content: center;
    gap: 15px;
}
```

## Productivity UI/UX Best Practices

Tailor your UI to enhance productivity:

- **Minimize Distractions**: Keep the interface clean, especially during focus sessions. Use subtle notifications.
- **Clear Information Hierarchy**: Prioritize essential information (e.g., current task, time remaining).
- **Real-time Feedback**: Use WebSockets or polling for live updates on timers, task progress, etc. Provide immediate visual feedback for actions.
- **Contextual Actions**: Offer relevant actions based on the current state (e.g., show "Complete Session" only when a session is active).
- **Keyboard Shortcuts**: Implement shortcuts for common actions (e.g., starting/stopping timers, completing tasks).
- **Personalization**: Allow users to customize themes, timer sounds, notification preferences, and dashboard layouts.
- **Data Visualization**: Use clear charts (bar, line, pie) to represent productivity trends, time allocation, and task completion rates. Libraries like Recharts, Chart.js, or Nivo can be helpful.
- **Accessibility (a11y)**: Ensure components are navigable via keyboard, compatible with screen readers (using ARIA attributes), and have sufficient color contrast.
- **Responsive Design**: Test and adapt layouts for various screen sizes (desktop, tablet, mobile) using CSS media queries or responsive utility classes.
- **Progress Indicators**: Clearly show progress for ongoing tasks or focus sessions.

## Integrating with New API Endpoints

Ensure your components correctly call the API service functions and handle loading/error states:

```jsx
// Example: frontend/src/pages/FocusSessionPage.jsx
import React, { useEffect } from 'react';
import { useParams } from 'react-router-dom';
import useFocusSessionStore from '../store/useFocusSessionStore'; // Zustand store hook
import FocusSessionTimer from '../components/FocusSession/FocusSessionTimer';
import { Spin, Alert } from 'antd';

const FocusSessionPage = () => {
    const { sessionId } = useParams();
    const fetchSession = useFocusSessionStore(state => state.fetchSession);
    const session = useFocusSessionStore(state => state.sessions[sessionId]);
    const isLoading = useFocusSessionStore(state => state.loading[sessionId]);
    const error = useFocusSessionStore(state => state.error[sessionId]);

    useEffect(() => {
        if (sessionId) {
            fetchSession(sessionId);
        }
    }, [sessionId, fetchSession]);

    if (isLoading === undefined && !session) {
        // Initial load state before fetch starts
        return <Spin tip="Loading session..." />;
    }
    
    if (isLoading) {
        return <Spin tip="Loading session..." />;
    }

    if (error) {
        return <Alert message="Error" description={`Failed to load session: ${error}`} type="error" showIcon />;
    }

    if (!session) {
        return <Alert message="Error" description="Focus session not found." type="error" showIcon />;
    }

    return (
        <div>
            <h1>Focus Session</h1>
            <FocusSessionTimer sessionId={sessionId} />
            {/* You might add related task details or notes section here */}
        </div>
    );
};

export default FocusSessionPage;
```

## Testing Frontend Components

Implement tests to ensure reliability:

- **Unit Tests**: Use Jest and React Testing Library to test individual components in isolation. Mock API calls and state management hooks.
- **Integration Tests**: Test the interaction between multiple components (e.g., ensuring clicking the "Focus" button on a `TaskItem` correctly navigates and displays the `FocusSessionTimer`).
- **End-to-End (E2E) Tests**: Use tools like Cypress or Playwright to simulate user flows through the entire application, interacting with a live (or mocked) backend.

## Conclusion

Customizing the frontend is essential for creating an effective and engaging Tempo agent. By carefully designing new components, integrating them with the API, managing state effectively, and adhering to productivity-focused UI/UX principles, you can build an interface that truly helps users manage their time and tasks efficiently.

Remember to iterate based on user feedback. A productivity tool is highly personal, so continuously refining the UI based on how people actually use it is key to its success.
# Adding and Integrating External Services in Tempo

## Introduction

Integrating Tempo with external services like calendars (Google Calendar, Outlook Calendar), project management tools (Jira, Asana, Trello), and communication platforms (Slack, Microsoft Teams) significantly enhances its capabilities. It allows Tempo to consolidate information, automate workflows, and provide a more holistic view of a user's tasks and schedule. This guide provides a comprehensive approach to designing, implementing, and managing external integrations within the Tempo structure.

## Identifying Integration Needs

Tempo can benefit from various integrations:

- **Calendar Integration**: Syncing events to provide a complete schedule view, block time for meetings, and avoid scheduling conflicts.
- **Project Management Tools**: Importing tasks, syncing statuses, and linking Tempo time entries back to specific project tasks.
- **Communication Platforms**: Creating tasks from messages, sending notifications, and providing status updates.
- **Note-Taking Apps**: Linking tasks or focus sessions to specific notes (e.g., Evernote, Notion).
- **Version Control Systems**: Linking tasks to code commits or pull requests (e.g., GitHub, GitLab).
- **Health & Wellness Apps**: Correlating productivity patterns with health data (e.g., sleep, activity levels) - requires careful privacy considerations.

Prioritize integrations based on your target users' most common tools and workflows.

## Integration Architecture

Integrate external services primarily within the backend to centralize logic, manage credentials securely, and handle synchronization reliably.

### Backend Structure

Organize integration code within the `backend/app/integrations/` directory:

```
backend/app/integrations/
├── __init__.py
├── base.py                # Base class for integrations
├── providers/             # Specific provider implementations
│   ├── __init__.py
│   ├── google_calendar.py
│   ├── jira.py
│   └── slack.py
├── services/              # Service logic for managing integrations
│   ├── __init__.py
│   ├── integration_service.py
│   └── sync_service.py
├── utils/                 # Shared utilities for integrations
│   ├── __init__.py
│   └── oauth_handler.py
└── models/                # Pydantic models specific to integrations
    ├── __init__.py
    └── google_models.py
```

### Key Components

- **Provider Modules**: Each external service gets its own module (e.g., `google_calendar.py`) containing the API client, data mapping logic, and specific authentication handling.
- **Integration Service**: Manages the lifecycle of integrations (adding, configuring, removing, checking status).
- **Sync Service**: Handles the logic for synchronizing data between Tempo and external services (can use background tasks, message queues, or scheduled jobs).
- **OAuth Handler**: Centralizes OAuth 2.0 flow management.
- **Database Models**: The `tempo.integrations` table (defined in `init-postgres.sql`) stores credentials and settings for each user's active integrations.

## Step-by-Step Guide to Adding an Integration

Let's use Google Calendar as an example:

### 1. Register Application with Provider

- Go to the Google Cloud Console.
- Create a new project or use an existing one.
- Enable the Google Calendar API.
- Create OAuth 2.0 credentials (Web application type).
- Configure the OAuth consent screen.
- Note down the Client ID and Client Secret.
- Set authorized redirect URIs (e.g., `http://localhost:8000/api/integrations/google/callback`).

### 2. Implement OAuth 2.0 Flow

Handle the OAuth authorization code flow to obtain user consent and tokens.

```python
# Example: backend/app/integrations/utils/oauth_handler.py
from fastapi import Request, HTTPException, status
from authlib.integrations.starlette_client import OAuth
from starlette.config import Config
import os

# Load configuration (Client ID/Secret)
# Best practice: Use environment variables or a config file
config = Config(".env") 

oauth = OAuth(config)

# Register Google OAuth client
oauth.register(
    name='google',
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_id=os.getenv("GOOGLE_CLIENT_ID"),
    client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
    client_kwargs={
        'scope': 'openid email profile https://www.googleapis.com/auth/calendar.events.readonly https://www.googleapis.com/auth/calendar.readonly',
        'redirect_uri': 'http://localhost:8000/api/integrations/google/callback' # Match registered URI
    }
)

async def google_login(request: Request):
    """Redirect user to Google for authentication."""
    redirect_uri = request.url_for('google_auth_callback')
    return await oauth.google.authorize_redirect(request, redirect_uri)

async def google_auth_callback(request: Request):
    """Handle the callback from Google after authentication."""
    try:
        token = await oauth.google.authorize_access_token(request)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=f"Google auth failed: {e}")
    
    user_info = token.get('userinfo')
    if not user_info:
        # Fetch user info if not included
        user_info = await oauth.google.userinfo(token=token)
        
    # Store token and user info securely (e.g., in the integrations table)
    # Link it to the Tempo user account
    access_token = token['access_token']
    refresh_token = token.get('refresh_token') # May not always be present
    expires_at = token.get('expires_at')
    
    # Placeholder for storing token - replace with actual DB logic
    print(f"Google Token Received for {user_info['email']}:")
    print(f" Access Token: {access_token[:10]}...")
    print(f" Refresh Token: {refresh_token[:10] if refresh_token else 'N/A'}...")
    print(f" Expires At: {expires_at}")
    
    # Redirect user back to frontend settings page
    # return RedirectResponse(url='/settings/integrations?status=google_success')
    return {"message": "Google authentication successful", "user_info": user_info}
```

### 3. Create API Endpoints for Integration Management

Add endpoints for initiating the OAuth flow and handling the callback.

```python
# Example: backend/app/routers/integrations.py (add Google routes)
from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session
from uuid import UUID

from app.db.session import get_db
from app.services.auth import get_current_user
from app.integrations.utils.oauth_handler import google_login, google_auth_callback
from app.services.integration_service import IntegrationService

router = APIRouter()

@router.get("/google/login")
async def initiate_google_auth(request: Request, current_user = Depends(get_current_user)):
    """Initiate Google Calendar integration via OAuth2."""
    # Store user_id in session or state for callback
    request.session['user_id_for_oauth'] = str(current_user.id)
    return await google_login(request)

@router.get("/google/callback")
async def handle_google_auth_callback(request: Request, db: Session = Depends(get_db)):
    """Handle the OAuth2 callback from Google."""
    user_id_str = request.session.get('user_id_for_oauth')
    if not user_id_str:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="OAuth state missing")
        
    user_id = UUID(user_id_str)
    
    # Process token and user info
    token_data = await google_auth_callback(request)
    
    # Save integration details to DB
    integration_service = IntegrationService(db)
    integration_service.save_google_integration(
        user_id=user_id,
        google_email=token_data['user_info']['email'],
        access_token=token_data['token']['access_token'],
        refresh_token=token_data['token'].get('refresh_token'),
        expires_at=token_data['token'].get('expires_at')
    )
    
    # Redirect or return success message
    return {"message": "Google Calendar integration successful!"}

# Add endpoints to list, configure, and delete integrations
@router.get("/")
def list_integrations(current_user = Depends(get_current_user), db: Session = Depends(get_db)):
    service = IntegrationService(db)
    return service.get_user_integrations(current_user.id)

@router.delete("/{integration_id}")
def delete_integration(integration_id: UUID, current_user = Depends(get_current_user), db: Session = Depends(get_db)):
    service = IntegrationService(db)
    service.delete_integration(current_user.id, integration_id)
    return {"message": "Integration deleted successfully"}
```

### 4. Implement the Provider-Specific Client

Create a class to interact with the external service's API.

```python
# Example: backend/app/integrations/providers/google_calendar.py
import httpx
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from authlib.integrations.httpx_client import OAuth2Auth

class GoogleCalendarClient:
    BASE_URL = "https://www.googleapis.com/calendar/v3"

    def __init__(self, access_token: str, refresh_token: Optional[str] = None, expires_at: Optional[int] = None):
        # Note: Need a mechanism to refresh the token if expired
        # This example uses a simple token; production needs refresh logic
        self.auth = OAuth2Auth(token={'access_token': access_token, 'token_type': 'Bearer'})
        self.client = httpx.AsyncClient(auth=self.auth)

    async def get_calendars(self) -> List[Dict]:
        """Fetch the list of user's calendars."""
        url = f"{self.BASE_URL}/users/me/calendarList"
        try:
            response = await self.client.get(url)
            response.raise_for_status() # Raise exception for 4xx/5xx errors
            return response.json().get("items", [])
        except httpx.HTTPStatusError as e:
            print(f"Error fetching calendars: {e.response.status_code} - {e.response.text}")
            # Handle specific errors (e.g., 401 Unauthorized)
            raise
        except Exception as e:
            print(f"Unexpected error fetching calendars: {e}")
            raise

    async def get_events(self, calendar_id: str = 'primary', time_min: Optional[datetime] = None, time_max: Optional[datetime] = None) -> List[Dict]:
        """Fetch events from a specific calendar."""
        url = f"{self.BASE_URL}/calendars/{calendar_id}/events"
        params = {
            'singleEvents': 'true',
            'orderBy': 'startTime',
            'maxResults': 250 # Adjust as needed
        }
        if time_min:
            params['timeMin'] = time_min.isoformat() + 'Z' # Google API expects RFC3339 format
        if time_max:
            params['timeMax'] = time_max.isoformat() + 'Z'

        try:
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            return response.json().get("items", [])
        except httpx.HTTPStatusError as e:
            print(f"Error fetching events: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            print(f"Unexpected error fetching events: {e}")
            raise

    async def close(self):
        await self.client.aclose()

    # Add methods for creating/updating events if write scope is requested
```

### 5. Implement Data Synchronization Logic

Create a service to handle syncing data between Tempo and the external service.

```python
# Example: backend/app/integrations/services/sync_service.py
from sqlalchemy.orm import Session
from uuid import UUID
from datetime import datetime, timedelta
import asyncio

from app.services.integration_service import IntegrationService
from app.services.calendar_service import CalendarService # Tempo's internal calendar service
from app.integrations.providers.google_calendar import GoogleCalendarClient
from app.models.calendar_event import CalendarEventCreate # Tempo's model

class SyncService:
    def __init__(self, db: Session):
        self.db = db
        self.integration_service = IntegrationService(db)
        self.calendar_service = CalendarService(db)

    async def sync_google_calendar(self, user_id: UUID):
        """Sync Google Calendar events for a user."""
        integration = self.integration_service.get_active_integration(user_id, 'google')
        if not integration or not integration.access_token:
            print(f"No active Google integration found for user {user_id}")
            return

        # TODO: Implement token refresh logic here if needed
        
        client = GoogleCalendarClient(access_token=integration.access_token)
        try:
            # Define sync window (e.g., past week to next month)
            time_min = datetime.utcnow() - timedelta(days=7)
            time_max = datetime.utcnow() + timedelta(days=30)

            google_events = await client.get_events(time_min=time_min, time_max=time_max)
            
            # Get existing Tempo events for comparison
            tempo_events = self.calendar_service.get_events_by_source(
                user_id=user_id, 
                source='google',
                start_time=time_min,
                end_time=time_max
            )
            tempo_event_map = {e.external_id: e for e in tempo_events}

            events_to_create = []
            events_to_update = []
            google_event_ids = set()

            for gevent in google_events:
                external_id = gevent['id']
                google_event_ids.add(external_id)

                # Basic data mapping (needs refinement)
                start = gevent.get('start', {}).get('dateTime') or gevent.get('start', {}).get('date')
                end = gevent.get('end', {}).get('dateTime') or gevent.get('end', {}).get('date')
                
                if not start or not end: continue # Skip events without proper times

                # Parse datetime (handle both date and dateTime)
                try:
                    start_time = datetime.fromisoformat(start.replace('Z', '+00:00'))
                    end_time = datetime.fromisoformat(end.replace('Z', '+00:00'))
                    is_all_day = 'date' in gevent.get('start', {})
                except ValueError:
                    continue # Skip invalid date formats

                event_data = {
                    'user_id': user_id,
                    'title': gevent.get('summary', 'No Title'),
                    'description': gevent.get('description'),
                    'start_time': start_time,
                    'end_time': end_time,
                    'location': gevent.get('location'),
                    'is_all_day': is_all_day,
                    'calendar_source': 'google',
                    'external_id': external_id,
                    'attendees': [{'email': a.get('email'), 'responseStatus': a.get('responseStatus')} 
                                  for a in gevent.get('attendees', [])]
                }

                if external_id in tempo_event_map:
                    # Check if update is needed
                    existing_event = tempo_event_map[external_id]
                    if self._needs_update(existing_event, event_data):
                        events_to_update.append({'id': existing_event.id, 'data': event_data})
                else:
                    # New event
                    events_to_create.append(CalendarEventCreate(**event_data))

            # Determine events to delete (present in Tempo but not in Google)
            events_to_delete = [
                e.id for e_id, e in tempo_event_map.items() 
                if e_id not in google_event_ids
            ]

            # Perform database operations
            if events_to_create:
                self.calendar_service.create_batch_events(events_to_create)
                print(f"Created {len(events_to_create)} Google Calendar events for user {user_id}")
            if events_to_update:
                self.calendar_service.update_batch_events(events_to_update)
                print(f"Updated {len(events_to_update)} Google Calendar events for user {user_id}")
            if events_to_delete:
                self.calendar_service.delete_batch_events(user_id, events_to_delete)
                print(f"Deleted {len(events_to_delete)} Google Calendar events for user {user_id}")

            # Update last sync time
            self.integration_service.update_last_sync(integration.id)

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                # Handle token expiry/revocation
                self.integration_service.deactivate_integration(integration.id, "Authentication failed")
            else:
                print(f"Sync failed for user {user_id}: {e}")
        except Exception as e:
            print(f"Unexpected sync error for user {user_id}: {e}")
        finally:
            await client.close()

    def _needs_update(self, existing_event, new_data) -> bool:
        """Check if a Tempo event needs updating based on new external data."""
        # Compare relevant fields (add more as needed)
        if existing_event.title != new_data['title']: return True
        if existing_event.description != new_data['description']: return True
        if existing_event.start_time != new_data['start_time']: return True
        if existing_event.end_time != new_data['end_time']: return True
        # Add more comparisons (location, attendees, etc.)
        return False

    async def trigger_sync_for_all_users(self):
        """Trigger sync for all users with active Google integrations."""
        active_integrations = self.integration_service.get_all_active_integrations('google')
        tasks = [self.sync_google_calendar(integration.user_id) for integration in active_integrations]
        await asyncio.gather(*tasks)
```

### 6. Schedule Synchronization

Use a task queue (like Celery with Redis/RabbitMQ) or a scheduler (like APScheduler) to run the sync process periodically.

```python
# Example: Using APScheduler within FastAPI
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from app.integrations.services.sync_service import SyncService
from app.db.session import SessionLocal

scheduler = AsyncIOScheduler()

@scheduler.scheduled_job('interval', minutes=15) # Sync every 15 minutes
async def run_google_sync():
    print("Running scheduled Google Calendar sync...")
    db = SessionLocal()
    try:
        sync_service = SyncService(db)
        await sync_service.trigger_sync_for_all_users()
    finally:
        db.close()

# In main.py, start the scheduler on application startup
@app.on_event("startup")
async def startup_event():
    scheduler.start()

@app.on_event("shutdown")
async def shutdown_event():
    scheduler.shutdown()
```

### 7. Update Frontend

- Add UI elements in the settings page for users to connect/disconnect integrations.
- Display synced data (e.g., Google Calendar events alongside Tempo tasks).
- Provide visual indicators for synced items.
- Show sync status and allow manual triggering.

## Best Practices for External Integrations

- **Secure Credential Storage**: Never store sensitive credentials (API keys, secrets, tokens) directly in code. Use environment variables, a secure configuration management system, or a dedicated secrets manager. Encrypt tokens stored in the database.
- **Implement Token Refresh**: For OAuth 2.0, implement logic to automatically refresh access tokens using the refresh token before they expire.
- **Rate Limiting**: Respect the rate limits of external APIs. Implement exponential backoff for retries.
- **Idempotency**: Design sync operations to be idempotent, meaning running them multiple times with the same input produces the same result.
- **Error Handling & Resilience**: Implement robust error handling for API calls and network issues. Log errors clearly. Consider using a dead-letter queue for failed sync tasks.
- **Data Mapping**: Carefully map data fields between Tempo and the external service. Handle differences in data structures and types gracefully.
- **User Control & Transparency**: Allow users to easily connect, disconnect, and configure integrations. Clearly indicate which data comes from external sources and when the last sync occurred.
- **Background Processing**: Perform synchronization tasks in the background (using task queues or schedulers) to avoid blocking API requests.
- **Monitoring**: Monitor the health and performance of integrations. Track API error rates and sync latency.
- **Graceful Degradation**: Ensure Tempo remains functional even if an external integration fails or is unavailable.

## Conclusion

Integrating external services transforms Tempo from a standalone tool into a central hub for productivity management. By carefully designing the architecture, handling authentication securely, implementing robust synchronization logic, and following best practices, you can build reliable and valuable integrations.

Start with the most critical integrations for your users and gradually expand. Remember that maintaining integrations requires ongoing effort, as external APIs can change. Keep your API client libraries updated and monitor for deprecation notices from service providers.
