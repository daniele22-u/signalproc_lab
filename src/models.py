"""
Model Implementations
Contains all model classes: MBeat, MHRV, MMorph, MMorph+HRV, and Fusion Model
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import joblib


class BaseModel:
    """Base class for all models"""
    
    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        """
        Initialize base model
        
        Args:
            n_estimators: Number of trees in Random Forest
            max_depth: Maximum depth of trees
            random_state: Random state for reproducibility
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )
        self.is_trained = False
    
    def train(self, X, y):
        """Train the model"""
        self.model.fit(X, y)
        self.is_trained = True
    
    def predict(self, X):
        """Predict class labels"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        return self.model.predict_proba(X)
    
    def evaluate(self, X, y):
        """
        Evaluate model performance
        
        Returns:
            Dictionary of evaluation metrics
        """
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'sensitivity': recall_score(y, y_pred, zero_division=0),  # Same as recall
            'f1_score': f1_score(y, y_pred, zero_division=0),
        }
        
        # Calculate specificity
        cm = confusion_matrix(y, y_pred)
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        else:
            metrics['specificity'] = 0.0
        
        # AUC
        if len(np.unique(y)) > 1:
            metrics['auc'] = roc_auc_score(y, y_proba)
        else:
            metrics['auc'] = 0.0
        
        return metrics
    
    def save(self, filepath):
        """Save model to file"""
        joblib.dump(self.model, filepath)
    
    def load(self, filepath):
        """Load model from file"""
        self.model = joblib.load(filepath)
        self.is_trained = True


class MBeat(BaseModel):
    """Beat-level model using morphology features"""
    
    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        super().__init__(n_estimators, max_depth, random_state)
        self.model_type = 'MBeat'


class MHRV(BaseModel):
    """Interval-level model using HRV features"""
    
    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        super().__init__(n_estimators, max_depth, random_state)
        self.model_type = 'MHRV'


class MMorph(BaseModel):
    """Interval-level model using aggregated morphology features"""
    
    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        super().__init__(n_estimators, max_depth, random_state)
        self.model_type = 'MMorph'


class MMorphHRV(BaseModel):
    """Interval-level model combining morphology and HRV features"""
    
    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        super().__init__(n_estimators, max_depth, random_state)
        self.model_type = 'MMorph+HRV'


class FusionModel:
    """
    Fusion model that combines multiple beat-level models trained at different thresholds
    """
    
    def __init__(self, thresholds_hypo=None, thresholds_hyper=None, 
                 n_estimators=100, max_depth=None, random_state=42):
        """
        Initialize Fusion Model
        
        Args:
            thresholds_hypo: List of thresholds for hypoglycemia (default: [55,60,65,70,75,80,85,90])
            thresholds_hyper: List of thresholds for hyperglycemia (default: [150,165,180,200,225,250])
            n_estimators: Number of trees in Random Forest
            max_depth: Maximum depth of trees
            random_state: Random state for reproducibility
        """
        if thresholds_hypo is None:
            self.thresholds_hypo = [55, 60, 65, 70, 75, 80, 85, 90]
        else:
            self.thresholds_hypo = thresholds_hypo
        
        if thresholds_hyper is None:
            self.thresholds_hyper = [150, 165, 180, 200, 225, 250]
        else:
            self.thresholds_hyper = thresholds_hyper
        
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        
        # Create beat-level models for each threshold
        self.beat_models_hypo = {}
        self.beat_models_hyper = {}
        
        # Final interval-level model
        self.interval_model = None
        self.is_trained = False
        self.task = None  # 'hypoglycemia' or 'hyperglycemia'
    
    def train_beat_models(self, X_beats, glucose_values, task='hypoglycemia'):
        """
        Train beat-level models at different thresholds
        
        Args:
            X_beats: Feature matrix for beats (morphology features)
            glucose_values: Corresponding glucose values for each beat
            task: 'hypoglycemia' or 'hyperglycemia'
        """
        self.task = task
        
        if task == 'hypoglycemia':
            thresholds = self.thresholds_hypo
            models_dict = self.beat_models_hypo
        else:
            thresholds = self.thresholds_hyper
            models_dict = self.beat_models_hyper
        
        for threshold in thresholds:
            # Create labels based on threshold
            if task == 'hypoglycemia':
                y = (glucose_values < threshold).astype(int)
            else:
                y = (glucose_values > threshold).astype(int)
            
            # Train beat-level model
            model = MBeat(self.n_estimators, self.max_depth, self.random_state)
            model.train(X_beats, y)
            models_dict[threshold] = model
    
    def get_fusion_features(self, X_beats, task='hypoglycemia'):
        """
        Get fusion features from beat-level model predictions
        
        Args:
            X_beats: Feature matrix for beats
            task: 'hypoglycemia' or 'hyperglycemia'
            
        Returns:
            Array of fusion features (probabilities from each threshold model)
        """
        if task == 'hypoglycemia':
            thresholds = self.thresholds_hypo
            models_dict = self.beat_models_hypo
        else:
            thresholds = self.thresholds_hyper
            models_dict = self.beat_models_hyper
        
        fusion_features = []
        for threshold in thresholds:
            model = models_dict[threshold]
            proba = model.predict_proba(X_beats)[:, 1]
            fusion_features.append(proba)
        
        return np.column_stack(fusion_features)
    
    def train_interval_model(self, X_intervals, y_intervals):
        """
        Train final interval-level model using fusion features + HRV features
        
        Args:
            X_intervals: Feature matrix for intervals (fusion features + HRV)
            y_intervals: Labels for intervals
        """
        self.interval_model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.interval_model.fit(X_intervals, y_intervals)
        self.is_trained = True
    
    def predict(self, X_intervals):
        """Predict class labels for intervals"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        return self.interval_model.predict(X_intervals)
    
    def predict_proba(self, X_intervals):
        """Predict class probabilities for intervals"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        return self.interval_model.predict_proba(X_intervals)
    
    def evaluate(self, X_intervals, y_intervals):
        """Evaluate interval-level model"""
        y_pred = self.predict(X_intervals)
        y_proba = self.predict_proba(X_intervals)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_intervals, y_pred),
            'precision': precision_score(y_intervals, y_pred, zero_division=0),
            'recall': recall_score(y_intervals, y_pred, zero_division=0),
            'sensitivity': recall_score(y_intervals, y_pred, zero_division=0),
            'f1_score': f1_score(y_intervals, y_pred, zero_division=0),
        }
        
        # Calculate specificity
        tn, fp, fn, tp = confusion_matrix(y_intervals, y_pred).ravel()
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # AUC
        if len(np.unique(y_intervals)) > 1:
            metrics['auc'] = roc_auc_score(y_intervals, y_proba)
        else:
            metrics['auc'] = 0.0
        
        return metrics


class MajorityVotingModel:
    """Majority voting model (MMV) - uses beat-level predictions with majority voting"""
    
    def __init__(self, beat_model, majority_threshold=0.5):
        """
        Initialize majority voting model
        
        Args:
            beat_model: Trained beat-level model
            majority_threshold: Threshold for majority voting (default: 0.5)
        """
        self.beat_model = beat_model
        self.majority_threshold = majority_threshold
    
    def predict_interval(self, beat_predictions):
        """
        Predict interval label based on majority voting of beat predictions
        
        Args:
            beat_predictions: Array of beat-level predictions (probabilities)
            
        Returns:
            Interval prediction (0 or 1)
        """
        # Convert probabilities to binary predictions
        binary_preds = (beat_predictions > 0.5).astype(int)
        
        # Majority voting
        majority = np.mean(binary_preds)
        return 1 if majority >= self.majority_threshold else 0
    
    def evaluate(self, beat_predictions_by_interval, y_intervals):
        """
        Evaluate majority voting model
        
        Args:
            beat_predictions_by_interval: List of beat prediction arrays for each interval
            y_intervals: True labels for intervals
            
        Returns:
            Dictionary of evaluation metrics
        """
        y_pred = np.array([self.predict_interval(preds) for preds in beat_predictions_by_interval])
        
        metrics = {
            'accuracy': accuracy_score(y_intervals, y_pred),
            'precision': precision_score(y_intervals, y_pred, zero_division=0),
            'recall': recall_score(y_intervals, y_pred, zero_division=0),
            'sensitivity': recall_score(y_intervals, y_pred, zero_division=0),
            'f1_score': f1_score(y_intervals, y_pred, zero_division=0),
        }
        
        # Calculate specificity
        cm = confusion_matrix(y_intervals, y_pred)
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        else:
            metrics['specificity'] = 0.0
        
        return metrics
