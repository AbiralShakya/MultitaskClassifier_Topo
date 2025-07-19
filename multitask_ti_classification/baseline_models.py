"""
Baseline Models - XGBoost and LightGBM baselines for comparison
with the neural fusion model.
"""

import numpy as np
import pandas as pd
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    xgb = None
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    lgb = None
    LIGHTGBM_AVAILABLE = False
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import json
from typing import Dict, List, Tuple, Optional
import torch
from torch_geometric.data import Data
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from helper.config import *


class FeatureExtractor:
    """Extract features from multimodal data for baseline models."""
    
    def __init__(self, use_crystal: bool = True, use_kspace: bool = True, 
                 use_scalar: bool = True, use_decomposition: bool = True, 
                 use_spectral: bool = True):
        self.use_crystal = use_crystal
        self.use_kspace = use_kspace
        self.use_scalar = use_scalar
        self.use_decomposition = use_decomposition
        self.use_spectral = use_spectral
        
    def extract_graph_features(self, graph_data: Data) -> np.ndarray:
        """Extract graph-level features from PyTorch Geometric data."""
        features = []
        
        # Node features statistics
        if hasattr(graph_data, 'x') and graph_data.x is not None:
            node_features = graph_data.x.numpy()
            features.extend([
                np.mean(node_features, axis=0),  # Mean node features
                np.std(node_features, axis=0),   # Std node features
                np.max(node_features, axis=0),   # Max node features
                np.min(node_features, axis=0),   # Min node features
                np.median(node_features, axis=0) # Median node features
            ])
        
        # Edge features statistics
        if hasattr(graph_data, 'edge_attr') and graph_data.edge_attr is not None:
            edge_features = graph_data.edge_attr.numpy()
            features.extend([
                np.mean(edge_features, axis=0),  # Mean edge features
                np.std(edge_features, axis=0),   # Std edge features
                np.max(edge_features, axis=0),   # Max edge features
                np.min(edge_features, axis=0),   # Min edge features
            ])
        
        # Graph topology features
        if hasattr(graph_data, 'edge_index') and graph_data.edge_index is not None:
            edge_index = graph_data.edge_index.numpy()
            num_nodes = graph_data.num_nodes if hasattr(graph_data, 'num_nodes') else len(graph_data.x)
            num_edges = edge_index.shape[1]
            
            # Degree statistics
            degrees = np.zeros(num_nodes)
            for i in range(edge_index.shape[1]):
                degrees[edge_index[0, i]] += 1
                degrees[edge_index[1, i]] += 1
            
            features.extend([
                np.mean(degrees),      # Average degree
                np.std(degrees),       # Degree std
                np.max(degrees),       # Max degree
                np.min(degrees),       # Min degree
                num_edges / num_nodes, # Edge density
                num_nodes,             # Number of nodes
                num_edges             # Number of edges
            ])
        
        return np.concatenate(features) if features else np.array([])
    
    def extract_all_features(self, batch_data: Dict[str, torch.Tensor]) -> np.ndarray:
        """Extract all features from multimodal batch data."""
        all_features = []
        
        # Crystal graph features
        if self.use_crystal and 'crystal_x' in batch_data and 'crystal_edge_index' in batch_data:
            crystal_data = Data(
                x=batch_data['crystal_x'],
                edge_index=batch_data['crystal_edge_index']
            )
            crystal_features = self.extract_graph_features(crystal_data)
            all_features.append(crystal_features)
        
        # K-space graph features
        if self.use_kspace and 'kspace_x' in batch_data and 'kspace_edge_index' in batch_data:
            kspace_data = Data(
                x=batch_data['kspace_x'],
                edge_index=batch_data['kspace_edge_index']
            )
            kspace_features = self.extract_graph_features(kspace_data)
            all_features.append(kspace_features)
        
        # Scalar features
        if self.use_scalar and 'scalar_features' in batch_data:
            scalar_features = batch_data['scalar_features'].numpy()
            all_features.append(scalar_features)
        
        # Decomposition features
        if self.use_decomposition and 'decomposition_features' in batch_data:
            decomposition_features = batch_data['decomposition_features'].numpy()
            all_features.append(decomposition_features)
        
        # Spectral features (if available)
        if self.use_spectral and 'spectral_features' in batch_data:
            spectral_features = batch_data['spectral_features'].numpy()
            all_features.append(spectral_features)
        
        # Concatenate all features
        if all_features:
            return np.concatenate(all_features, axis=1)
        else:
            return np.array([])


class BaselineModels:
    """Collection of baseline models for comparison."""
    
    def __init__(self, feature_extractor: FeatureExtractor):
        self.feature_extractor = feature_extractor
        self.models = {}
        self.scalers = {}
        self.results = {}
        
    def prepare_data(self, train_loader, val_loader, test_loader) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for baseline models."""
        
        print("Extracting features from training data...")
        X_train, y_train = self._extract_features_from_loader(train_loader)
        
        print("Extracting features from validation data...")
        X_val, y_val = self._extract_features_from_loader(val_loader)
        
        print("Extracting features from test data...")
        X_test, y_test = self._extract_features_from_loader(test_loader)
        
        print(f"Feature dimensions: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def _extract_features_from_loader(self, data_loader) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features from a data loader."""
        all_features = []
        all_labels = []
        
        for batch_data, labels in data_loader:
            # Extract features for each sample in batch
            batch_features = []
            for i in range(len(labels)):
                # Create single sample data
                sample_data = {}
                for key, value in batch_data.items():
                    if isinstance(value, torch.Tensor):
                        if key.endswith('_x'):
                            sample_data[key] = value[i:i+1]  # Single sample
                        elif key.endswith('_edge_index'):
                            sample_data[key] = value  # Keep edge index as is
                        else:
                            sample_data[key] = value[i:i+1]  # Single sample
                    else:
                        sample_data[key] = value
                
                # Extract features
                features = self.feature_extractor.extract_all_features(sample_data)
                if len(features.shape) == 1:
                    features = features.reshape(1, -1)
                batch_features.append(features.flatten())
            
            all_features.extend(batch_features)
            all_labels.extend(labels.numpy())
        
        return np.array(all_features), np.array(all_labels)
    
    def train_xgboost(self, X_train: np.ndarray, y_train: np.ndarray, 
                     X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """Train XGBoost model with hyperparameter tuning."""
        
        print("Training XGBoost model...")
        
        # Hyperparameter search space
        param_grid = {
            'n_estimators': [100, 200, 500],
            'max_depth': [3, 6, 9, 12],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.1, 1],
            'reg_lambda': [0, 0.1, 1]
        }
        
        # Grid search
        xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
        grid_search = GridSearchCV(
            xgb_model, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Best model
        best_model = grid_search.best_estimator_
        self.models['xgboost'] = best_model
        
        # Evaluate
        train_acc = accuracy_score(y_train, best_model.predict(X_train))
        val_acc = accuracy_score(y_val, best_model.predict(X_val))
        
        results = {
            'model': 'XGBoost',
            'best_params': grid_search.best_params_,
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'feature_importance': best_model.feature_importances_
        }
        
        self.results['xgboost'] = results
        
        print(f"XGBoost - Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
        
        return results
    
    def train_lightgbm(self, X_train: np.ndarray, y_train: np.ndarray, 
                      X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """Train LightGBM model with hyperparameter tuning."""
        
        print("Training LightGBM model...")
        
        # Hyperparameter search space
        param_grid = {
            'n_estimators': [100, 200, 500],
            'max_depth': [3, 6, 9, 12],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.1, 1],
            'reg_lambda': [0, 0.1, 1]
        }
        
        # Grid search
        lgb_model = lgb.LGBMClassifier(random_state=42, verbose=-1)
        grid_search = GridSearchCV(
            lgb_model, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Best model
        best_model = grid_search.best_estimator_
        self.models['lightgbm'] = best_model
        
        # Evaluate
        train_acc = accuracy_score(y_train, best_model.predict(X_train))
        val_acc = accuracy_score(y_val, best_model.predict(X_val))
        
        results = {
            'model': 'LightGBM',
            'best_params': grid_search.best_params_,
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'feature_importance': best_model.feature_importances_
        }
        
        self.results['lightgbm'] = results
        
        print(f"LightGBM - Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
        
        return results
    
    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray, 
                           X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """Train Random Forest model."""
        
        print("Training Random Forest model...")
        
        # Hyperparameter search
        param_grid = {
            'n_estimators': [100, 200, 500],
            'max_depth': [3, 6, 9, 12, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        rf_model = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(
            rf_model, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Best model
        best_model = grid_search.best_estimator_
        self.models['random_forest'] = best_model
        
        # Evaluate
        train_acc = accuracy_score(y_train, best_model.predict(X_train))
        val_acc = accuracy_score(y_val, best_model.predict(X_val))
        
        results = {
            'model': 'Random Forest',
            'best_params': grid_search.best_params_,
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'feature_importance': best_model.feature_importances_
        }
        
        self.results['random_forest'] = results
        
        print(f"Random Forest - Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
        
        return results
    
    def train_logistic_regression(self, X_train: np.ndarray, y_train: np.ndarray, 
                                 X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """Train Logistic Regression model."""
        
        print("Training Logistic Regression model...")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        self.scalers['logistic_regression'] = scaler
        
        # Hyperparameter search
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']
        }
        
        lr_model = LogisticRegression(random_state=42, max_iter=1000)
        grid_search = GridSearchCV(
            lr_model, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train_scaled, y_train)
        
        # Best model
        best_model = grid_search.best_estimator_
        self.models['logistic_regression'] = best_model
        
        # Evaluate
        train_acc = accuracy_score(y_train, best_model.predict(X_train_scaled))
        val_acc = accuracy_score(y_val, best_model.predict(X_val_scaled))
        
        results = {
            'model': 'Logistic Regression',
            'best_params': grid_search.best_params_,
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'feature_importance': np.abs(best_model.coef_[0])
        }
        
        self.results['logistic_regression'] = results
        
        print(f"Logistic Regression - Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
        
        return results
    
    def train_all_models(self, X_train: np.ndarray, y_train: np.ndarray, 
                        X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """Train all baseline models."""
        
        results = {}
        
        # Train each model
        results['xgboost'] = self.train_xgboost(X_train, y_train, X_val, y_val)
        results['lightgbm'] = self.train_lightgbm(X_train, y_train, X_val, y_val)
        results['random_forest'] = self.train_random_forest(X_train, y_train, X_val, y_val)
        results['logistic_regression'] = self.train_logistic_regression(X_train, y_train, X_val, y_val)
        
        return results
    
    def evaluate_on_test(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Evaluate all models on test set."""
        
        test_results = {}
        
        for name, model in self.models.items():
            if name == 'logistic_regression':
                # Scale test data
                X_test_scaled = self.scalers[name].transform(X_test)
                predictions = model.predict(X_test_scaled)
                probabilities = model.predict_proba(X_test_scaled)
            else:
                predictions = model.predict(X_test)
                probabilities = model.predict_proba(X_test)
            
            # Calculate metrics with warning suppression
            accuracy = accuracy_score(y_test, predictions)
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                precision, recall, f1, _ = precision_recall_fscore_support(
                    y_test, predictions, average='weighted'
                )
            
            test_results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'predictions': predictions,
                'probabilities': probabilities
            }
            
            print(f"{name} - Test Acc: {accuracy:.4f}, F1: {f1:.4f}")
        
        return test_results
    
    def plot_results(self, test_results: Dict, save_path: str = 'baseline_results.png'):
        """Plot baseline model results."""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy comparison
        models = list(test_results.keys())
        accuracies = [test_results[model]['accuracy'] for model in models]
        
        ax1.bar(models, accuracies)
        ax1.set_title('Test Accuracy Comparison')
        ax1.set_ylabel('Accuracy')
        ax1.tick_params(axis='x', rotation=45)
        
        # F1 score comparison
        f1_scores = [test_results[model]['f1'] for model in models]
        
        ax2.bar(models, f1_scores)
        ax2.set_title('Test F1 Score Comparison')
        ax2.set_ylabel('F1 Score')
        ax2.tick_params(axis='x', rotation=45)
        
        # Feature importance (XGBoost)
        if 'xgboost' in self.results:
            importance = self.results['xgboost']['feature_importance']
            top_indices = np.argsort(importance)[-20:]  # Top 20 features
            
            ax3.barh(range(len(top_indices)), importance[top_indices])
            ax3.set_title('XGBoost Feature Importance (Top 20)')
            ax3.set_xlabel('Importance')
        
        # Confusion matrix (best model)
        best_model = max(test_results.keys(), key=lambda x: test_results[x]['accuracy'])
        predictions = test_results[best_model]['predictions']
        cm = confusion_matrix(y_test, predictions)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4)
        ax4.set_title(f'Confusion Matrix - {best_model}')
        ax4.set_xlabel('Predicted')
        ax4.set_ylabel('True')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_models(self, save_dir: str = 'baseline_models'):
        """Save trained models."""
        
        os.makedirs(save_dir, exist_ok=True)
        
        for name, model in self.models.items():
            model_path = os.path.join(save_dir, f'{name}.pkl')
            joblib.dump(model, model_path)
        
        # Save scalers
        for name, scaler in self.scalers.items():
            scaler_path = os.path.join(save_dir, f'{name}_scaler.pkl')
            joblib.dump(scaler, scaler_path)
        
        # Save results
        results_path = os.path.join(save_dir, 'results.json')
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"Models saved to {save_dir}")


def run_baseline_pipeline(train_loader, val_loader, test_loader, 
                         save_dir: str = 'baseline_results') -> Dict:
    """Run complete baseline model pipeline."""
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Create feature extractor
    feature_extractor = FeatureExtractor(
        use_crystal=True,
        use_kspace=True,
        use_scalar=True,
        use_decomposition=True,
        use_spectral=True
    )
    
    # Create baseline models
    baseline_models = BaselineModels(feature_extractor)
    
    # Prepare data
    X_train, y_train, X_val, y_val, X_test, y_test = baseline_models.prepare_data(
        train_loader, val_loader, test_loader
    )
    
    # Train all models
    train_results = baseline_models.train_all_models(X_train, y_train, X_val, y_val)
    
    # Evaluate on test set
    test_results = baseline_models.evaluate_on_test(X_test, y_test)
    
    # Plot results
    baseline_models.plot_results(test_results, os.path.join(save_dir, 'baseline_results.png'))
    
    # Save models
    baseline_models.save_models(os.path.join(save_dir, 'models'))
    
    # Save test results
    with open(os.path.join(save_dir, 'test_results.json'), 'w') as f:
        json.dump(test_results, f, indent=2, default=str)
    
    print(f"\nBaseline pipeline completed! Results saved to {save_dir}")
    
    return test_results


if __name__ == "__main__":
    # Test feature extraction
    print("Testing baseline models...")
    
    # Create dummy data
    feature_extractor = FeatureExtractor()
    baseline_models = BaselineModels(feature_extractor)
    
    print("Baseline models initialized successfully!") 