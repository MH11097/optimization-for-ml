"""
Data splitting utilities for machine learning workflows.
This module provides systematic data splitting capabilities including
train/test splitting, batch generation, and cross-validation support
with scientific reproducibility.
"""
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Union, Iterator
from enum import Enum
import warnings

class SplitMethod(Enum):
    """Enumeration of data splitting methods."""
    RANDOM = "random"
    STRATIFIED = "stratified" 
    TEMPORAL = "temporal"
    GROUPED = "grouped"

class DataSplitter:
    """
    Scientific data splitting for machine learning experiments.
    
    Provides systematic approaches to split data while maintaining
    statistical properties and ensuring reproducibility.
    """
    
    @staticmethod
    def train_test_split(X: np.ndarray,
                        y: np.ndarray,
                        test_size: float = 0.2,
                        random_state: Optional[int] = None,
                        shuffle: bool = True,
                        stratify: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into training and testing sets.
        
        Args:
            X: Feature matrix
            y: Target vector
            test_size: Proportion of test set (0.0 - 1.0)
            random_state: Random seed for reproducibility
            shuffle: Whether to shuffle data before splitting
            stratify: Array for stratified splitting
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
            
        Raises:
            ValueError: If inputs have incompatible shapes or invalid parameters
        """
        if len(X) != len(y):
            raise ValueError("X and y must have same number of samples")
            
        if not 0.0 < test_size < 1.0:
            raise ValueError("test_size must be between 0 and 1")
            
        n_samples = len(X)
        n_test = int(n_samples * test_size)
        
        if random_state is not None:
            np.random.seed(random_state)
            
        indices = np.arange(n_samples)
        
        if stratify is not None:
            return DataSplitter._stratified_split(X, y, indices, n_test, stratify)
        elif shuffle:
            np.random.shuffle(indices)
            
        # Simple random split
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]
        
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        
        return X_train, X_test, y_train, y_test
    
    @staticmethod
    def _stratified_split(X: np.ndarray, 
                         y: np.ndarray, 
                         indices: np.ndarray, 
                         n_test: int,
                         stratify: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Perform stratified splitting to maintain class distribution."""
        unique_classes, class_counts = np.unique(stratify, return_counts=True)
        
        train_indices = []
        test_indices = []
        
        for class_val, class_count in zip(unique_classes, class_counts):
            class_indices = indices[stratify == class_val]
            n_test_class = int(np.round(n_test * class_count / len(indices)))
            
            # Ensure at least one sample in test set for each class
            n_test_class = max(1, min(n_test_class, class_count - 1))
            
            np.random.shuffle(class_indices)
            test_indices.extend(class_indices[:n_test_class])
            train_indices.extend(class_indices[n_test_class:])
            
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        
        return X_train, X_test, y_train, y_test
    
    @staticmethod
    def train_val_test_split(X: np.ndarray,
                           y: np.ndarray,
                           test_size: float = 0.2,
                           val_size: float = 0.2,
                           random_state: Optional[int] = None,
                           shuffle: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into training, validation, and testing sets.
        
        Args:
            X: Feature matrix
            y: Target vector
            test_size: Proportion of test set
            val_size: Proportion of validation set (from remaining after test split)
            random_state: Random seed for reproducibility
            shuffle: Whether to shuffle data
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = DataSplitter.train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=shuffle)
        
        # Second split: separate validation from remaining training data
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = DataSplitter.train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, 
            random_state=random_state + 1 if random_state else None, shuffle=shuffle)
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    @staticmethod
    def temporal_split(X: np.ndarray,
                      y: np.ndarray,
                      timestamps: np.ndarray,
                      test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split time series data maintaining temporal order.
        
        Args:
            X: Feature matrix
            y: Target vector
            timestamps: Timestamps for temporal ordering
            test_size: Proportion of most recent data for testing
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        # Sort by timestamp
        sorted_indices = np.argsort(timestamps)
        X_sorted = X[sorted_indices]
        y_sorted = y[sorted_indices]
        
        # Split maintaining temporal order
        n_samples = len(X_sorted)
        n_train = int(n_samples * (1 - test_size))
        
        X_train, X_test = X_sorted[:n_train], X_sorted[n_train:]
        y_train, y_test = y_sorted[:n_train], y_sorted[n_train:]
        
        return X_train, X_test, y_train, y_test

class BatchGenerator:
    """
    Efficient batch generation for training workflows.
    
    Provides memory-efficient batch generation with support for
    shuffling, balanced sampling, and custom batch processing.
    """
    
    def __init__(self, 
                 batch_size: int = 32,
                 shuffle: bool = True,
                 drop_last: bool = False,
                 random_state: Optional[int] = None):
        """
        Initialize batch generator.
        
        Args:
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle data each epoch
            drop_last: Whether to drop last incomplete batch
            random_state: Random seed for reproducibility
        """
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.random_state = random_state
        
    def generate_batches(self, 
                        X: np.ndarray, 
                        y: np.ndarray) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate batches from data.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Yields:
            Tuples of (X_batch, y_batch)
        """
        if len(X) != len(y):
            raise ValueError("X and y must have same number of samples")
            
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        if self.shuffle:
            if self.random_state is not None:
                np.random.seed(self.random_state)
            np.random.shuffle(indices)
            
        # Generate batches
        for start_idx in range(0, n_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, n_samples)
            
            # Skip incomplete batch if drop_last is True
            if self.drop_last and (end_idx - start_idx) < self.batch_size:
                break
                
            batch_indices = indices[start_idx:end_idx]
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]
            
            yield X_batch, y_batch
    
    def get_batch_list(self, 
                      X: np.ndarray, 
                      y: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Get all batches as a list.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            List of (X_batch, y_batch) tuples
        """
        return list(self.generate_batches(X, y))
    
    def count_batches(self, n_samples: int) -> int:
        """
        Count the number of batches for given sample size.
        
        Args:
            n_samples: Number of samples
            
        Returns:
            Number of batches
        """
        if self.drop_last:
            return n_samples // self.batch_size
        else:
            return (n_samples + self.batch_size - 1) // self.batch_size

class CrossValidator:
    """
    Scientific cross-validation utilities.
    
    Provides systematic cross-validation approaches with proper
    statistical methodology and reproducibility.
    """
    
    @staticmethod
    def k_fold_split(X: np.ndarray,
                    y: np.ndarray,
                    k: int = 5,
                    shuffle: bool = True,
                    random_state: Optional[int] = None) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Generate K-fold cross-validation splits.
        
        Args:
            X: Feature matrix
            y: Target vector
            k: Number of folds
            shuffle: Whether to shuffle before splitting
            random_state: Random seed
            
        Returns:
            List of (X_train, X_val, y_train, y_val) tuples for each fold
        """
        if k <= 1:
            raise ValueError("k must be greater than 1")
            
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        if shuffle:
            if random_state is not None:
                np.random.seed(random_state)
            np.random.shuffle(indices)
        
        fold_size = n_samples // k
        splits = []
        
        for i in range(k):
            start_idx = i * fold_size
            end_idx = (i + 1) * fold_size if i < k - 1 else n_samples
            
            val_indices = indices[start_idx:end_idx]
            train_indices = np.concatenate([indices[:start_idx], indices[end_idx:]])
            
            X_train, X_val = X[train_indices], X[val_indices]
            y_train, y_val = y[train_indices], y[val_indices]
            
            splits.append((X_train, X_val, y_train, y_val))
            
        return splits

# Backward compatibility functions
def chia_train_test(X: np.ndarray, 
                   y: np.ndarray, 
                   test_size: float = 0.2,
                   random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Backward compatibility wrapper for train-test split."""
    return DataSplitter.train_test_split(X, y, test_size, random_state)

def tao_batches(X: np.ndarray, 
               y: np.ndarray, 
               batch_size: int = 32,
               shuffle: bool = True) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Backward compatibility wrapper for batch generation."""
    generator = BatchGenerator(batch_size=batch_size, shuffle=shuffle)
    return generator.get_batch_list(X, y)