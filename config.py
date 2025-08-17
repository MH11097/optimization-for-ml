"""Configuration settings using Pydantic BaseSettings"""

from pydantic_settings import BaseSettings
from typing import Optional, Tuple
from pathlib import Path


class Settings(BaseSettings):
    """Application settings - reads from .env file"""
    
    # Model settings
    random_state: int = 42
    test_size: float = 0.2
    
    # Gradient Descent
    gd_learning_rate: float = 0.01
    gd_max_iterations: int = 1000
    gd_tolerance: float = 1e-6
    
    # Newton method
    newton_regularization: float = 1e-8
    
    # Visualization settings
    figure_size: Tuple[int, int] = (12, 8)
    dpi: int = 100
    save_plots: bool = True
    plot_format: str = "png"
    
    # Data processing settings
    chunk_size: int = 5000
    max_rows: Optional[int] = 100000
    missing_threshold: float = 0.5
    
    # Optional Redis cache settings
    redis_url: Optional[str] = None
    cache_ttl: int = 1800
    
    # Project paths
    @property
    def data_dir(self) -> Path:
        return Path("data")
    
    @property
    def model_output_dir(self) -> Path:
        return self.data_dir / "model_output"
    
    @property
    def processed_data_dir(self) -> Path:
        return self.data_dir / "processed"
    
    class Config:
        env_file = ".env"
        env_prefix = "ML_"
        case_sensitive = False


# Global instance
settings = Settings()

# Ensure directories exist
settings.data_dir.mkdir(exist_ok=True)
settings.model_output_dir.mkdir(exist_ok=True)
settings.processed_data_dir.mkdir(exist_ok=True)