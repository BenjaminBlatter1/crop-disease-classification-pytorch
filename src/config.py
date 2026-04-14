"""
Centralized configuration settings for the tomato leaf disease classification project.

This module defines the Config class, which stores all project-wide constants
related to dataset paths, training hyperparameters and model settings. Centralizing 
these values ensures consistent usage across the codebase and simplifies future 
modifications.
"""

class Config:
    """
    Container for global configuration parameters used throughout the project.

    Attributes:
        data_dir (str): Root directory containing raw and processed datasets.
        train_dir (str): Directory containing the training split.
        val_dir (str): Directory containing the validation split.

        batch_size (int): Number of samples per training batch.
        num_workers (int): Number of subprocesses used for DataLoader operations.
        learning_rate (float): Learning rate for the optimizer.

        image_size (tuple[int, int]): Target spatial resolution for input images.
        
        use_augmentation (bool): Enables optional data augmentation during
        training. When True, the training transform pipeline applies random
        flips, rotations, and color jitter to improve model robustness.
    """

    # Data
    data_dir = "data"
    train_dir = "data/processed/train"
    val_dir = "data/processed/val"

    # Training
    batch_size = 32
    num_workers = 4
    learning_rate = 1e-3

    # Model
    image_size = (224, 224)
    
    #Misc
    use_augmentation = False
