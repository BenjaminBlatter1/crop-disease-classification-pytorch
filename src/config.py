class Config:
    #Data
    data_dir = "data"
    train_dir = "data/processed/train"
    val_dir = "data/processed/val"
    
    # Training
    batch_size = 32
    num_epochs = 10
    learning_rate = 1e-3
    num_workers = 4
    num_classes = 2

    # Model
    image_size = (224, 224)
    device = "cuda"