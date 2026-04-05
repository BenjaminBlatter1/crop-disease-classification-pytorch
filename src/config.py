class Config:
    #Data
    data_dir = "data"
    
    # Training
    batch_size = 32
    num_epochs = 10
    learning_rate = 1e-3
    num_classes = 2

    # Model
    device = "cuda"