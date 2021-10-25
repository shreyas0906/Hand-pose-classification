from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau

def model_checkpoint(save_path):
    return ModelCheckpoint(save_path,
                            monitor='accuracy',
                            verbose=1,
                            save_best_only=True, 
                            mode='max')

def tensorboard():
    return TensorBoard(log_dir='logs')

def reduce_lr_on_plateau():
    return ReduceLROnPlateau(
        monitor="accuracy",
        factor=0.1,
        patience=3,
        verbose=0,
        mode="auto",
        min_delta=0.01,
        cooldown=0,
        min_lr=0.00001,
    )
