def train(model, X_train, y_train):
    model.fit(X_train, y_train, epochs = 10, batch_size = 32)

    return model
