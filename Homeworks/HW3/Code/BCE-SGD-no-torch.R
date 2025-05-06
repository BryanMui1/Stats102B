# Parameters to simulate binary classification dataset with label noise
seed_parameter=123
n = 10000 # sample size
p = 100 # number of covariates
corr_factor = 0.5 # correlation factor between predictors
flip_factor = 0.15 # label noise factor

# Set-up size of training, validation and test data set
train_idx <- 1:floor(0.6 * n)
val_idx <- (floor(0.6 * n) + 1):floor(0.8 * n)
test_idx <- (floor(0.8 * n) + 1):n

# set hyperparameters for MLP training
epochs = 50 # epochs
batch_sizes = c(4) # mini-batch size
hidden_sizes = c(16, 64, 256) # hidden layer dimension
base_step_size = 0.1 # initial value of step size 
decay_rate = 0.7 # decreases step size on a fixed schedule

# Generate correlated dataset
generate_correlated_data = function(n_samples = n, n_features = p, 
                                     rho = corr_factor, flip_f=flip_factor, seed=NULL) 
  {
  if (!is.null(seed)) set.seed(seed)
  
  # Covariance matrix with correlation rho
  Sigma = matrix(rho, nrow = n_features, ncol = n_features)
  diag(Sigma) = 1  # Set diagonal to 1
  
  # Generate predictors (X) from multivariate normal distribution
  mu = rep(0, n_features)
  X = mvrnorm(n = n_samples, mu = mu, Sigma = Sigma)
  
  # Define true coefficients for logistic regression
  beta = runif(n_features, -1, 1)
  logits = X %*% beta 
  
  # Convert logits to probabilities
  probs = 1 / (1 + exp(-logits))
  
  # Generate class labels
  y = rbinom(n_samples, 1, probs)
  # add noise in the labels by flipping an percentage of them
  flip_idx = sample(1:n_samples, size = flip_f * n_samples)
  y[flip_idx] = 1 - y[flip_idx]
  
  
  # Return data frame
  data = data.frame(y = y, X)
  colnames(data) = c("y", paste0("X", 1:n_features))
  
  return(data)
}

# Generate dataset
data = generate_correlated_data(n_samples = n, n_features = p, 
                        rho = corr_factor, flip_f=flip_factor, seed=seed_parameter)
X = as.matrix(data[, -1])  # Features
y = as.matrix(data[, 1])   # Labels

# Data split
X_train = X[train_idx, ]; y_train = y[train_idx]
X_val = X[val_idx, ]; y_val = y[val_idx]
X_test = X[test_idx, ]; y_test = y[test_idx]

# Helper functions
sigmoid = function(x) 1 / (1 + exp(-x))

relu = function(x) {
  x * (x > 0)
}

bce_loss = function(y_true, y_pred) {
  eps = 1e-8
  n = length(y_true)
  -sum(y_true * log(y_pred + eps) + (1 - y_true) * log(1 - y_pred + eps)) / n
}

# Xavier initialization function
xavier_init = function(in_dim, out_dim) {
  limit = sqrt(6 / (in_dim + out_dim))
  matrix(runif(in_dim * out_dim, min = -limit, max = limit), nrow = in_dim)
}

# He initialization function
he_init = function(in_dim, out_dim) {
  stddev = sqrt(2 / in_dim)
  matrix(rnorm(in_dim * out_dim, mean = 0, sd = stddev), nrow = in_dim)
}

# Store results
results = list()

# Initialize variables to track best model
best_model = NULL
best_val_loss = Inf

### THIS PART OF THE CODE CONTAINS THE CORE OF SGD FOR THE MLP 
# Training loop

for (batch_size in batch_sizes) {
  for (hidden_size in hidden_sizes) {
    
    # Initialize weights
    W1 <- he_init(p, hidden_size)
    b1 <- rep(0, hidden_size)
    W2 <- xavier_init(hidden_size, 1)
    W2 <- matrix(W2, nrow = hidden_size, ncol = 1)
    b2 <- 0
    
    train_loss_history <- c()
    val_loss_history <- c()
    
    for (epoch in 1:epochs) {
      # Learning rate schedule
      lr <- base_step_size * (decay_rate ^ (epoch - 1))
      
      # Shuffle training data
      idx <- sample(1:nrow(X_train))
      X_train <- X_train[idx, ]
      y_train <- y_train[idx]
      
      for (i in seq(1, nrow(X_train), by = batch_size)) {
        batch_end <- min(i + batch_size - 1, nrow(X_train))
        X_batch <- X_train[i:batch_end, , drop = FALSE]
        y_batch <- y_train[i:batch_end]
        
        # Forward pass
        A <- X_batch %*% W1 + matrix(b1, nrow(X_batch), hidden_size, byrow = TRUE)
        H <- relu(A)
        Z <- H %*% W2 + b2
        y_hat <- sigmoid(Z)
        
        # Backward pass
        dZ <- y_hat - matrix(y_batch, ncol = 1)
        dW2 <- t(H) %*% dZ / nrow(X_batch)
        db2 <- mean(dZ)
        
        dH <- dZ %*% t(W2)
        dA <- dH * (A > 0)  # ReLU derivative
        dW1 <- t(X_batch) %*% dA / nrow(X_batch)
        db1 <- colMeans(dA)
        
        # Update weights
        W1 <- W1 - lr * dW1
        b1 <- b1 - lr * db1
        W2 <- W2 - lr * dW2
        b2 <- b2 - lr * db2
      }
      
      # Compute and track loss
      A_train <- relu(X_train %*% W1 + matrix(b1, nrow(X_train), hidden_size, byrow = TRUE))
      y_train_pred <- sigmoid(A_train %*% W2 + b2)
      train_loss <- bce_loss(y_train, y_train_pred)
      
      A_val <- relu(X_val %*% W1 + matrix(b1, nrow(X_val), hidden_size, byrow = TRUE))
      y_val_pred <- sigmoid(A_val %*% W2 + b2)
      val_loss <- bce_loss(y_val, y_val_pred)
      
      train_loss_history <- c(train_loss_history, train_loss)
      val_loss_history <- c(val_loss_history, val_loss)
    }
    
    # Save training history
    key <- paste0("Batch_", batch_size, "_Hidden_", hidden_size)
    results[[key]] <- list(train = train_loss_history, val = val_loss_history)
    
    # Update best model
    if (min(val_loss_history) < best_val_loss) {
      best_val_loss <- min(val_loss_history)
      best_model <- list(W1 = W1, b1 = b1, W2 = W2, b2 = b2, val_loss = best_val_loss)
    }
  }
}

##################################################################


# Plot all models' training and validation losses

library(ggplot2)
library(dplyr)
library(tidyr)

# Prepare data frame for ggplot
loss_df = data.frame()
for (key in names(results)) {
  df = data.frame(
    Epoch = 1:epochs,
    Training = results[[key]]$train,
    Validation = results[[key]]$val,
    Model = key
  )
  loss_df = rbind(loss_df, df)
}

# Convert to long format for ggplot
loss_long = pivot_longer(loss_df, cols = c("Training", "Validation"),
                          names_to = "LossType", values_to = "Loss")

# === Training Loss Plot ===
ggplot(loss_long %>% filter(LossType == "Training"), 
       aes(x = Epoch, y = Loss, color = Model)) +
  geom_line(size = 1) +
  labs(title = "Training Loss over Epochs", y = "Training Loss") +
  theme_minimal() +
  theme(legend.position = "bottom")

# === Validation Loss Plot ===
ggplot(loss_long %>% filter(LossType == "Validation"), 
       aes(x = Epoch, y = Loss, color = Model)) +
  geom_line(size = 1, linetype = "dashed") +
  labs(title = "Validation Loss over Epochs", y = "Validation Loss") +
  theme_minimal() +
  theme(legend.position = "bottom")


