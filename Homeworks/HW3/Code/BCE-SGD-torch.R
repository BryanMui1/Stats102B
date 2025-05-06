library(torch)
library(MASS)
library(ggplot2)
library(dplyr)
library(tidyr)
library(ROCR)

# Parameters
seed_parameter = 123
n = 10000
p = 100
corr_factor = 0.5
flip_factor = 0.15
epochs = 50
batch_sizes = c(16, 64)
hidden_layers = list(c(32, 128), c(64, 128))  # two-layer configurations
base_lr = 0.1
decay_rate = 0.7

# Data generation
generate_correlated_data = function(n_samples = n, n_features = p, 
                                    rho = corr_factor, flip_f = flip_factor, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  Sigma = matrix(rho, nrow = n_features, ncol = n_features)
  diag(Sigma) = 1
  X = mvrnorm(n_samples, mu = rep(0, n_features), Sigma = Sigma)
  beta = runif(n_features, -1, 1)
  logits = X %*% beta
  probs = 1 / (1 + exp(-logits))
  y = rbinom(n_samples, 1, probs)
  flip_idx = sample(1:n_samples, size = flip_f * n_samples)
  y[flip_idx] = 1 - y[flip_idx]
  data.frame(y = y, X)
}

# Data
data = generate_correlated_data(n, p, corr_factor, flip_factor, seed_parameter)
X = as.matrix(data[, -1])
Y = as.numeric(data[, 1])

# Train/Val/Test split
train_idx = 1:floor(0.6 * n)
val_idx = (floor(0.6 * n) + 1):floor(0.8 * n)
test_idx = (floor(0.8 * n) + 1):n
X_train = X[train_idx, ]; Y_train = Y[train_idx]
X_val = X[val_idx, ]; Y_val = Y[val_idx]
X_test = X[test_idx, ]; Y_test = Y[test_idx]

# Torch tensors
x_train = torch_tensor(X_train, dtype = torch_float())
y_train = torch_tensor(matrix(Y_train), dtype = torch_float())
x_val = torch_tensor(X_val, dtype = torch_float())
y_val = torch_tensor(matrix(Y_val), dtype = torch_float())
x_test = torch_tensor(X_test, dtype = torch_float())
y_test = torch_tensor(matrix(Y_test), dtype = torch_float())

# MLP model definition
mlp_model = function(input_dim, hidden1, hidden2) {
  nn_module(
    initialize = function() {
      self$fc1 = nn_linear(input_dim, hidden1)
      self$fc2 = nn_linear(hidden1, hidden2)
      self$fc3 = nn_linear(hidden2, 1)
    },
    forward = function(x) {
      x %>% 
        self$fc1() %>% nnf_relu() %>%
        self$fc2() %>% nnf_relu() %>%
        self$fc3() %>% torch_sigmoid()
    }
  )
}

# Generate all combinations of hidden layer sizes
hidden_combinations = expand.grid(hidden_layers[[1]], hidden_layers[[2]])
hidden_combinations = as.data.frame(hidden_combinations)
colnames(hidden_combinations) = c("h1", "h2")

# Training loop
results = list()
best_val_loss = Inf
best_model = NULL
best_config = NULL

for (batch_size in batch_sizes) {
  for (i in 1:nrow(hidden_combinations)) {
    h1 = hidden_combinations$h1[i]
    h2 = hidden_combinations$h2[i]
    
    # Print the model configuration being used
    cat("Trying model with h1 =", h1, "and h2 =", h2, "\n")
    
    # Initialize model and optimizer
    model = mlp_model(p, h1, h2)()
    optimizer = optim_sgd(model$parameters, lr = base_lr)
    loss_fn = nn_bce_loss()
    
    train_loss_history = c()
    val_loss_history = c()
    
    for (epoch in 1:epochs) {
      # Learning rate decay
      lr = base_lr * decay_rate^(epoch - 1)
      optimizer$param_groups[[1]]$lr = lr
      
      # Shuffle the data
      idx = sample(nrow(x_train))
      x_shuffled = x_train[idx, ]
      y_shuffled = y_train[idx, ]
      
      for (i in seq(1, nrow(x_shuffled), by = batch_size)) {
        end_idx = min(i + batch_size - 1, nrow(x_shuffled))
        x_batch = x_shuffled[i:end_idx, ]
        y_batch = y_shuffled[i:end_idx, , drop = FALSE]
        
        optimizer$zero_grad()
        y_pred = model(x_batch)
        loss = loss_fn(y_pred, y_batch)
        loss$backward()
        optimizer$step()
      }
      
      # Track training and validation loss
      with_no_grad({
        train_pred <- model(x_train)
        val_pred <- model(x_val)
        train_loss <- loss_fn(train_pred, y_train)$item()
        val_loss <- loss_fn(val_pred, y_val)$item()
      })
      
      train_loss_history = c(train_loss_history, train_loss)
      val_loss_history = c(val_loss_history, val_loss)
    }
    
    # Store the results
    key = paste0("Batch_", batch_size, "_Hidden_", h1, "_", h2)
    results[[key]] = list(train = train_loss_history, val = val_loss_history)
    
    # Update best model if needed
    if (min(val_loss_history) < best_val_loss) {
      best_val_loss = min(val_loss_history)
      best_model = model
      best_config = key
    }
  }
}

cat("Best model config:", best_config, "\n")

# Plotting training and validation loss
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
loss_long = pivot_longer(loss_df, cols = c("Training", "Validation"),
                         names_to = "LossType", values_to = "Loss")

# Separate plots for Training and Validation loss
ggplot(loss_long %>% filter(LossType == "Training"),
       aes(x = Epoch, y = Loss, color = Model)) +
  geom_line() + theme_minimal() + ggtitle("Training Loss")

ggplot(loss_long %>% filter(LossType == "Validation"),
       aes(x = Epoch, y = Loss, color = Model)) +
  geom_line(linetype = "dashed") + theme_minimal() + ggtitle("Validation Loss")

