gradient_descent <- function(X, y, eta = NULL, tol = 1e-6, max_iter = 10000, 
                             backtracking = TRUE, epsilon = 0.5, tau = 0.8) {
  # Initialize
  n <- nrow(X)
  p <- ncol(X)
  beta <- rep(0, p)
  obj_values <- numeric(max_iter)
  eta_values <- numeric(max_iter)  # To store eta values used each iteration
  eta_bt <- 1  # Initial step size for backtracking
  
  # Objective function: Mean Squared Error (MSE)
  obj_function <- function(beta) {
    sum((X %*% beta - y)^2) / (2 * n)
  }
  
  # Gradient function
  gradient <- function(beta) {
    t(X) %*% (X %*% beta - y) / n
  }
  
  for (iter in 1:max_iter) {
    grad <- gradient(beta)
    
    if (backtracking) {
      if (iter == 1) eta_bt <- 1  # Reset only in the first iteration
      beta_new <- beta - eta_bt * grad
      
      while (obj_function(beta_new) > obj_function(beta) - epsilon * eta_bt * sum(grad^2)) {
        eta_bt <- tau * eta_bt
        beta_new <- beta - eta_bt * grad
      }
      eta_used <- eta_bt
    } else {
      if (is.null(eta)) stop("When backtracking is FALSE, a fixed eta must be provided.")
      beta_new <- beta - eta * grad
      eta_used <- eta
    }
    
    eta_values[iter] <- eta_used
    
    obj_values[iter] <- obj_function(beta_new)
    
    if (sqrt(sum((beta_new - beta)^2)) < tol) {
      obj_values <- obj_values[1:iter]
      eta_values <- eta_values[1:iter]
      break
    }
    
    beta <- beta_new
  }
  
  return(list(beta = beta, obj_values = obj_values, eta_values = eta_values))
}

# Example usage
set.seed(42)
X <- matrix(rnorm(100 * 2), 100, 5)  # 100 samples, 2 features
y <- X %*% c(2, -3, 1, 5, -2) + rnorm(100)  # True coefficients (2, -3) with noise

result <- gradient_descent(X, y, backtracking = TRUE)  # Specify eta when backtracking is FALSE

# Plot objective function values over iterations
plot(result$obj_values, type = "o", col = "blue", pch = 16,
     xlab = "Iteration", ylab = "Objective Function Value",
     main = "Gradient Descent Convergence")
