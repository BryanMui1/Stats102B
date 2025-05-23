train = read.csv("mnist_train.csv")
training_data = train[train$label == 3 | train$label == 5, ]
training_data$label = as.numeric(training_data$label == 3)
model <- glm(label ~ ., data = training_data, family = binomial(link = "logit"), control = glm.control(maxit = 10000))
library(pROC)
test = read.csv("mnist_test.csv")
test_data = test[test$label == 3 | test$label == 5, ]
test_data$label = as.numeric(test_data$label == 3)
predictions <- predict(model, newdata = test_data, type = "response")
roc_obj <- roc(test_data$label ~ predictions)
plot(roc_obj, main = "ROC Curve", xlab = "False Positive Rate", ylab = "True Positive Rate")
auc_value <- auc(roc_obj)
print(auc_value)
