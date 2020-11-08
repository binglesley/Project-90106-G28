# Reading Data In
library(glm2)
library(caret)
library(faraway)
library(e1071)
data <- read.csv("C:/Users/BingLesleyYuan/Documents/WeChat Files/lesleyice/FileStorage/File/2020-09/cleaned_4.csv",header = TRUE)
names(data) <- c("X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "X9", "X10", "X11", "X12",
                 "X13", "X14", "X15","X16")
summary(data)

# 10-fold cross_validation
train_control <- trainControl(method = "cv", number = 10)
# train the model on data set
model <- train(factor(X2)~factor(X4)+X5+X6+X8+factor(X9)+X10+X11+X12+factor(X13)+X14+X15+X16,
               data = data,
               trControl = train_control,
               method = "glm",
               family="binomial")
# print cv scores
summary(model)

model$results
