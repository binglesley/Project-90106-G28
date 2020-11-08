# Reading Data In
library(glm2)
library(caret)
library(faraway)
library(e1071)
data <- read.csv("C:/Users/BingLesleyYuan/Desktop/UniProj/Project-90106-G28/merged_data_final.csv",header = TRUE)
names(data) <- c("X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "X9", "X10", "X11", "X12",
                 "X13", "X14", "X15","X16","X17","X18","X19","X20","X21","X22","X23")
summary(data)

mod1 <- glm(factor(X1)~factor(X2)+factor(X3)+X4+X5+X6+factor(X7)+factor(X8)+factor(X11)+X12+X13+X14+factor(X15)+factor(X16)+factor(X17)+X18+X19+factor(X20)+factor(X21)+factor(X22)+factor(X23),
            family = "binomial", data = data)
# 10-fold cross_validation
train_control <- trainControl(method = "cv", number = 10)
# train the model on data set
model <- train(factor(X1)~factor(X2)+factor(X3)+X4+X5+X6+factor(X11)+X14+factor(X15)+factor(X16)+factor(X17)+X18+X19+factor(X22),
               data = data,
               trControl = train_control,
               method = "glm",
               family="binomial")
# print cv scores
summary(model)

model$results


