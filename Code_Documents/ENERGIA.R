
library(readxl)
library(RWeka)
library(RWekajars)
library(RSNNS)
library(e1071)
library(caret)


#Data of Cross-Validation of ANFIS
CV_tr_1 <- read_excel("CrossV_tr1.xlsx")
CV_tr_2 <- read_excel("CrossV_tr2.xlsx")
CV_tr_3 <- read_excel("CrossV_tr3.xlsx")
CV_tr_4 <- read_excel("CrossV_tr4.xlsx")
CV_tr_5 <- read_excel("CrossV_tr5.xlsx")
#Test CV
CV_ts_1 <- read_excel("CrossV_ts1.xlsx")
CV_ts_2 <- read_excel("CrossV_ts2.xlsx")
CV_ts_3 <- read_excel("CrossV_ts3.xlsx")
CV_ts_4 <- read_excel("CrossV_ts4.xlsx")
CV_ts_5 <- read_excel("CrossV_ts5.xlsx")

Error <- function(y_pred, y_test){
  n = length(y_test);
  # EMA
  EMA = sum(abs(y_pred - y_test)) / n;
  # REQM
  REQM = sqrt(sum((y_test - y_pred)^2) / n);
  # ERA
  ERA = sum(abs(y_test - y_pred)) / sum(abs(y_test - mean(y_test)));
  # EQR
  EQR = sqrt(sum((y_test - y_pred)^2) / sum((y_test - mean(y_test))^2));
  # r
  Sup_ = (n * sum(y_test*y_pred)) - (sum(y_test) * sum(y_pred));
  Inf_ = sqrt(((n * sum(y_test^2)) - (sum(y_test)^2)) * ((n * sum(y_pred^2)) - (sum(y_pred)^2)));
  r = Sup_ / Inf_;
  # R2
  R2 = 1 - (sum((y_test - y_pred)^2) / sum((y_test - mean(y_test))^2)); 
  err = data.frame(EMA, REQM, ERA, EQR, r, R2)
  
  return(err)
}

Treinamento <- function(model){
  error = data.frame(matrix(ncol = 6, nrow = 5))
  kfold = 1:5
  for (i in kfold) {
    cv_train <- paste("CV_tr",toString(i), sep="_")
    cv_train <- data.frame(eval(parse(text = cv_train)))
    cv_test <- paste("CV_ts",toString(i), sep="_")
    cv_test <- data.frame(eval(parse(text = cv_test)))
    if(model == "R_Linear"){# Regressão linear
      mod = lm(PE ~ AT + V + AP + RH, data=cv_train)
      y_pred = predict(mod, cv_test[,1:4])
      error[i,] <- Error(y_pred, cv_test[,5:5])
      cat("Modelo ", i, "\n")
    } else if(model == "M5P") {# Árvore modelo M5P
      conf = c("Yes","No")
      TG <- expand.grid(pruned = conf, smoothed = conf, rules = conf)
      mod = caret::train(x = cv_train[,1:4], y = cv_train[,5:5], method = "M5",
                         metric = "Rsquared", tuneGrid = TG) 
      y_pred = predict(mod, cv_test[,1:4])
      error[i,] <- Error(y_pred, cv_test[,5:5])
      cat("Modelo ", i, "\n")
    } else if(model == "MVS") {# Máquina de vetor de suporte
      TG <- expand.grid(cost= 10)#seq(10,90, by =30)
      mod = caret::train(x = cv_train[,1:4], y = cv_train[,5:5], method = 'svmLinear2',
                         metric = "Rsquared", tuneGrid = TG) 
      y_pred = predict(mod, cv_test[,1:4])
      error[i,] <- Error(y_pred, cv_test[,5:5])
      cat("Modelo ", i, "\n")
      
    } else {# Multilayer Perceptron
      TG <- expand.grid(size = 25, decay = c(0.5,0.1))#seq(7, 25, 2)
      mod = caret::train(x = cv_train[,1:4], y = cv_train[,5:5], method = 'mlpWeightDecay',
                         metric = "Rsquared", tuneGrid = TG) 
      y_pred = predict(mod, cv_test[,1:4])
      error[i,] <- Error(y_pred, cv_test[,5:5])
      cat("Modelo ", i, "\n")
    }
    #error[i,] <- Error(y_pred, cv_test[,5:5])
    #cat("Modelo ", i, "\n")
  }
  colnames(error) <- c('EMA', 'REQM', 'ERA', 'EQR', 'r', 'R2')
  print(error)
  FileName = paste("Error",toString(model), sep="_")
  write.csv(error, file=FileName)
  #return()
} 

Treinamento(model = "R_Linear")
Treinamento(model = "M5P")
Treinamento(model = "MVS")
Treinamento(model = "MLP")



