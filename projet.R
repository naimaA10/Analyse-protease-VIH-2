install.packages("gplots")
install.packages("ggplot2")
if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager")
BiocManager::install("limma")
BiocManager::install("edgeR")
install.packages("caret")
install.packages("corrplot")
library(ggplot2)
library(lattice)
library(caret)
library(corrplot)

#-------------------------------------------------
#Etape1: Préparation du jeu de données
#-------------------------------------------------

#ouverture du jeux de données, description des variable

dtf <- read.table("matrice_descripteurs.dat",sep=';',header=T)
str(dtf)#201 obs et 97 variables, variable quantitative X10_A_CA.15_A_CA  et X10_B_CA.15_B_CA "logi" ont que NA
par(mfrow=c(1,1),mar=c(9,3,0,0))
boxplot(dtf,las=2)

attributes(dtf)
#supprimé NA, valeur abérante, variance null, variables corrélation à plus de 0.9
col.supp <- which(colnames(dtf)=="X10_A_CA.15_A_CA") 
dtf <- dtf[,-col.supp]
col.supp <- which(colnames(dtf)=="X10_B_CA.15_B_CA") 
dtf <- dtf[,-col.supp]#On a supp les 2 va qui ont que des NA, il reste 201 obs et 95 va

dtf <- na.omit(dtf) #201 obs et 95 variables aucune obs et aucune va supp

var <- apply( dtf ,MARGIN = 2,FUN = var)
dtf <- dtf[,-(which(var<0.1))] #supp va  avec variance proche 0 il reste 201 obs et 28 va 

colnames(dtf)
par(mfrow=c(1,1),mar=c(0, 0, 1, 1))
mat.cor = cor(dtf)
corrplot(mat.cor, method="circle")
#Supp les va avec une forte corrélation 
varSup = findCorrelation(mat.cor, cutoff=0.9)
dtf <- dtf[, -varSup]

#représentation des variables en boxplot
par(mfrow=c(1,1),mar=c(10, 3, 1, 1))
boxplot(dtf,las=2,col=2)#tt les va proche de 0 sauf 3 va C_ATOM,SURFACE_HULL et VOLUME_HULL
par(mar=c(0, 2,2, 1), mfrow=c(6,5))
for(i in 1:(ncol(dtf))){
  boxplot(dtf[,i],
          main=c(colnames(dtf)[i]),
          col=2)
}
#moy proche -1 pour charge avec valeur abbérant à 0 et 1
#valeur abbérante pour X50_A_CA.50_B_CA et X50_A_CA.25_A_CA ..qui sont continue

#centrer et réduire matrice (as.data.frame)
matDesc<- scale(dtf,center=T,scale=T)

#corrélation entre variable
matCor <- cor(matDesc)
par(mfrow=c(1,1),mar=c(0, 3, 1, 1))
corrplot(matCor,method="circle")

colnames(dtf)

#-------------------------------------------------
#Etape2 : Création des jeux de données d'apprentissage et test
#-------------------------------------------------

#création des vect contenant les numéros d'individus des échantillons
#d'apprentissage et test
vIndApp <- sample(1:nrow(matDesc), size = round(2*nrow(matDesc)/3,0))
length(vIndApp)#134

vIndTest <- (1:nrow(matDesc))[-vIndApp]
length(vIndTest)#67

#création des matrices contenant les valeurs des descripteurs pour les individus
#des échantillons d'apprentissage et test
matApp <- as.data.frame(matDesc[vIndApp,])
dim(matApp)#134  28

matTest <-as.data.frame(matDesc[vIndTest,])
dim(matTest)#67 28

#-------------------------------------------------
#Etape3 : Validation des échantillons d'apprentissage et de test
#-------------------------------------------------

par(mar=c(2, 2, 1, 0), mfrow=c(6,5))
for(i in 1:ncol(matApp)){
  hist(matApp[,i], main=colnames(matApp)[i], xlab=colnames(matApp)[i])
}
par(mfrow=c(6,5), mar=c(2, 2, 1, 1))
for(i in 1:dim(matTest)[2]){
  hist(matTest[,i], main=colnames(matTest)[i], xlab=colnames(matTest)[i])
}
colnames(dtf)
dtf <- dtf[,-2]
#-------------------------------------------------
#Etape4 : Apprentissage du modèle complet
#-------------------------------------------------
fit <- lm(asymPos.nbr~., data=as.data.frame(matApp))
summary(fit)


#-------------------------------------------------
#Etape5 : Etude des performances du modèle sur l'échantillon
#d'apprentissage
#-------------------------------------------------
#vect avec valeur obs de asymPos.nbr
yobsApp <- matApp[,"asymPos.nbr"]

#vect avec valeur prédit de asymPos.nbr
ypredApp <- predict(fit, newdata = as.data.frame(matApp))

#valeurs prédtite en fonction valeur obs dans ech app
par(mfrow=c(1,1),mar=c(5,4,1,1))
plot(yobsApp,ypredApp, xlab="observed asymPos.nbr", ylab="predicted asymPos.nbr", main=" Training set", pch=16)

#coeff corrélation entre va obs et prédite sur ech app
round(cor(yobsApp,ypredApp), 2)

#coeff de détermination
round(summary(fit)$r.squared,2)
round(summary(fit)$adj.r.squared,2)

#RMSEP
RMSEPApp <- sqrt(sum((yobsApp-ypredApp)^2)/nrow(matApp))
print(round(RMSEPApp,2))


#-------------------------------------------------
#Etape 6 : Etude des performances du modèle sur l'échantillon test
#-------------------------------------------------
#coeff corrélation sur jeu test
yobsTest <- matTest[,"asymPos.nbr"]
ypredTest <- predict(fit, newdata = as.data.frame(matTest))
plot(yobsTest,ypredTest, xlab="Obs asymPos.nbr", ylab="Pred asymPos.nbr", main="Set Test", pch=16)

round(cor(yobsTest,ypredTest),2)

#coeff détermination prédit
dem = sum((yobsTest - mean(yobsApp))^2)
num = sum(( yobsTest - ypredTest)^2)
r2Pred = 1-(num/dem)
print(round(r2Pred,2))

#RMSEP
RMSEPTest <- sqrt(sum((yobsTest-ypredTest)^2)/nrow(matTest))
print(round(RMSEPTest,2))

#-------------------------------------------------
#Etape 7 : Etude résidus du modèle
#-------------------------------------------------
#analyse distribution résidus
residus = fit$residuals
hist(residus, xlab="residues",prob=T, br=20)
newx = seq(min(residus), max(residus),by=0.01)
lines(newx, dnorm(x=newx, mean=mean(residus), sd = sd(residus)), col = 2)

#calcul moy des résidu
theta <- mean(residus)
print(theta)

#analyse l'homoscédasticité et indp des résidus 
par(mfrow=c(1,1),mar=c(5,4,3,1))
plot(ypredApp,residus,main="residus en fonction des valeurs prédites de asymPos.nbr", ylab="residus", xlab="predited values of asymPos.nbr",
     pch=19)
abline(h=0,col=2)
summary(fit)
#-------------------------------------------------
#Etape 8 : Sélection des variables les plus significatives et analyse
#d'un nouveau modèle
#-------------------------------------------------
#selection des va
matDescSelApp <- data.frame(asymPos.nbr=matApp[,"asymPos.nbr"],X62_A_CA.66_A_CA=matApp[,"X62_A_CA.66_A_CA"],DIAMETER_HULL=matApp[,"DIAMETER_HULL"])
matDescSelTest <- data.frame(asymPos.nbr=matApp[,"asymPos.nbr"],X62_A_CA.66_A_CA =matTest[,"X62_A_CA.66_A_CA"],DIAMETER_HULL=matTest[,"DIAMETER_HULL"])
#calcul du modèle
fit2 <- lm(asymPos.nbr~., data = matDescSelApp)
#coeff du modèle
summary(fit2)
#calcul des performances du modèles
yobsApp2 <- matApp[,"asymPos.nbr"]
ypredApp2 <- predict(fit2, newdata = matApp)
yobsTest2 <- matTest[,"asymPos.nbr"]
ypredTest2 <- predict(fit2, newdata = matTest)

par(mfrow=c(1,2))
plot(yobsApp2,ypredApp2, xlab="observed asymPos.nbr", ylab="predicted asymPos.nbr", main=" Training set", pch=16)
plot(yobsTest2,ypredTest2, xlab="observed asymPos.nbr", ylab="predicted asymPos.nbr", main=" Test set", pch=16)

round(cor(yobsApp2,ypredApp2),2)

round(cor(yobsTest2,ypredTest2),2)

#étude des résidus
residus3 = fit2$residuals

par(mfrow=c(1,1))
hist(residus3, xlab="residues",prob=T, br=20)
newx = seq(min(residus3), max(residus3),by=0.01)
lines(newx, dnorm(x=newx, mean=mean(residus3), sd = sd(residus3)), col = 2)

#calcul de la moy des résidus
theta <- mean(residus3)
print(theta)
#analyse homoscédasticité et indep des résidus 
plot(ypredApp2,residus3, ylab="residus", xlab="predited values of asymPos.nbr",
     main="residus en fonction des valeurs prédites de asymPos.nbr",
     pch=19)
abline(h=0,col=2)
