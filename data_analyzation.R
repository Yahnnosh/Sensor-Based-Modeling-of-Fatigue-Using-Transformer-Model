# Interaction fatigue <-> physiological data
# 1. Read data
path = "C:/Users/jjung/iCloudDrive/ETH/MSc 3rd semester/Semester project/Output/"
path.macOS = "/Users/janoschjungo/Library/Mobile Documents/com~apple~CloudDocs/ETH/MSc 3rd semester/Semester project/Output/"

data.mean = read.csv(paste(path.macOS, "combined_data_mean.csv", sep=""), header=TRUE, sep=",")
str(data.mean)
n = nrow(data.mean)
p = 12
variables = names(dat)[1:p]

# factor variables
data.mean$VAS = as.factor(data.mean$VAS)
levels(data.mean$VAS)
levels(data.mean$VAS) = c("vigilant", "fatigued")

data.mean$MF = as.factor(data.mean$MF)
levels(data.mean$MF)
levels(data.mean$MF) = c("vigilant", "fatigued")

data.mean$phF = as.factor(data.mean$phF)
levels(data.mean$phF)
levels(data.mean$phF) = c("vigilant", "fatigued")

data.mean$ReIP = as.factor(data.mean$ReIP)
levels(data.mean$ReIP)
levels(data.mean$ReIP) = c("worse", "same", "better")

# TODO: mean of ActivitiyClass (a factor var.) is meaningless -> change! (to most common value)
unique(data.mean$ActivityClass)
data.mean$ActivityClass = as.factor(data.mean$ActivityClass)
levels(data.mean$ActivityClass)
levels(data.mean$ActivityClass) = c("undefined", "resting", "other", "biking", "running", "walking")

# remove metadata
dat = subset(data.mean, select=c(-X, -subjectID, -date, -sport, -n_answers, -timezone))
# TODO: include ActivityClass
p = 11
variables = names(dat)[1:p]
dat = subset(dat, select=-ActivityClass)
str(dat)

# 2. Statistical Analysis
library(ggplot2)
library(GGally)
library(reshape)
library(plot.matrix)
melted = melt(dat)

# boxplot for relationship: VAS <-> physiological variables
ggplot(data=melted, aes(x=variable, y=value)) + 
  geom_boxplot(aes(fill=VAS)) +
  facet_wrap(~variable, scales="free")
# boxplot for relationship: phF <-> physiological variables
ggplot(data=melted, aes(x=variable, y=value)) + 
  geom_boxplot(aes(fill=phF)) +
  facet_wrap(~variable, scales="free")
# boxplot for relationship: MF <-> physiological variables
ggplot(data=melted, aes(x=variable, y=value)) + 
  geom_boxplot(aes(fill=MF)) +
  facet_wrap(~variable, scales="free")
# boxplot for relationship: ReIP <-> physiological variables
ggplot(data=melted, aes(x=variable, y=value)) + 
  geom_boxplot(aes(fill=ReIP)) +
  facet_wrap(~variable, scales="free")

# t-test
t.tests = data.frame(matrix(nrow=3, ncol=p), 
                     row.names=c("VAS", "phF", "MF"))
colnames(t.tests)=variables

for (fatigue in 12:14) {
  p.values = numeric(p)
  for (col in 1:p) {
    p.values[col] = (t.test(dat[, col]~dat[, fatigue]))$p.value
  }
  t.tests[fatigue-11, ] = p.values
}
t.tests

significance.level = 0.01
plot(as.matrix(t.tests < significance.level), las=2, cex.axis=0.7, col=c("white", "black"))

significance.level = 0.05
plot(as.matrix(t.tests < significance.level), las=2, cex.axis=0.7, col=c("white", "black"))

# pairs plot
pairs(dat)

# covariance
covariance.matrix = cov(na.omit(subset(dat, select=variables)))
library(plot.matrix)
for (i in 1:ncol(covariance.matrix)) { # set upper triangle to NaN
  for (j in i:ncol(covariance.matrix)) {
    covariance.matrix[i, j] = NA
  }
}
plot(covariance.matrix, las=2, cex.axis=0.7, breaks=6, digits=4, text.cell=list(cex=0.5)) # different than in suppl. material???

# histogram
ggplot(data=melted, aes(x=value)) + 
  geom_histogram(position="identity") + 
  facet_wrap(~variable, scales="free")
