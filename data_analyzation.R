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


# intra-subject variability
dat2 = subset(data.mean, select=c(-X, -date, -sport, -n_answers, -timezone, -ActivityClass))
dat2$subjectID = as.factor(dat2$subjectID)
melted2 = melt(dat2)

ggplot(data=melted2, aes(x=variable, y=value)) + 
  geom_boxplot(aes(fill=subjectID)) +
  facet_wrap(~variable, scales="free")


# questionnaires per subject
y = numeric(27)
for (i in 1:27) {
  y[i] = sum(dat2$subjectID == i)
}
x = 1:27

table(data.mean$subjectID) # treats as categorical data with multiple draws
barplot(table(data.mean$subjectID),
        main="days with filled out questionnaires",
        xlab="subject", 
        ylab="days")

sort(table(data.mean$subjectID))[23:27] # top 4 “contributors“
(sum(sort(table(data.mean$subjectID))[23:27]) / n) * 100 # 4/27 give data for 65%


# boxplots
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
colnames(t.tests) = variables

for (fatigue in 12:14) {
  p.values = numeric(p)
  for (col in 1:p) {
    p.values[col] = (t.test(dat[, col]~dat[, fatigue]))$p.value
  }
  t.tests[fatigue-11, ] = p.values
}
t.tests

# a) significance = 0.01
significance.level = 0.01
plot(as.matrix(t.tests < significance.level), las=2, cex.axis=0.7, col=c("white", "black"))

response = matrix(nrow=3, ncol=p)
colnames(response) = variables
rownames(response) = c("VAS", "phF", "MF")

significance = t.tests < significance.level
fatigue = c("VAS", "phF", "MF")
for (row in 1:3) {
  for (col in 1:11) {
    label = fatigue[row]
    var = variables[col]
    
    selection.fatigue = dat[, label] == "fatigued"
    selection.vigilant = dat[, label] == "vigilant"
    
    var.fatigue = na.omit(dat[selection.fatigue, var])
    var.vigilant = na.omit(dat[selection.vigilant, var])
    
    mean.fatigue = mean(var.fatigue)
    mean.vigilant = mean(var.vigilant)
    
    # only if significant
    if (significance[row, col]) {
      response[row, col] = ifelse(mean.fatigue > mean.vigilant,
                                  "↑", "↓")
    }
    else {
      response[row, col] = NA
    }
  }
}
response

plot(response, las=2, cex.axis=0.7, col=c("green", "red"), 
     main="physiological response to fatigue")

# b) significance = 0.05
significance.level = 0.05
plot(as.matrix(t.tests < significance.level), las=2, cex.axis=0.7, col=c("white", "black"))


response = matrix(nrow=3, ncol=p)
colnames(response) = variables
rownames(response) = c("VAS", "phF", "MF")

significance = t.tests < significance.level
fatigue = c("VAS", "phF", "MF")
for (row in 1:3) {
  for (col in 1:11) {
    label = fatigue[row]
    var = variables[col]
    
    selection.fatigue = dat[, label] == "fatigued"
    selection.vigilant = dat[, label] == "vigilant"
    
    var.fatigue = na.omit(dat[selection.fatigue, var])
    var.vigilant = na.omit(dat[selection.vigilant, var])
    
    mean.fatigue = mean(var.fatigue)
    mean.vigilant = mean(var.vigilant)
    
    # only if significant
    if (significance[row, col]) {
      response[row, col] = ifelse(mean.fatigue > mean.vigilant,
                                  "↑", "↓")
    }
    else {
      response[row, col] = NA
    }
  }
}
response

plot(response, las=2, cex.axis=0.7, col=c("green", "red"), 
     main="physiological response to fatigue")


# pairs plot
pairs(dat)

pairs(subset(dat, select=variables), cex.labels=1.2)


# covariance
covariance.matrix = cov(na.omit(subset(dat, select=variables)))
for (i in 1:ncol(covariance.matrix)) { # set upper triangle to NaN
  for (j in i:ncol(covariance.matrix)) {
    covariance.matrix[i, j] = NA
  }
}
plot(covariance.matrix, las=2, cex.axis=0.7, breaks=6, digits=4, text.cell=list(cex=0.5)) # different than in suppl. material???


# correlation
correlation.matrix = cor(na.omit(subset(dat, select=variables)))
for (i in 1:ncol(correlation.matrix)) { # set upper triangle to NaN
  for (j in i:ncol(correlation.matrix)) {
    correlation.matrix[i, j] = NA
  }
}
plot(correlation.matrix, las=2, cex.axis=0.7, breaks=6, digits=4, text.cell=list(cex=1.5)) # different than in suppl. material???
plot(abs(correlation.matrix), las=2, cex.axis=0.7, breaks=6, digits=4, text.cell=list(cex=1.5)) # different than in suppl. material???


# histogram
ggplot(data=melted, aes(x=value)) + 
  geom_histogram(position="identity") + 
  facet_wrap(~variable, scales="free")
