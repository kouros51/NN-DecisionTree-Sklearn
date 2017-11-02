# Load the data-set "zoo"
zoo=read.csv(file = "zoo.csv")

# Transform all atrribute to factors one to run the Decision tree algorithm
for(i in 1:ncol(zoo)){
  zoo[,i]=as.factor(zoo[,i])
}

# Split Into train and test sets
sampl=sample(1:101,0.25*100)
zootrain=zoo[-sampl,]
zootest=zoo[sampl,]

# Train the model
DT50Zoo= C50::C5.0(x=zootrain[,2:17],y=zootrain[,18])

# Predict and show results
estType=predict(DT50Zoo,zootest,type="class")
table(estType,zootest$type)

# Calcul taux d'erreur
1-((8+6+0+4+1+0+4)/25)

summary(DT50Zoo)

