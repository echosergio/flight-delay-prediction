# Import libraries

libraries <- c("FactoMineR","factoextra")
lapply(libraries, require, character.only = TRUE)

# Load file

file=file.path("2008.csv")
data <- read.csv(file, header = TRUE, sep = ",", quote = "\"", dec = ".", fill = TRUE)

# Remove forbidden variables, not quantitative and null values

forbiddenVariables <- c("ArrTime","ActualElapsedTime","AirTime","TaxiIn","Diverted","CarrierDelay","WeatherDelay","NASDelay","SecurityDelay","LateAircraftDelay")
data <- data[, !(colnames(data) %in% forbiddenVariables)]

notQuantitativeVariables <- c("UniqueCarrier","TailNum","Origin","Dest","CancellationCode")
data <- data[, !(colnames(data) %in% notQuantitativeVariables)]

data=na.omit(data)

# data$date <- as.Date(with(data, paste(Year, Month, DayofMonth, sep="-")), "%Y-%m-%d")

# Linear Regression Analysis

ggplot(data, aes(x=DepDelay, y=ArrDelay)) + geom_point() + geom_smooth(method=lm)
cor(data$DepDelay, data$ArrDelay)

# PCA Analysis

data_pca_r=PCA(data,scale.unit=TRUE, graph=FALSE)
summary(data_pca_r)
fviz_screeplot(data_pca_r)

data_pca_r$var$contrib

fviz_contrib(data_pca_r, choice = "var", axes = 1)
fviz_contrib(data_pca_r, choice = "var", axes = 2)

data_pca_r$var$coord

fviz_pca_var(data_pca_r, col.var="cos2") + scale_color_gradient2(low="white", mid="blue", high="red", midpoint=0.5) + theme_minimal()