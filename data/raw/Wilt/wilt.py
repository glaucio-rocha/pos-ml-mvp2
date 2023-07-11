# Classification of the Wilt dataset using neural networks
import numpy
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from keras.utils import np_utils

numpy.set_printoptions(threshold='nan')

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)


# load wilt dataset
dataset = numpy.loadtxt("training.csv", delimiter=",")
print(dataset.shape)

# split into input (X) and output (Y) variables
X = dataset[:,1:6]
Y = dataset[:,0]

# create model
model = Sequential()
model.add(Dense(12, input_dim=5, init='uniform', activation='relu'))
model.add(Dense(5, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))

# Compile model
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X, Y, nb_epoch=100, batch_size=10,  verbose=2)

print "----------------------------------------------"
dataset1 = numpy.loadtxt("testing.csv", delimiter=",")
print(dataset1.shape)


# split into input (X) and output (Y) variables
X_test = dataset[:,1:6]
Y_test = dataset[:,0]

#evaluate
score = model.evaluate(X_test, Y_test, verbose=0)
print score
print model.metrics_names
print "----------------------------------------------"


# calculate predictions and compare for manual verification
predictions = model.predict(X_test)
fp=open("pred.txt","w")
# round off predictions
for x,y in zip(predictions,Y_test):
	x=round(x,5)
	y=round(y,0)
	fp.write(str(x)+"  --  "+str(y)+"\n")
fp.close()




        












