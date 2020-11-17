from tensorflow.keras import models, layers, losses, activations, optimizers, callbacks
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np

class PlotPoints(callbacks.Callback):
    def __init__(self):
        super().__init__()
        df = pd.read_csv("test.csv", header=None)
        x = df.iloc[:,0]
        y = abs(df.iloc[:,1])
        self.output = packTuple(x,y)
        self.x = x
        self.y = y
        self.N = -1
        plt.ion()
        plt.show()
        self.fig = plt.figure()
        self.ax = plt.axes(projection="3d")
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_zlabel('z')
        z = [0 for i in range(len(x))]
        z[0] = -10
        z[1] = 10
        self.scatter = self.ax.scatter(x,y,z,s=1)

    def on_epoch_end(self, epoch, logs=None):
        self.N += 1
        if self.N % 10 == 0:
            results = np.array(model.predict(np.array(self.output)))
            results = results.reshape(len(results))
            self.scatter._offsets3d = (self.x, self.y, results)
            self.fig.canvas.draw()
            plt.pause(0.001)

# pack tuple to be inpt into NN (,2)
def packTuple(x,y):
    result = []
    for index, item in enumerate(x):
        result.append([item,y[index]])
    return result

# reads train data and splits it into x,y,z points
df = pd.read_csv("train.csv")
x = df.iloc[:,0]
y = abs(df.iloc[:,1])
z = df.iloc[:,2]
inpt = packTuple(x,y)

# simple NN with adam optimizer and MSE loss metric
model = models.Sequential()
model.add(layers.Dense(16, activation='tanh', input_shape=(2,)))
model.add(layers.Dense(64, activation='tanh'))
model.add(layers.Dense(128, activation='tanh'))
model.add(layers.Dense(64, activation='tanh'))
model.add(layers.Dense(16, activation='tanh'))    
model.add(layers.Dense(1, activation='exponential'))
model.add(layers.Lambda(lambda x: x * 20 - 10))

optim = optimizers.Adam(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    amsgrad=False,
    name="Adam"
)

callback = callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0,
    patience=100,
    verbose=1,
    mode="min",
    baseline=None,
    restore_best_weights=True,
)

savemodel = callbacks.ModelCheckpoint('weights.h5', 
                                     save_weights_only=True, period=10)

plotpoints = PlotPoints()

model.compile(optimizer=optim,
                loss=losses.MeanSquaredError(),
                metrics=['mean_squared_error'])

model.load_weights('weights.h5')
print(model.optimizer)

### Train the modeal to be decent first.
##history = model.fit(np.array(inpt),np.array(z), epochs=300, verbose=1, callbacks=[savemodel, plotpoints])
# Now slower learning rate.
K.set_value(model.optimizer.learning_rate, 1e-4)
history = model.fit(np.array(inpt),np.array(z), epochs=10000, verbose=1, validation_split=0.2, callbacks=[callback, savemodel, plotpoints], shuffle=True)

# for outputting predictions of model
df = pd.read_csv("test.csv", header=None)
x = df.iloc[:,0]
y = abs(df.iloc[:,1])
output = packTuple(x,y)

results = model.predict(np.array(output))
np.savetxt("test_results.csv", np.c_[results], delimiter=",")

# Display predictions
fig = plt.figure()
ax = plt.axes(projection="3d")
ax.scatter(x,y,results,s=1)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

plt.show()
