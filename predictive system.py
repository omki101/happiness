import numpy as np
import pickle 



loaded_model=pickle.load(open('C:/Users/Omkar/OneDrive/Documents/Happiness INdex/trained model.sav','rb'))

input_data=(12,0.9,70,0.9)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction=loaded_model.predict(input_data_reshaped)
print(prediction)
if(prediction<5):
  print("You are not that happy")
else:
  print("You are the quite happy")









