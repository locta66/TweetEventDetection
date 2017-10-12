import __init__
import numpy as np
from EventClassifier import EventClassifier

dim=100
samplenum=20

classifier = EventClassifier(vocab_size=dim, learning_rate=0)
idfmtx = np.random.random([samplenum,dim])
print(classifier.predict(idfmtx))

print([classifier.predict([idfmtx[i]])[0] for i in range(samplenum) if classifier.predict([idfmtx[i]])[0]>0.5])


