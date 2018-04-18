import numpy as np
import pandas as pd
from tgym.core import DataGenerator
#from tgym.gens.csvstream import CSVStreamer
#a = np.array([47.18,1,1], dtype=np.float)
#b = np.array([2,2,2], dtype=np.float)

#c=np.hstack((a,b)) # 合併

#print(c)

#generator1 = CSVStreamer(filename='./test_6.csv')
#generator2 = CSVStreamer(filename='./test_5.csv')

#c = np.hstack((generator2.next(),generator1.next()))
#print(c)

class get_CSV_data(DataGenerator):

    def _generator(self, filename):
        f = open(filename)
        df = pd.read_csv(f)
        data = df.iloc[:,:].values
        
        data_train=data[:]
        train_mean=np.mean(data_train,axis=0)
        train_std=np.std(data_train,axis=0)
        normalized_train_data=(data_train-train_mean)/train_std

        for i in range(len(normalized_train_data)):
            
            #try:
            yield  normalized_train_data[i,:]
                #print(np.array(normalized_train_data[i,:], dtype=np.float))
            #except StopIteration:
                #print("done")

			
    def _iterator_end(self):
        """Rewinds if end of data reached.
        """
        #print "End of data reached, rewinding."
        super(self.__class__, self).rewind()

    def rewind(self):
        """For this generator, we want to rewind only when the end of the data is reached.
        """
        pass