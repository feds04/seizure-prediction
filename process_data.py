import getfeatures as gf
import random
from sklearn.linear_model import LogisticRegression
import scipy.io as sio
f1=gf.get_features("Dog_1")
f1test=gf.get_test_features("Dog_1")
f2=gf.get_features("Dog_2")
f2test=gf.get_test_features("Dog_2")
f3=gf.get_features("Dog_3")
f3test=gf.get_test_features("Dog_3")
f4=gf.get_features("Dog_4")
f4test=gf.get_test_features("Dog_4")
f5=gf.get_features("Dog_5")
f5test=gf.get_test_features("Dog_5")

sio.savemat("features", {"f1":f1,"f2":f2,"f3":f3,"f4":f4, "f5":f5,"f1test":f1test,"f2test":f2test,"f3test":f3test,"f4test":f4test, "f5test":f5test}, appendmat=True, format='5', long_field_names=False, do_compression=False, oned_as='row')