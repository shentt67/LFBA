import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np

from dataset.utils import get_labeled_data


class NusOrderTests(unittest.TestCase):
    def test_feature_order_does_not_depend_on_directory_enumeration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for sub in ['Groundtruth/TrainTestLabels','Low_Level_Features','NUS_WID_Tags']:
                (root/sub).mkdir(parents=True)
            np.savetxt(root/'Groundtruth/TrainTestLabels/Labels_a_Train.txt',[1,0],fmt='%d')
            np.savetxt(root/'Groundtruth/TrainTestLabels/Labels_b_Train.txt',[0,1],fmt='%d')
            names=['Train_Normalized_a.dat','Train_Normalized_b.dat']
            for i,name in enumerate(names):
                np.savetxt(root/'Low_Level_Features'/name,np.full((2,2),i+1))
            np.savetxt(root/'NUS_WID_Tags/Train_Tags1k.dat',np.ones((2,3)),delimiter='\t')
            with patch('dataset.utils.os.listdir',return_value=names):
                a=get_labeled_data(tmp,['a','b'],None)[0]
            with patch('dataset.utils.os.listdir',return_value=names[::-1]):
                b=get_labeled_data(tmp,['a','b'],None)[0]
            np.testing.assert_array_equal(a,b)
