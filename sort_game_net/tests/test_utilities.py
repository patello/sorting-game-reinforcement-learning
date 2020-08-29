import os
import pytest

from ..utilities import run_batch, train

file_path = os.path.dirname(os.path.realpath(__file__))

@pytest.mark.parametrize("test_data",[
    {"model_name":"integer_encoded.pth","state_fun":"int"},
    {"model_name":"binary_encoded.pth","state_fun":"encoded"},
    ])
class TestUtilities:
    def test_run_batch(self,test_data):
        run_batch(model_path=file_path+"/resources/"+test_data["model_name"],batches=5)

    def test_train(self,test_data):
        train(state_fun=test_data["state_fun"],batch_size=2,batches=2)