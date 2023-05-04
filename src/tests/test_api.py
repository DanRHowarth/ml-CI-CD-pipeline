from fastapi.testclient import TestClient
from main import app

# provide data for prediction test
predict_zero = {'age': 38,
                'workclass': 'Private',
                'fnlgt': 215646,
                'education': 'HS-grad',
                'education-num': 9,
                'marital-status': 'Divorced',
                'occupation': 'Handlers-cleaners',
                'relationship': 'Not-in-family',
                'race': 'White',
                'sex': 'Male',
                'capital-gain': 0,
                'capital-loss': 0,
                'hours-per-week': 40,
                'native-country': 'United-States'}

predict_one = {'age': 52,
               'workclass': 'Self-emp-not-inc',
               'fnlgt': 209642,
               'education': 'HS-grad',
               'education-num': 9,
               'marital-status': 'Married-civ-spouse',
               'occupation': 'Exec-managerial',
               'relationship': 'Husband',
               'race': 'White',
               'sex': 'Male',
               'capital-gain': 0,
               'capital-loss': 0,
               'hours-per-week': 45,
               'native-country': 'United-States'}

client = TestClient(app)

def test_say_hello():
    request = client.get("/")
    assert request.status_code == 200, f'Status code {request.status_code} returned instead of 200'


def test_create_item():
    request = client.post("/prediction/", json=predict_zero)
    assert request.status_code == 200, f'Status code {request.status_code} returned instead of 200'
    assert request.content == b'{"prediction:":0}', f'{request.content} returned instead of 0'

    request = client.post("/prediction/", json=predict_one)
    assert request.status_code == 200, f'Status code {request.status_code} returned instead of 200'
    assert request.content == b'{"prediction:": 1}', f'{request.content} returned instead of 1'
