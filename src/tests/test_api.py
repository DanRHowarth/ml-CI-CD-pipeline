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

predict_one = {}

client = TestClient(app)


def test_say_hello():
    request = client.get("/")
    assert request.status_code == 200


def test_create_item():
    request = client.post("/prediction/", json=predict_zero)
    # post a body
    assert request.status_code == 200
    assert request.contet == '{"prediction:": 0}'

    request = client.post("/prediction/", json=predict_one)
    # post a body
    assert request.status_code == 200
    assert request.contet == '{"prediction:": 1}'
