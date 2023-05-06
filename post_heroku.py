import requests

url = 'https://mldevops-project-3.herokuapp.com/prediction/'

body = {'age': 38,
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

response = requests.post(url, json=body)

assert response.status_code, f'invalid status of {response.status_code}'

print('Response is: ', response.content)
print('Status Code is: ', response.status_code)
