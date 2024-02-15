
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

'''
def test_negative_prediction():
    response = client.post('/v1/predictions', json= {
        'format': 1,
        'comments': -10,
        'likes': -5,
        'saved': -1,
        'shares': -1
    })
    assert response.status_code == 200
    assert response.json()['reach'] <= 0
'''

def test_random_prediction():
    response = client.post('/v1/predictions', json= {
        'format': 2,
        'comments': 40,
        'likes': 300,
        'saved': 200,
        'shares': 59
    })
    assert response.status_code == 200
    assert response.json()['reach'] >= 0
