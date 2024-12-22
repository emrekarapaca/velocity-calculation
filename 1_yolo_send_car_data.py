import requests

url = 'http://192.168.2.222:8000/car_pass'

# 3 citypoint için 3 velocity gönderelim
for velocity in range(3):
    for citypoint in range(3):
        car_data = {'citypoint':citypoint, 'velocity': velocity}
        response = requests.post(url, json = car_data)
        print(response.text)
