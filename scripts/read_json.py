



pathJson = '/home/wen/JI_ST-cloudy-day_2019-08-27-21-55-47/set_1/radar_obstacles/radar_obstacles.json'
with open(pathJson, "r") as json_file:
    dataR = json.load(json_file)
    dataR = dataR.get('radar')