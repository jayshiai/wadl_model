from flask import Flask, request, jsonify
import torch
import json
import os
import pandas as pd
app = Flask(__name__)


# Enable CORS for all routes
@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'POST'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response


def get_predictions(data):
    model = torch.load('./chkpt/wd.pth').to('cuda')
    model.eval()
    # Make predictions
    with torch.no_grad():
        predictions = model(data)
    
    return predictions.tolist()

@app.route('/home')
def home():
    return "Welcome to Home Page"

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        data =  request.get_json()
        print(data)
        sample_data = torch.tensor([data['list']]).to('cuda')
        predictions = get_predictions(sample_data)
        return  predictions
    else:
        return "Wrong Method."


@app.route('/data', methods=['POST'])
def data():
    if request.method == 'POST':
        data =  request.get_json()
        json_file_path = './data/data.json'
        print(data)
        existing_data = []
        if os.path.exists(json_file_path):
            with open(json_file_path, 'r') as json_file:
                existing_data = json.load(json_file)
        # Append new data to the existing list
        existing_data.append(data)

        # Write the updated data back to the file
        with open(json_file_path, 'w') as json_file:
            json.dump(existing_data, json_file)
        
        print("Data saved successfully.")
        return jsonify({"message": "Data stored successfully"})
    else:
        return "Wrong Method."


@app.route('/events', methods=['POST'])
def events():
    if request.method == 'POST':
        data =  request.get_json()
        json_file_path = './data/events.json'

        existing_data = []
        if os.path.exists(json_file_path):
            with open(json_file_path, 'r') as json_file:
                existing_data = json.load(json_file)
        # Append new data to the existing list
        existing_data.append(data)

        # Write the updated data back to the file
        with open(json_file_path, 'w') as json_file:
            json.dump(existing_data, json_file)
        
        print("Data saved successfully.")
        return jsonify({"message": "Data stored successfully"})
    else:
        return "Wrong Method."


@app.route('/events_get', methods=['GET'])
def events_get():
    if request.method == 'GET':
        json_file_path = './data/events.json'

        if os.path.exists(json_file_path):
            with open(json_file_path, 'r') as json_file:
                events_data = json.load(json_file)
            return jsonify(events_data)
        else:
            return jsonify({"error": "File not found"}), 404
    else:
        return "Wrong Method."



@app.route('/data_get', methods=['GET'])
def data_get():
    if request.method == 'GET':
        json_file_path = './data/data.json'

        if os.path.exists(json_file_path):
            # Read JSON data into a Pandas DataFrame
            df = pd.read_json(json_file_path)

            # Manipulate DataFrame as per your requirements
            df['pageTime'] = df['sessionEnd'] - df['sessionStart']

            max_session_end = df.groupby(['userID', 'sessionID'])['sessionEnd'].transform('max')
            df['sessionTime'] = max_session_end - df['sessionID']
            df['pageTime'] = df['pageTime'] / 1000
           
            print(df['osName'])
            # Convert DataFrame to JSON
            events_data = df.to_json(orient='records')

            return jsonify(json.loads(events_data))
        else:
            return jsonify({"error": "File not found"}), 404
    else:
        return "Wrong Method."
if __name__ == "__main__":
    app.run("0.0.0.0", 80)
