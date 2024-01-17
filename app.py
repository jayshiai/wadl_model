from flask import Flask, request, jsonify
import torch
app = Flask(__name__)


def get_predictions(data):
    model = torch.load('./chkpt/wd.pth').to('cuda')
    model.eval()
    # Make predictions
    with torch.no_grad():
        predictions = model(data)
    
    return predictions.tolist()

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
   