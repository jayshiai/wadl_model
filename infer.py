import torch
if __name__ == '__main__':
    
    model = torch.load('./chkpt/wd.pth').to('cuda')
    model.eval()
    # Create sample data
sample_data = torch.tensor([[6040,562]]).to('cuda')

# Make predictions
with torch.no_grad():
    predictions = model(sample_data)

print(predictions[0])