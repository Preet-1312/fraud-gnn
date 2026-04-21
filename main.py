from src.data_loader import load_data
from src.preprocess import preprocess_data
from src.train import train_model
from src.evaluate import evaluate_model

def main():
    print('Loading data...')
    data = load_data()
    
    print('Preprocessing data...')
    x, y, train_mask, test_mask = preprocess_data(data)
    
    print('Training model...')
    model = train_model(data, x, y, train_mask)
    
    print('Evaluating model...')
    evaluate_model(model, x, y, test_mask)

if __name__ == '__main__':
    main()