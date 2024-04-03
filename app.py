from flask import Flask, render_template, request
import pandas as pd
from mlcode1 import mypredict  # Import function to make predictions from your ML model

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/mypredict', methods=['POST'])
def predict_route():
    if request.method == 'POST':
        # Get form data
        source = request.form['source']
        destination = request.form['destination']
        duration=request.form['duration']
        congestion = float(request.form['congestion'])
        speed = float(request.form['speed'])
        
        # Prepare input data as a DataFrame
        #input_data = pd.DataFrame([[source, target, speed, congestion_factor]],
                                #  columns=['source', 'target', 'speed', 'congestion_factor'])

        # Pass input data to ML model for prediction
        prediction = mypredict(source, destination, speed, congestion,duration)


        return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
