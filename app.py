from flask import Flask, request, render_template

# Use the full path defined in your template.py
from src.house_price_prediction.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.house_price_prediction.constants import APP_HOST, APP_PORT

application = Flask(__name__)
app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        def get_int_form(field_name, default=0):
            """Helper to safely cast form fields to integers."""
            try:
                val = request.form.get(field_name)
                return int(val) if val and val.strip() else default
            except (ValueError, TypeError):
                return default

        # Capture ALL fields from the form
        data = CustomData(
            area=get_int_form('area'),
            bedrooms=get_int_form('bedrooms'),
            bathrooms=get_int_form('bathrooms'),
            stories=get_int_form('stories'),
            mainroad=request.form.get('mainroad'),
            guestroom=request.form.get('guestroom'),
            basement=request.form.get('basement'),
            hotwaterheating=request.form.get('hotwaterheating'),
            airconditioning=request.form.get('airconditioning'),
            parking=get_int_form('parking'),
            prefarea=request.form.get('prefarea'),
            furnishingstatus=request.form.get('furnishingstatus')
        )

        pred_df = data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template('index.html', results=round(results[0], 2))
    
if __name__ == "__main__":
    app.run(host=APP_HOST, port=APP_PORT)