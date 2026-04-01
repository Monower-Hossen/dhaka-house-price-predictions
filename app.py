from flask import Flask, request, render_template

# Use the full path defined in your template.py
from src.house_price_prediction.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.house_price_prediction.constants import APP_HOST, APP_PORT

application = Flask(__name__)
app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dhp_predictdata', methods=['GET', 'POST'])
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
        
        def get_float_form(field_name, default=0.0):
            """Helper to safely cast form fields to floats."""
            try:
                val = request.form.get(field_name)
                return float(val) if val and val.strip() else default
            except (ValueError, TypeError):
                return default

        # Capture ALL fields from the form
        data = CustomData(
            Location=request.form.get('Location'),
            Type=request.form.get('Type'),
            No_Beds=get_int_form('No_Beds'),
            No_Baths=get_int_form('No_Baths'),
            Area=get_float_form('Area'),
            Latitude=get_float_form('Latitude'),
            Longitude=get_float_form('Longitude'),
            Region=request.form.get('Region'),
            Sub_region=request.form.get('Sub_region')
        )

        pred_df = data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template('index.html', results=round(results[0], 2))
    
if __name__ == "__main__":
    app.run(host=APP_HOST, port=APP_PORT)