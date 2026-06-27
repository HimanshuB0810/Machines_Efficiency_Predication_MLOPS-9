from flask import Flask,render_template,request
import joblib
import numpy as np

app = Flask(__name__)

MODEL_PATH = "artifacts/model/model.pkl"
SCALER_PATH = "artifacts/preprocessed/scaler.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

FEATURES = ['Operation_Mode_Labels', 'Temperature_C', 'Vibration_Hz',
            'Power_Consumption_kW', 'Network_Latency_ms', 'Packet_Loss_%',
            'Quality_Control_Defect_Rate_%', 'Production_Speed_units_per_hr',
            'Predictive_Maintenance_Score', 'Error_Rate_%','Year', 'Month', 'Day', 'Hour'
            ]

Efficiency_Status_LABELS = {
    0:"High",
    1:"Low",
    2:"Medium"
}

Operation_Mode_LABELS = {
    "Active": 0,
    "Idle": 1,
    "Maintenance": 2
}

@app.route("/",methods=["GET","POST"])
def index():
    predication = None

    if request.method == "POST":
        try:
            input_data = []

            for feature in FEATURES:
                if feature == "Operation_Mode_Labels":
                    operation_mode = request.form["Operation_Mode_Labels"]
                    input_data.append(Operation_Mode_LABELS[operation_mode])
                else:
                    input_data.append(float(request.form[feature]))
            input_array = np.array(input_data).reshape(1,-1)

            scaled_array = scaler.transform(input_array)

            pred = model.predict(scaled_array)[0]
            predication = Efficiency_Status_LABELS.get(pred,"Unknown")

        except Exception as e:
            predication = f"Error: {e}"
    
    return render_template("index.html",predication=predication, features = FEATURES)

if __name__=="__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)