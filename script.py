import joblib
from flask import Flask, request, jsonify
import pandas as pd
from google.cloud import storage

app = Flask(__name__)


@app.route('/')
def index_page():
    return 'Index page'


@app.route("/predict", methods=['POST'])
def post():
    if request.method == "POST":

        battery_level = int(request.form['battery_level'])
        battery_status = bool(request.form['battery_status'])
        device_interactive = bool(request.form['device_interactive'])
        display_state = float(request.form['display_state'])
        location_conf = float(request.form['location_conf'])
        notifs_active = float(request.form['notifs_active'])

        df = prepare_input_value(battery_level, battery_status, device_interactive,
                                 display_state, location_conf, notifs_active)

        get_model()
        prediction = run_stored_model(df)
        print(int(prediction[0] * 1000))
        response = jsonify({
            "time_prediction": int(prediction[0] * 1000)
        })
        return response
    else:
        raise RuntimeError("Weird - don't know how to handle method {}".format(request.method))


def prepare_input_value(battery_level, battery_status, device_interactive, display_state, location_conf, notifs_active):
    data = {
        'battery_level': battery_level,
        'battery_status': battery_status,
        'device_interactive': device_interactive,
        'display_state': display_state,
        'location_conf': location_conf,
        'notifs_active': notifs_active
    }
    return pd.DataFrame(data, index=[0])


def get_model():
    storage_client = storage.Client.from_service_account_json()
    bucket = storage_client.bucket("diaxytos-313718.appspot.com")

    blob = bucket.blob("model.pkl")

    with open('./downloads/model.pkl', 'wb') as file_obj:
        blob.download_to_file(file_obj)


def run_stored_model(input_df):
    model = joblib.load('./downloads/model.pkl')
    return model.predict(input_df)


if __name__ == '__main__':
    app.run()
