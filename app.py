from fastapi import FastAPI
import pandas as pd

app = FastAPI()


@app.get("/course_name/{course_name}")
def read_item(course_name:str):
    model = pd.read_pickle("LogisticRegressionModel.pickle")
    pred = model.predict([course_name])[0]
    return {"course_name":course_name, "keyword_prediction":pred}

# @app.route("/course/", methods=["GET"])
# def predict():
#     model = pd.read_pickle("LogisticRegressionModel.pickle")
#     course = str(request.args.get("course"))
#     prediction = model.predict([course])[0]
#     dataset = {"course_name":f"{course}", "predicted_category":f"{prediction}"}
#     return json.dumps(dataset)