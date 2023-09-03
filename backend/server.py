from flask import Flask, request
from flask.helpers import send_from_directory
from flask_cors import CORS, cross_origin
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import and_
import mimetypes
import requests
import math
import random
import copy
import joblib
import pandas as pd
import torch
import torch.nn as nn
from socket import gethostname

mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("text/css", ".css")

app = Flask(__name__, static_folder="../frontend/dist", static_url_path="")
CORS(app)

# Use on pythonanywhere

SQLALCHEMY_DATABASE_URI = "mysql+mysqlconnector://{username}:{password}@{hostname}/{databasename}".format(
    username="Dospix",
    password="MySQLprojectpassword123",
    hostname="Dospix.mysql.pythonanywhere-services.com",
    databasename="Dospix$default",
)
app.config["SQLALCHEMY_DATABASE_URI"] = SQLALCHEMY_DATABASE_URI
app.config["SQLALCHEMY_POOL_RECYCLE"] = 299
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Use locally

# app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///project.db"
# app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

class Users(db.Model):
    Username = db.Column(db.String(64), primary_key=True)
    Days = db.relationship("Days", backref="users", cascade="all, delete-orphan")
    Tasks = db.relationship("Tasks", backref="users", cascade="all, delete-orphan")
    Habits = db.relationship("Habits", backref="users", cascade="all, delete-orphan")

class Days(db.Model):
    DayId = db.Column(db.Integer, primary_key=True, autoincrement=True)
    Username = db.Column(db.String(64), db.ForeignKey('users.Username'))
    DayIndex = db.Column(db.Integer)
    Tasks = db.relationship("Tasks", backref="days", cascade="all, delete-orphan")
    Habits = db.relationship("Habits", backref="days", cascade="all, delete-orphan")

class Tasks(db.Model):
    TaskId = db.Column(db.Integer, primary_key=True, autoincrement=True)
    Username = db.Column(db.String(64), db.ForeignKey('users.Username'))
    DayIndex = db.Column(db.Integer, db.ForeignKey('days.DayIndex'))
    TaskIndex = db.Column(db.Integer)
    Text = db.Column(db.String(128))
    Completed = db.Column(db.Boolean)

class Habits(db.Model):
    HabitId = db.Column(db.Integer, primary_key=True, autoincrement=True)
    Username = db.Column(db.String(64), db.ForeignKey('users.Username'))
    DayIndex = db.Column(db.Integer, db.ForeignKey('days.DayIndex'))
    HabitIndex = db.Column(db.Integer)
    Text = db.Column(db.String(128))
    Completed = db.Column(db.Boolean)


@app.route("/", defaults={"path": ""})
@app.route("/<path>")
@cross_origin()
def serve_react(path):
    return send_from_directory(app.static_folder, "index.html")

# The Google API won't allow a value bigger than 40
MAX_BOOKS_FETCHED = 40
@app.route("/google-api-project/submit", methods=["POST"])
@cross_origin()
def fetch_books_from_google_api():
    form_response_json = request.get_json()
    # "Alice Wonderland" -> "intitle:Alice+intitle:Wonderland"
    title_keywords = "+".join([f"intitle:{word}" for word in form_response_json["titleKeywords"].split()])
    author_keywords = "+".join([f"inauthor:{word}" for word in form_response_json["authorKeywords"].split()])
    subjects = "+".join([f"subject:{key}" for key in list(form_response_json.keys()) if key not in ["titleKeywords", "authorKeywords", "previewFilter"] and form_response_json[key] == True])
    preview_filter = "" if form_response_json['previewFilter'] == "none" else f"&filter={form_response_json['previewFilter']}"
    
    if author_keywords:
        title_keywords += "+"
    if subjects:
        if author_keywords:
            author_keywords += "+"
        else:
            title_keywords += "+"
    
    google_api_url_start = "https://www.googleapis.com/books/v1/volumes?q="
    gzip_headers = {
        "Accept-Encoding": "gzip",
        "User-Agent": "FlaskApp (gzip)"
    }
    # Make one request to see how many books there are that fit our search criteria, then another request that will use that number to get a random group of books
    number_of_results = requests.get(f"{google_api_url_start}{title_keywords}{author_keywords}{subjects}{preview_filter}&fields=totalItems", headers=gzip_headers).json().get("totalItems")
    if (not number_of_results) or number_of_results <= 0:
        return []
    random_index =  return_index_for_random_batch(number_of_results, MAX_BOOKS_FETCHED)

    returned_fields = "&fields=items(id, volumeInfo/title, volumeInfo/subtitle, volumeInfo/authors, volumeInfo/description, \
                    volumeInfo/imageLinks/thumbnail, volumeInfo/ratingsCount, volumeInfo/averageRating, accessInfo/viewability, volumeInfo/previewLink)"
    response = requests.get(f"{google_api_url_start}{title_keywords}{author_keywords}{subjects}{preview_filter}{returned_fields}&maxResults=40&startIndex={random_index}", headers=gzip_headers)

    response = sorted(response.json().get("items"), key= lambda book: book.get("volumeInfo").get("ratingsCount", 0), reverse= True)

    return response

def return_index_for_random_batch(total_items, items_per_batch):
    """
    Ex: There are 150 books (total_items) and 40 books are displayed at any one time (items_per_batch).
    The function would return either 0, 40, 80, or 111 randomly. The reason why is because:
    1) Indexes returned should be spread out to prevent batches with duplicates as much as possible, 
    2) Indexes returned should always result in items_per_batch(40) books. (Which is why at the end we use 111 instead of 120.)
    """

    max_batch_number = math.floor(total_items / items_per_batch)
    random_batch_selected = random.randint(0, max_batch_number)
    batch_index = random_batch_selected * items_per_batch

    if batch_index > (total_items - items_per_batch) + 1:
        batch_index = (total_items - items_per_batch) + 1

        if batch_index < 0:
            batch_index = 0
    
    return batch_index

@app.route("/used-cars-machine-learning-project/submit", methods=["POST"])
@cross_origin()
def predict_used_car_price_using_neural_network():
    form_response_json = request.get_json()
    form_response = convert_form_response_to_valid_neural_network_input(form_response_json)

    return form_response

def convert_form_response_to_valid_neural_network_input(intial_response):
    converted_response = copy.deepcopy(intial_response)

    for key in converted_response.keys():
        try:
            converted_response[key] = float(converted_response[key])
        except ValueError:
            pass
    
    with open('neural_network_input_columns.txt', 'r') as f:
        neural_network_input_columns = [line.strip() for line in f]
    
    valid_neural_network_input = dict()
    valid_neural_network_input.update([(column, False) for column in neural_network_input_columns])
    valid_neural_network_input["entry_year"] = converted_response["manufacturingYear"]
    valid_neural_network_input["condition"] = converted_response["condition"]
    valid_neural_network_input["cylinders"] = converted_response["cylinders"]
    valid_neural_network_input["odometer"] = converted_response["odometer"]
    valid_neural_network_input["vehicle_size"] = converted_response["size"]

    manufacturer_column_name = f"manufacturer_{converted_response['manufacturer']}"
    model_column_name = f"model_{converted_response['carModel']}"
    fuel_column_name = f"fuel_{converted_response['fuel']}"
    vehicle_status_column_name = f"vehicle_status_{converted_response['status']}"
    transmission_column_name = f"transmission_{converted_response['transmission']}"
    drive_column_name = f"drive_{converted_response['drive']}"
    vehicle_type_column_name = f"vehicle_type_{converted_response['type']}"
    if manufacturer_column_name in valid_neural_network_input.keys():
        valid_neural_network_input[manufacturer_column_name] = True
    if model_column_name in valid_neural_network_input.keys():
        valid_neural_network_input[model_column_name] = True
    if fuel_column_name in valid_neural_network_input.keys():
        valid_neural_network_input[fuel_column_name] = True
    if vehicle_status_column_name in valid_neural_network_input.keys():
        valid_neural_network_input[vehicle_status_column_name] = True
    if transmission_column_name in valid_neural_network_input.keys():
        valid_neural_network_input[transmission_column_name] = True
    if drive_column_name in valid_neural_network_input.keys():
        valid_neural_network_input[drive_column_name] = True
    if vehicle_type_column_name in valid_neural_network_input.keys():
        valid_neural_network_input[vehicle_type_column_name] = True
    
    for key, value in valid_neural_network_input.items():
        valid_neural_network_input[key] = [value]
    valid_neural_network_input = pd.DataFrame(valid_neural_network_input)
    scaler = joblib.load("used_cars_scaler.pkl")
    scaled_input = scaler.transform(valid_neural_network_input)

    input_tensor = torch.tensor(scaled_input, dtype=torch.float32)
    model = nn.Sequential(
        nn.Linear(2114, 235),
        nn.ReLU(),
        nn.Dropout(0.306),
        nn.Linear(235, 324),
        nn.ReLU(),
        nn.Dropout(0.265),
        nn.Linear(324, 1)
    )
    model.load_state_dict(torch.load("used_cars_price_prediction_neural_network.pt"))
    model.eval()
    with torch.no_grad():
        prediction = model(input_tensor)

    return {"price": f"{float(prediction):,.2f}"}

@app.route("/mysql-project/submit", methods=["POST"])
@cross_origin()
def change_mysql_project_user():
    form_response_json = request.get_json()
    username = form_response_json["formUsername"]
    reached_registration_limit = form_response_json["reachedRegistrationLimit"]

    has_registered = False
    if Users.query.filter_by(Username=username).count() == 0 and reached_registration_limit:
        return {
            "currUser": "",
            "hasRegistered": False
        }
    elif Users.query.filter_by(Username=username).count() == 0:
        user = Users(Username=username)
        db.session.add(user)
        db.session.commit()
        has_registered = True
    
    return {
        "currUser": username,
        "hasRegistered": has_registered
    }

if __name__ == "__main__":
    with app.app_context():
        db.create_all()

        # user = Users(Username="Ana")
        # db.session.add(user)
        # db.session.commit()
        # day = Days(Username="Ana", DayIndex=1)
        # db.session.add(day)
        # db.session.commit()
        # task = Tasks(Username="Ana", DayIndex=1, TaskIndex=1, Text="hello", Completed=False)
        # db.session.add(task)
        # db.session.commit()
        # task = Tasks(Username="Ana", DayIndex=1, TaskIndex=2, Text="hello", Completed=False)
        # db.session.add(task)
        # db.session.commit()
        # day = Days(Username="Ana", DayIndex=2)
        # db.session.add(day)
        # db.session.commit()
        # habit = Habits(Username="Ana", DayIndex=2, HabitIndex=1, Text="hello", Completed=False)
        # db.session.add(habit)
        # db.session.commit()
        # habit = Habits(Username="Ana", DayIndex=2, HabitIndex=2, Text="hello", Completed=False)
        # db.session.add(habit)
        # db.session.commit()
        # db.session.query(Users).filter(Users.Username == "Marco").delete()
        # db.session.query(Days).filter(Days.Username == "Ana").filter(Days.DayIndex == 1).delete()
        # db.session.commit()

        # task_rows = db.session.query(Users, Days, Tasks).join(Days, Users.Username == Days.Username).join(Tasks, and_(Users.Username == Tasks.Username, \
        #     Days.DayIndex == Tasks.DayIndex)).all()
        # habit_rows = db.session.query(Users, Days, Habits).join(Days, Users.Username == Days.Username).join(Habits, and_(Users.Username == Habits.Username, \
        #     Days.DayIndex == Habits.DayIndex)).all()
        # for user, day, task in task_rows:
        #     print(user.Username, day.DayIndex, task.TaskIndex)
        # for user, day, habit in habit_rows:
        #     print(user.Username, day.DayIndex, habit.HabitIndex)
    
    if "liveconsole" not in gethostname():
        app.run()
