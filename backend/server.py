from flask import Flask, request
from flask.helpers import send_from_directory
from flask_cors import CORS, cross_origin
import mimetypes
import requests
import math
import random

mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("text/css", ".css")

app = Flask(__name__, static_folder="../frontend/dist", static_url_path="")
CORS(app)


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

if __name__ == "__main__":
    app.run()
